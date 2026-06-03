#include "HailoFeatureExtractor.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgproc/imgproc.hpp>

#include "hailo/hailort.hpp"

namespace ORB_SLAM3
{

struct HailoFeatureExtractor::Impl
{
    std::unique_ptr<hailort::VDevice> vdevice;
    std::shared_ptr<hailort::InferModel> infer_model;
    std::unique_ptr<hailort::ConfiguredInferModel> configured;

    std::string input_name;
    std::vector<std::string> output_names;

    size_t input_frame_size = 0;
    std::vector<size_t> output_frame_sizes;

    std::vector<uint8_t> input_buffer;
};

HailoFeatureExtractor::HailoFeatureExtractor(const std::string& hef_path)
    : mImpl(std::make_unique<Impl>())
{
    hailo_vdevice_params_t vdevice_params = {};
    auto init_status = hailo_init_vdevice_params(&vdevice_params);
    if (HAILO_SUCCESS != init_status) {
        throw std::runtime_error("HailoFeatureExtractor: hailo_init_vdevice_params failed, status = "
                                 + std::to_string(init_status));
    }
    vdevice_params.group_id = "SHARED";
    vdevice_params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    auto vdevice_exp = hailort::VDevice::create(vdevice_params);
    if (!vdevice_exp) {
        throw std::runtime_error("HailoFeatureExtractor: VDevice::create failed, status = "
                                 + std::to_string(vdevice_exp.status()));
    }
    mImpl->vdevice = vdevice_exp.release();

    auto infer_model_exp = mImpl->vdevice->create_infer_model(hef_path);
    if (!infer_model_exp) {
        throw std::runtime_error("HailoFeatureExtractor: create_infer_model('" + hef_path +
                                 "') failed, status = " + std::to_string(infer_model_exp.status()));
    }
    mImpl->infer_model = infer_model_exp.release();
    mImpl->infer_model->set_batch_size(1);

    const auto& input_names = mImpl->infer_model->get_input_names();
    if (input_names.size() != 1) {
        throw std::runtime_error("HailoFeatureExtractor: expected exactly 1 input, got "
                                 + std::to_string(input_names.size()));
    }
    mImpl->input_name = input_names[0];
    mImpl->output_names = mImpl->infer_model->get_output_names();

    mImpl->infer_model->input(mImpl->input_name)->set_format_type(HAILO_FORMAT_TYPE_UINT8);
    for (const auto& on : mImpl->output_names) {
        mImpl->infer_model->output(on)->set_format_type(HAILO_FORMAT_TYPE_UINT8);
    }

    auto in_infos = mImpl->infer_model->hef().get_input_vstream_infos();
    if (!in_infos) {
        throw std::runtime_error("HailoFeatureExtractor: get_input_vstream_infos failed");
    }
    const auto& in_shape = in_infos.value()[0].shape;
    mInputHeight   = in_shape.height;
    mInputWidth    = in_shape.width;
    mInputChannels = in_shape.features;

    mImpl->input_frame_size = mImpl->infer_model->input(mImpl->input_name)->get_frame_size();
    mImpl->input_buffer.assign(mImpl->input_frame_size, 0);

    auto out_infos = mImpl->infer_model->hef().get_output_vstream_infos();
    if (!out_infos) {
        throw std::runtime_error("HailoFeatureExtractor: get_output_vstream_infos failed");
    }
    mOutputs.clear();
    mImpl->output_frame_sizes.clear();
    mHeatmapOutputIndex = -1;
    for (const auto& info : out_infos.value()) {
        const std::string name(info.name);
        size_t frame_size = mImpl->infer_model->output(name)->get_frame_size();
        OutputBuffer ob;
        ob.name     = name;
        ob.height   = info.shape.height;
        ob.width    = info.shape.width;
        ob.features = info.shape.features;
        ob.data.assign(frame_size, 0);
        if (mHeatmapOutputIndex < 0 &&
            ob.height == mInputHeight &&
            ob.width  == mInputWidth &&
            ob.features == 1 &&
            frame_size == mInputHeight * mInputWidth) {
            mHeatmapOutputIndex = static_cast<int>(mOutputs.size());
        }
        mOutputs.push_back(std::move(ob));
        mImpl->output_frame_sizes.push_back(frame_size);
    }

    auto configured_exp = mImpl->infer_model->configure();
    if (!configured_exp) {
        throw std::runtime_error("HailoFeatureExtractor: infer_model->configure() failed, status = "
                                 + std::to_string(configured_exp.status()));
    }
    mImpl->configured = std::make_unique<hailort::ConfiguredInferModel>(configured_exp.release());

    std::cerr << "[HailoFeatureExtractor] HEF=" << hef_path
              << " input=" << mImpl->input_name
              << " shape=" << mInputHeight << "x" << mInputWidth << "x" << mInputChannels
              << " frame_size=" << mImpl->input_frame_size << "\n";
    for (size_t i = 0; i < mOutputs.size(); ++i) {
        std::cerr << "[HailoFeatureExtractor] output[" << i << "]='" << mOutputs[i].name
                  << "' shape=" << mOutputs[i].height << "x" << mOutputs[i].width
                  << "x" << mOutputs[i].features
                  << " frame_size=" << mImpl->output_frame_sizes[i] << "\n";
    }
    if (mHeatmapOutputIndex >= 0) {
        std::cerr << "[HailoFeatureExtractor] heatmap output = '"
                  << mOutputs[mHeatmapOutputIndex].name
                  << "' (matches input " << mInputHeight << "x" << mInputWidth << ")\n";
    } else {
        std::cerr << "[HailoFeatureExtractor] WARNING: no output matches the input "
                  << mInputHeight << "x" << mInputWidth << "x1 raster; "
                     "GetHeatmap() will return empty\n";
    }
}

HailoFeatureExtractor::~HailoFeatureExtractor() = default;

bool HailoFeatureExtractor::Run(const cv::Mat& gray)
{
    if (gray.empty()) return false;

    cv::Mat single_channel;
    if (gray.channels() == 1) {
        single_channel = gray;
    } else {
        cv::cvtColor(gray, single_channel, cv::COLOR_BGR2GRAY);
    }

    cv::Mat resized;
    if (static_cast<size_t>(single_channel.rows) == mInputHeight &&
        static_cast<size_t>(single_channel.cols) == mInputWidth) {
        resized = single_channel;
    } else {
        cv::resize(single_channel, resized,
                   cv::Size(static_cast<int>(mInputWidth), static_cast<int>(mInputHeight)),
                   0, 0, cv::INTER_AREA);
    }

    if (resized.type() != CV_8UC1) {
        resized.convertTo(resized, CV_8UC1);
    }

    cv::Mat contiguous = resized.isContinuous() ? resized : resized.clone();
    const size_t expected_size = mInputHeight * mInputWidth * mInputChannels;
    if (contiguous.total() * contiguous.elemSize() < expected_size) {
        std::cerr << "[HailoFeatureExtractor] input buffer too small: have "
                  << contiguous.total() * contiguous.elemSize()
                  << " need " << expected_size << "\n";
        return false;
    }
    std::memcpy(mImpl->input_buffer.data(), contiguous.data, expected_size);

    auto bindings_exp = mImpl->configured->create_bindings();
    if (!bindings_exp) {
        std::cerr << "[HailoFeatureExtractor] create_bindings failed, status = "
                  << bindings_exp.status() << "\n";
        return false;
    }
    auto bindings = bindings_exp.release();

    auto in_status = bindings.input(mImpl->input_name)->set_buffer(
        hailort::MemoryView(mImpl->input_buffer.data(), mImpl->input_frame_size));
    if (HAILO_SUCCESS != in_status) {
        std::cerr << "[HailoFeatureExtractor] set input buffer failed, status = " << in_status << "\n";
        return false;
    }

    for (size_t i = 0; i < mOutputs.size(); ++i) {
        auto out_status = bindings.output(mOutputs[i].name)->set_buffer(
            hailort::MemoryView(mOutputs[i].data.data(), mImpl->output_frame_sizes[i]));
        if (HAILO_SUCCESS != out_status) {
            std::cerr << "[HailoFeatureExtractor] set output buffer '" << mOutputs[i].name
                      << "' failed, status = " << out_status << "\n";
            return false;
        }
    }

    auto wait_status = mImpl->configured->wait_for_async_ready(std::chrono::milliseconds(5000), 1);
    if (HAILO_SUCCESS != wait_status) {
        std::cerr << "[HailoFeatureExtractor] wait_for_async_ready failed, status = "
                  << wait_status << "\n";
        return false;
    }

    auto job_exp = mImpl->configured->run_async(bindings);
    if (!job_exp) {
        std::cerr << "[HailoFeatureExtractor] run_async failed, status = "
                  << job_exp.status() << "\n";
        return false;
    }
    auto job = job_exp.release();
    auto job_status = job.wait(std::chrono::milliseconds(5000));
    if (HAILO_SUCCESS != job_status) {
        std::cerr << "[HailoFeatureExtractor] job.wait failed, status = " << job_status << "\n";
        return false;
    }

    ++mFrameCount;
    return true;
}

cv::Mat HailoFeatureExtractor::GetHeatmap()
{
    if (mHeatmapOutputIndex < 0) return cv::Mat();
    OutputBuffer& ob = mOutputs[mHeatmapOutputIndex];
    return cv::Mat(static_cast<int>(mInputHeight),
                   static_cast<int>(mInputWidth),
                   CV_8UC1,
                   ob.data.data());
}

} // namespace ORB_SLAM3
