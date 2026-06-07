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

    // Bindings + job for the most-recent async submission. Must outlive the
    // inference; replaced (and the previous one freed) when the next
    // submission happens — by which time the previous inference's callback
    // has already fired and returned, so it's safe to drop.
    std::unique_ptr<hailort::ConfiguredInferModel::Bindings> current_bindings;
    hailort::AsyncInferJob last_job;
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
    mResizeOutputIndex  = -1;
    for (const auto& info : out_infos.value()) {
        const std::string name(info.name);
        size_t frame_size = mImpl->infer_model->output(name)->get_frame_size();
        OutputBuffer ob;
        ob.name     = name;
        ob.height   = info.shape.height;
        ob.width    = info.shape.width;
        ob.features = info.shape.features;
        ob.data.assign(frame_size, 0);
        const bool is_uint8_one_channel =
            ob.features == 1 && frame_size == ob.height * ob.width;
        if (mHeatmapOutputIndex < 0 &&
            is_uint8_one_channel &&
            ob.height == mInputHeight &&
            ob.width  == mInputWidth) {
            mHeatmapOutputIndex = static_cast<int>(mOutputs.size());
        } else if (mResizeOutputIndex < 0 &&
                   is_uint8_one_channel &&
                   (ob.height != mInputHeight || ob.width != mInputWidth)) {
            mResizeOutputIndex = static_cast<int>(mOutputs.size());
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
    if (mResizeOutputIndex >= 0) {
        const auto& ob = mOutputs[mResizeOutputIndex];
        std::cerr << "[HailoFeatureExtractor] resize output  = '" << ob.name
                  << "' (" << ob.height << "x" << ob.width << ")\n";
    }
}

HailoFeatureExtractor::~HailoFeatureExtractor() = default;

bool HailoFeatureExtractor::LoadInput(const cv::Mat& gray)
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
    return true;
}

bool HailoFeatureExtractor::SubmitWithCurrentInput()
{
    // Build a fresh Bindings object; previous one (if any) is dropped here,
    // which is safe because its inference has already completed and its
    // completion callback returned (that callback is what scheduled us).
    auto bindings_exp = mImpl->configured->create_bindings();
    if (!bindings_exp) {
        std::cerr << "[HailoFeatureExtractor] create_bindings failed, status = "
                  << bindings_exp.status() << "\n";
        return false;
    }
    auto new_bindings = std::make_unique<hailort::ConfiguredInferModel::Bindings>(
        bindings_exp.release());

    auto in_status = new_bindings->input(mImpl->input_name)->set_buffer(
        hailort::MemoryView(mImpl->input_buffer.data(), mImpl->input_frame_size));
    if (HAILO_SUCCESS != in_status) {
        std::cerr << "[HailoFeatureExtractor] set input buffer failed, status = " << in_status << "\n";
        return false;
    }
    for (size_t i = 0; i < mOutputs.size(); ++i) {
        auto out_status = new_bindings->output(mOutputs[i].name)->set_buffer(
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

    auto job_exp = mImpl->configured->run_async(
        *new_bindings,
        [this](const hailort::AsyncInferCompletionInfo& info) {
            HandleInferenceComplete(info.status);
        });
    if (!job_exp) {
        std::cerr << "[HailoFeatureExtractor] run_async failed, status = "
                  << job_exp.status() << "\n";
        return false;
    }

    mImpl->current_bindings = std::move(new_bindings);
    auto job = job_exp.release();
    job.detach();
    mImpl->last_job = std::move(job);
    return true;
}

void HailoFeatureExtractor::HandleInferenceComplete(int status)
{
    if (HAILO_SUCCESS != status) {
        std::cerr << "[HailoFeatureExtractor] async inference failed, status = "
                  << status << "\n";
        mChainOk.store(false, std::memory_order_relaxed);
    }

    // Submit the next stage FIRST, so its NPU work overlaps with this stage's
    // CPU-side OnComplete callback below. The next stage's input is just this
    // stage's resize1 output buffer; we memcpy it in.
    if (mNext && mChainOk.load(std::memory_order_relaxed)) {
        if (mResizeOutputIndex < 0) {
            std::cerr << "[HailoFeatureExtractor] chain: no resize output to feed next stage\n";
            mChainOk.store(false, std::memory_order_relaxed);
        } else {
            const auto& resize_ob = mOutputs[mResizeOutputIndex];
            const size_t need = mNext->mImpl->input_frame_size;
            const size_t have = resize_ob.data.size();
            if (have < need) {
                std::cerr << "[HailoFeatureExtractor] chain: resize output ("
                          << have << "B) smaller than next input (" << need << "B)\n";
                mChainOk.store(false, std::memory_order_relaxed);
            } else {
                std::memcpy(mNext->mImpl->input_buffer.data(),
                            resize_ob.data.data(),
                            need);
                if (!mNext->SubmitWithCurrentInput()) {
                    mChainOk.store(false, std::memory_order_relaxed);
                }
            }
        }
    }

    // Run this stage's user callback (keypoint extraction, mvImagePyramid
    // populate, ...). Runs in parallel with the NPU work for the next stage.
    if (mOnComplete) {
        try { mOnComplete(); }
        catch (const std::exception& e) {
            std::cerr << "[HailoFeatureExtractor] OnComplete threw: " << e.what() << "\n";
            mChainOk.store(false, std::memory_order_relaxed);
        }
    }

    // Signal completion for THIS stage. WaitForChain walks the chain and
    // waits for every stage's mDone, so the main thread only resumes after
    // every callback has returned (no race on shared per-stage state).
    {
        std::lock_guard<std::mutex> lock(mDoneMu);
        mDone = true;
        mDoneCv.notify_all();
    }
    ++mFrameCount;
}

void HailoFeatureExtractor::SetNext(HailoFeatureExtractor* next)
{
    mNext = next;
}

void HailoFeatureExtractor::SetOnComplete(CompletionCallback cb)
{
    mOnComplete = std::move(cb);
}

bool HailoFeatureExtractor::SubmitChain(const cv::Mat& input)
{
    if (!LoadInput(input)) return false;

    // Propagate "we're starting; nobody has signalled yet" to every stage of
    // the chain reachable from here. Whichever stage doesn't submit a next
    // stage (i.e., the last one to complete on this frame) will be the one to
    // signal — and from this stage's WaitForChain perspective that's the only
    // one we care about, but we initialize them all defensively in case the
    // chain breaks midway.
    for (HailoFeatureExtractor* s = this; s; s = s->mNext) {
        std::lock_guard<std::mutex> lock(s->mDoneMu);
        s->mDone = false;
        s->mChainOk.store(true, std::memory_order_relaxed);
    }

    return SubmitWithCurrentInput();
}

bool HailoFeatureExtractor::WaitForChain(int timeout_ms)
{
    // Walk the chain — wait for every stage's mDone. This ensures every
    // stage's user callback has fully returned before the caller can read
    // any shared per-frame state (allKeypoints, mvImagePyramid, ...).
    for (HailoFeatureExtractor* s = this; s; s = s->mNext) {
        std::unique_lock<std::mutex> lock(s->mDoneMu);
        if (!s->mDoneCv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                  [s] { return s->mDone; })) {
            return false;
        }
    }
    return mChainOk.load(std::memory_order_relaxed);
}

bool HailoFeatureExtractor::Run(const cv::Mat& gray)
{
    // Legacy sync API: submit + block. Chain disabled.
    auto saved_next = mNext;
    auto saved_cb   = std::move(mOnComplete);
    mNext = nullptr;
    mOnComplete = nullptr;
    bool ok = SubmitChain(gray) && WaitForChain(5000);
    mNext = saved_next;
    mOnComplete = std::move(saved_cb);
    return ok;
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

cv::Mat HailoFeatureExtractor::GetResize()
{
    if (mResizeOutputIndex < 0) return cv::Mat();
    OutputBuffer& ob = mOutputs[mResizeOutputIndex];
    return cv::Mat(static_cast<int>(ob.height),
                   static_cast<int>(ob.width),
                   CV_8UC1,
                   ob.data.data());
}

} // namespace ORB_SLAM3
