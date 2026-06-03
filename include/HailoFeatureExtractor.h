#ifndef HAILO_FEATURE_EXTRACTOR_H
#define HAILO_FEATURE_EXTRACTOR_H

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

#include <opencv2/core/core.hpp>

namespace hailort {
    class VDevice;
    class InferModel;
    class ConfiguredInferModel;
}

namespace ORB_SLAM3
{

class HailoFeatureExtractor
{
public:
    struct OutputBuffer
    {
        std::string name;
        std::vector<uint8_t> data;
        size_t height;
        size_t width;
        size_t features;
    };

    explicit HailoFeatureExtractor(const std::string& hef_path);
    ~HailoFeatureExtractor();

    HailoFeatureExtractor(const HailoFeatureExtractor&) = delete;
    HailoFeatureExtractor& operator=(const HailoFeatureExtractor&) = delete;

    bool Run(const cv::Mat& gray);

    const std::vector<OutputBuffer>& GetOutputs() const { return mOutputs; }

    // Non-owning view over the corner-weight heatmap output ('activation1'),
    // laid out as an HxW UINT8 raster at the model's input resolution.
    // Empty if the model has no NHWC HxWx1 output matching the input size.
    cv::Mat GetHeatmap();

    // Non-owning view over the downscaled-image output ('resize1'),
    // laid out as an HxW UINT8 raster at the next pyramid level's resolution.
    // Empty if the model has no UINT8 1-feature output smaller than the input.
    cv::Mat GetResize();

    size_t GetInputHeight() const { return mInputHeight; }
    size_t GetInputWidth()  const { return mInputWidth; }

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;

    std::vector<OutputBuffer> mOutputs;
    size_t mInputHeight  = 0;
    size_t mInputWidth   = 0;
    size_t mInputChannels = 0;
    int mHeatmapOutputIndex = -1;
    int mResizeOutputIndex  = -1;
    long long mFrameCount = 0;
};

} // namespace ORB_SLAM3

#endif // HAILO_FEATURE_EXTRACTOR_H
