#ifndef HAILO_FEATURE_EXTRACTOR_H
#define HAILO_FEATURE_EXTRACTOR_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

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

    // ---- Async chain API ----
    //
    // To run several extractors as a pipeline, do this once (e.g. in your
    // owner's ctor):
    //
    //   stage[0].SetNext(&stage[1]);  stage[1].SetNext(&stage[2]);  ...
    //
    // Then per frame:
    //
    //   stage[N].SetOnComplete([] { ...read heatmap N, do CPU work... });
    //   stage[0].SubmitChain(image_for_stage_0);
    //   // ... do other CPU work while the NPU pipeline executes ...
    //   stage[last].WaitForChain();
    //
    // After Submit, each stage's HailoRT completion callback (a) copies its
    // resize1 output into the next stage's input buffer and submits it
    // asynchronously, then (b) invokes the user-provided OnComplete callback
    // for *this* stage. The last stage signals WaitForChain when its
    // OnComplete returns.

    void SetNext(HailoFeatureExtractor* next);

    using CompletionCallback = std::function<void()>;
    void SetOnComplete(CompletionCallback cb);

    bool SubmitChain(const cv::Mat& input);

    // Wait until this stage's last submitted inference (and its OnComplete)
    // has finished. Returns false on timeout / failure.
    bool WaitForChain(int timeout_ms = 30000);

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

    // Chain state
    HailoFeatureExtractor* mNext = nullptr;
    CompletionCallback     mOnComplete;
    std::mutex             mDoneMu;
    std::condition_variable mDoneCv;
    bool                   mDone = true;   // true = no submission in flight
    std::atomic<bool>      mChainOk{true}; // false if any stage's submission failed

    // Called from the HailoRT completion thread.
    void HandleInferenceComplete(int status);

    // Helpers that operate on data already loaded into mImpl->input_buffer.
    bool LoadInput(const cv::Mat& gray);
    bool SubmitWithCurrentInput();
};

} // namespace ORB_SLAM3

#endif // HAILO_FEATURE_EXTRACTOR_H
