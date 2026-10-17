#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>
#include <vector>

#include "vrt/detection.h"
#include "vrt/stats.h"
#include "vrt/trt_engine.h"
#include "vrt/video_source.h"

namespace vrt {

// Per-frame timings. Host stages are wall-clock; GPU stages come from CUDA events.
struct StageTimes {
  double preprocess_ms = 0;  // CPU letterbox (+ FP32 -> FP16 conversion for FP16 engines)
  double gpu_ms = 0;         // host wall-clock for H2D + inference + D2H, including the sync
  double postprocess_ms = 0;
  float h2d_ms = 0, infer_ms = 0, d2h_ms = 0;  // CUDA events
};

// The M1 pipeline: the naive baseline's stages with TensorRT in place of PyTorch.
//   CPU letterbox -> pageable H2D -> TensorRT -> pageable D2H -> CPU NMS, one frame at a time.
// Everything except inference stays where the baseline had it, so comparing the two
// isolates what the engine itself buys.
class SyncPipeline {
 public:
  // cuda_graph: replay the engine from a captured CUDA graph instead of calling enqueueV3.
  explicit SyncPipeline(const std::string& engine_path, bool cuda_graph = false);
  ~SyncPipeline();
  SyncPipeline(const SyncPipeline&) = delete;
  SyncPipeline& operator=(const SyncPipeline&) = delete;

  Detections run(const BgrImage& frame, float conf_thresh, float iou_thresh, StageTimes* times);

  const TrtEngine& engine() const { return engine_; }
  int input_w() const { return in_w_; }
  int input_h() const { return in_h_; }
  // Last preprocessed input as CHW float (for checking against the Python reference).
  std::vector<float> last_input() const;

 private:
  TrtEngine engine_;
  int in_w_ = 0, in_h_ = 0, num_anchors_ = 0, num_classes_ = 0;
  bool fp16_input_ = false;
  std::vector<float> chw_;
  std::vector<std::uint16_t> chw_half_;
  std::vector<float> preds_;
  void* d_in_ = nullptr;
  void* d_out_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t ev_[4] = {};
};

// GPU, driver and library versions, as fields of the current JSON object.
void write_environment(JsonWriter& j);

}  // namespace vrt
