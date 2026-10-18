#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

#include "vrt/detection.h"
#include "vrt/nvdec_source.h"
#include "vrt/postprocess_gpu.h"
#include "vrt/preprocess_gpu.h"
#include "vrt/trt_engine.h"

namespace vrt {

// GPU time of each stage for one frame, from CUDA events recorded on the stream.
struct GpuStageTimes {
  float preprocess_ms = 0, infer_ms = 0, postprocess_ms = 0, d2h_ms = 0;
  float total_ms = 0;  // first kernel start to result copy done
};

// The M2 pipeline. A decoded NV12 frame in GPU memory goes through
//   fused letterbox kernel -> TensorRT (CUDA graph) -> GPU score filter + NMS
//   -> one small copy of the results into pinned host memory,
// all queued on one stream with no host synchronisation in between. Only the final
// detections (a few hundred bytes) ever cross PCIe.
//
// Results go to one of kSlots pinned buffers. submit() returns as soon as the work is queued,
// so the caller can decode the next frame while the GPU runs this one, then collect() the
// previous slot. One stream keeps frames in order on the GPU, so the device buffers (input
// tensor, engine output, NMS scratch) can be shared; only the host-side result buffers need
// one copy per frame in flight.
class GpuPipeline {
 public:
  static constexpr int kSlots = 2;

  GpuPipeline(const std::string& engine_path, cudaStream_t stream, int max_det = 100);
  // Shares an already-loaded model: this pipeline gets its own execution context only.
  GpuPipeline(std::shared_ptr<TrtModel> model, cudaStream_t stream, int max_det = 100);
  ~GpuPipeline();
  GpuPipeline(const GpuPipeline&) = delete;
  GpuPipeline& operator=(const GpuPipeline&) = delete;

  // Build everything that depends on the frame size and thresholds (letterbox tap tables, the
  // NMS CUDA graph) now, instead of on the first frame. Call it before starting other threads
  // that use the GPU: capturing a graph while another thread does something that synchronises
  // the device invalidates the capture.
  void prepare(int frame_w, int frame_h, float conf_thresh, float iou_thresh);

  // Queue all GPU work for `frame`; results will land in `slot`. The frame must stay alive
  // until collect(slot) returns.
  void submit(const GpuFrame& frame, int slot, float conf_thresh, float iou_thresh);

  // Wait for `slot`, then return its detections (frame pixels, highest score first).
  Detections collect(int slot, GpuStageTimes* times);

  // The input tensor as last written by the preprocessing kernel, as float CHW. Synchronises.
  std::vector<float> input_tensor();

  const TrtEngine& engine() const { return engine_; }
  cudaStream_t stream() const { return stream_; }

 private:
  void init(int max_det);

  // Postprocessing is ~12 small launches (memset, score, 4 CUB sort kernels, gather, mask,
  // walk), and under WSL each launch costs more than the kernel. All its arguments are fixed
  // for a given video and thresholds, so it is captured into a CUDA graph on first use and
  // recaptured only if the scale or thresholds change.
  void run_postprocess(float scale, float conf_thresh, float iou_thresh);
  void ensure_post_graph(float scale, float conf_thresh, float iou_thresh);

  struct Slot {
    void* host = nullptr;    // pinned GpuDetections block
    cudaEvent_t ev[5] = {};  // begin, preprocessed, inferred, postprocessed, copied
    float scale = 1;
  };

  TrtEngine engine_;
  cudaStream_t stream_;
  int in_w_ = 0, in_h_ = 0;
  TensorFormat format_ = TensorFormat::kFloat32;
  void* d_in_ = nullptr;
  void* d_out_ = nullptr;
  std::unique_ptr<GpuLetterbox> letterbox_;
  std::unique_ptr<GpuPostprocess> post_;
  cudaGraphExec_t post_graph_ = nullptr;
  float graph_scale_ = 0, graph_conf_ = 0, graph_iou_ = 0;
  Slot slots_[kSlots];
};

}  // namespace vrt
