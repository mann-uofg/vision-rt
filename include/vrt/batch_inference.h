#pragma once

#include <cuda_runtime_api.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vrt/detection.h"
#include "vrt/nvdec_source.h"
#include "vrt/postprocess_gpu.h"
#include "vrt/preprocess_gpu.h"
#include "vrt/trt_engine.h"

namespace vrt {

// One frame in a batch: a decoded frame from some camera, plus the event that marks its
// surface copy as finished on that camera's decode stream.
struct BatchItem {
  const GpuFrame* frame;
  cudaEvent_t ready;
};

struct BatchTimes {
  int batch = 0;   // frames in the batch
  int engine = 0;  // batch size of the engine that ran it (>= batch; the rest is padding)
  float preprocess_ms = 0, infer_ms = 0, postprocess_ms = 0, total_ms = 0;
};

// Cross-camera batching: frames from different cameras go through one inference call.
//
// Takes static-batch engines of several sizes (say 1, 2, 4, 8, 16), all bound to the same
// input and output buffers. A batch of k frames runs on the smallest engine with size >= k:
// the fused letterbox kernel writes each frame into its slot of the batched input tensor, the
// engine runs from its CUDA graph, and the GPU NMS runs once per frame on its slice of the
// output, captured with the result copies as one CUDA graph per k. Each frame's NMS is its own
// branch of that graph (forked onto a side stream and joined back), so the k NMS chains run
// concurrently: each is a dozen small kernels that would leave most of the GPU idle if run one
// after another. Padding slots hold stale data; they are computed and ignored.
//
// Everything runs on one stream; run() waits for its results. Concurrency comes from the
// decoders, which keep working on their own streams while a batch runs.
class BatchInference {
 public:
  BatchInference(const std::vector<std::string>& engine_paths, cudaStream_t stream, float conf,
                 float iou, int max_det = 100);
  ~BatchInference();
  BatchInference(const BatchInference&) = delete;
  BatchInference& operator=(const BatchInference&) = delete;

  int max_batch() const { return max_batch_; }

  // Build the letterbox tables and capture the NMS graphs for every batch size now. Call it
  // before other GPU threads start (see GpuPipeline::prepare).
  void prepare(int frame_w, int frame_h);

  // Detections for each item, in order (frame pixels, highest score first).
  std::vector<Detections> run(const std::vector<BatchItem>& items, BatchTimes* times);

 private:
  struct Engine {
    int batch;
    std::unique_ptr<TrtEngine> ctx;
  };
  const Engine& pick(int k) const;
  cudaGraphExec_t post_graph(int k, float scale);

  cudaStream_t stream_;
  float conf_, iou_;
  int max_det_, max_batch_ = 0;
  std::vector<Engine> engines_;  // ascending batch size
  TensorFormat format_ = TensorFormat::kFloat16;
  std::size_t in_slot_bytes_ = 0;    // one image of input tensor
  std::size_t out_slot_floats_ = 0;  // one image of engine output
  int num_anchors_ = 0, num_classes_ = 0;
  void* d_in_ = nullptr;
  float* d_out_ = nullptr;
  std::unique_ptr<GpuLetterbox> letterbox_;
  std::vector<std::unique_ptr<GpuPostprocess>> post_;  // one per batch slot
  void* host_ = nullptr;                               // pinned: max_batch result blocks
  std::size_t result_bytes_ = 0;
  std::map<int, cudaGraphExec_t> post_graphs_;  // by batch size k
  float graph_scale_ = 0;
  std::vector<cudaStream_t> side_;  // one per batch slot, for the NMS graph branches
  std::vector<cudaEvent_t> join_;
  cudaEvent_t fork_ = nullptr;
  cudaEvent_t ev_[4] = {};
};

}  // namespace vrt
