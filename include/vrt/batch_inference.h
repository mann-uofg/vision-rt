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
// engine runs from its CUDA graph, and the batched GPU NMS (GpuBatchPostprocess) runs over the
// first k slots of the output as one CUDA graph per k: a handful of launches for the whole
// batch, not a dozen per frame. One copy brings all k result blocks to the host. Padding slots
// hold stale data; the engine computes them and NMS skips them.
//
// Everything runs on one stream, so consecutive batches can share every device buffer: batch
// n+1's kernels are queued behind batch n's result copy. Only the pinned result blocks and the
// timing events are per batch, in `sets` result sets, so up to `sets` batches can be in flight:
// submit() queues a batch and returns at once; wait() blocks for its results. With two sets the
// caller queues batch n+1 before waiting on batch n, and the stream never drains while the host
// unpacks results and gathers the next batch. run() is submit() + wait().
class BatchInference {
 public:
  BatchInference(const std::vector<std::string>& engine_paths, cudaStream_t stream, float conf,
                 float iou, int max_det = 100, int sets = 2);
  ~BatchInference();
  BatchInference(const BatchInference&) = delete;
  BatchInference& operator=(const BatchInference&) = delete;

  int max_batch() const { return max_batch_; }
  int sets() const { return static_cast<int>(sets_.size()); }

  // Build the letterbox tables and capture the NMS graphs for every batch size now. Call it
  // before other GPU threads start (see GpuPipeline::prepare).
  void prepare(int frame_w, int frame_h);

  // Queue a batch. Returns a ticket for wait(); throws if all result sets are still waited on.
  int submit(const std::vector<BatchItem>& items);
  // Detections for each item of that batch, in order (frame pixels, highest score first).
  std::vector<Detections> wait(int ticket, BatchTimes* times);
  std::vector<Detections> run(const std::vector<BatchItem>& items, BatchTimes* times) {
    return wait(submit(items), times);
  }

 private:
  struct Engine {
    int batch;
    std::unique_ptr<TrtEngine> ctx;
  };
  struct ResultSet {
    void* host = nullptr;  // pinned: max_batch result blocks
    cudaEvent_t ev[4] = {};
    int batch = 0, engine = 0;
    bool pending = false;
  };
  const Engine& pick(int k) const;
  cudaGraphExec_t post_graph(int k, float scale);

  cudaStream_t stream_;
  float conf_, iou_;
  int max_det_, max_batch_ = 0;
  std::vector<Engine> engines_;  // ascending batch size
  TensorFormat format_ = TensorFormat::kFloat16;
  std::size_t in_slot_bytes_ = 0;  // one image of input tensor
  int num_anchors_ = 0, num_classes_ = 0;
  void* d_in_ = nullptr;
  float* d_out_ = nullptr;
  std::unique_ptr<GpuLetterbox> letterbox_;
  std::unique_ptr<GpuBatchPostprocess> post_;
  std::size_t result_bytes_ = 0;
  std::map<int, cudaGraphExec_t> post_graphs_;  // by batch size k
  float graph_scale_ = 0;
  std::vector<ResultSet> sets_;
  int next_set_ = 0;
};

}  // namespace vrt
