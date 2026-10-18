#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace vrt {

// Result block written by GpuPostprocess, laid out so one small copy brings it to the host:
//   int count; int pad; float dets[max_det][6] (x1, y1, x2, y2, score, cls) in frame pixels.
struct GpuDetections {
  int count;
  int pad;
  float dets[1];  // really max_det * 6
};

std::size_t gpu_detections_bytes(int max_det);

// GPU postprocessing for one image, equivalent to vrt::postprocess() on the CPU:
//   1. score filter: per anchor, best class, score = objectness * class score, keep >= conf
//   2. sort all anchors by (score desc, anchor index asc) with a CUB radix sort on a 64-bit key
//   3. NMS: IoU bitmask of the sorted candidates (each 64-thread block compares 64 boxes against
//      64 others), then one block walks the candidates in order and keeps the survivors
//   4. scale kept boxes back to frame pixels
// Everything is queued on the caller's stream; nothing waits on the host. The candidate
// count lives on the device, so kernels launch for the worst case and exit early.
class GpuPostprocess {
 public:
  GpuPostprocess(int num_anchors, int num_classes, int max_det);
  ~GpuPostprocess();
  GpuPostprocess(const GpuPostprocess&) = delete;
  GpuPostprocess& operator=(const GpuPostprocess&) = delete;

  // preds: device [num_anchors, 5 + num_classes] float32. Output goes to output().
  void run(const float* preds, float scale, float conf_thresh, float iou_thresh,
           cudaStream_t stream);

  // Device pointer to the GpuDetections block (gpu_detections_bytes(max_det) bytes).
  const void* output() const { return out_; }
  std::size_t output_bytes() const { return gpu_detections_bytes(max_det_); }
  int max_det() const { return max_det_; }

 private:
  int num_anchors_, num_classes_, max_det_, col_blocks_;
  unsigned long long* keys_in_ = nullptr;
  unsigned long long* keys_out_ = nullptr;
  int* idx_in_ = nullptr;
  int* idx_out_ = nullptr;
  int* cls_ = nullptr;     // best class per anchor
  int* count_ = nullptr;   // number of candidates (anchors above threshold)
  float* cand_ = nullptr;  // sorted candidates: x1, y1, x2, y2 (input space)
  int* cand_cls_ = nullptr;
  float* cand_score_ = nullptr;
  unsigned long long* mask_ = nullptr;  // [num_anchors][col_blocks]
  void* sort_temp_ = nullptr;
  std::size_t sort_temp_bytes_ = 0;
  void* out_ = nullptr;
};

// The same postprocessing for a batch of images [batch, num_anchors, 5 + num_classes], with
// one launch per step instead of one per image, and results bit-identical to GpuPostprocess
// on each image:
//   1. score filter: one kernel over all images; per-image candidate counts
//   2. one radix sort over batch * num_anchors keys. The key is
//        (inverted image index : 8 bits)(score : 32)(inverted anchor index : 16),
//      so a descending sort groups each image's candidates together, in the same order the
//      single-image sort gives (score desc, anchor asc). Below-threshold anchors get key 0
//      and sort after every image's candidates. 56 key bits: one radix pass fewer than 64.
//   3. gather into per-image candidate arrays, then the IoU bitmask (blocks loop over tiles,
//      so the grid doesn't scale with the worst-case candidate count) and the walk, one block
//      per image
// Output: `batch` GpuDetections blocks back to back, each gpu_detections_bytes(max_det).
class GpuBatchPostprocess {
 public:
  GpuBatchPostprocess(int max_batch, int num_anchors, int num_classes, int max_det);
  ~GpuBatchPostprocess();
  GpuBatchPostprocess(const GpuBatchPostprocess&) = delete;
  GpuBatchPostprocess& operator=(const GpuBatchPostprocess&) = delete;

  // preds: device [batch, num_anchors, 5 + num_classes] float32; batch <= max_batch.
  void run(const float* preds, int batch, float scale, float conf_thresh, float iou_thresh,
           cudaStream_t stream);

  // Device pointer to `batch` GpuDetections blocks; block i is image i.
  const void* output() const { return out_; }
  std::size_t result_bytes() const { return gpu_detections_bytes(max_det_); }
  int max_det() const { return max_det_; }

 private:
  int max_batch_, num_anchors_, num_classes_, max_det_, col_blocks_;
  unsigned long long* keys_in_ = nullptr;  // [max_batch * num_anchors], and so on below
  unsigned long long* keys_out_ = nullptr;
  int* idx_in_ = nullptr;  // global anchor index: image * num_anchors + anchor
  int* idx_out_ = nullptr;
  int* cls_ = nullptr;
  int* counts_ = nullptr;  // [max_batch] candidates per image
  float* cand_ = nullptr;  // per image, num_anchors rows apart
  int* cand_cls_ = nullptr;
  float* cand_score_ = nullptr;
  unsigned long long* mask_ = nullptr;  // per image [num_anchors][col_blocks]
  void* sort_temp_ = nullptr;
  std::size_t sort_temp_bytes_ = 0;
  void* out_ = nullptr;
};

}  // namespace vrt
