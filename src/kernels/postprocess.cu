#include <cub/device/device_radix_sort.cuh>
#include <stdexcept>

#include "vrt/cuda_check.h"
#include "vrt/postprocess_gpu.h"

namespace vrt {

namespace {

constexpr int kBlock = 64;  // boxes per NMS tile: one bit each in a 64-bit word

// Sort key: score in the high 32 bits (positive floats compare like their bit patterns),
// inverted anchor index in the low 32, so a descending sort gives highest score first and,
// on ties, the lower anchor first. Anchors below threshold get key 0 and sort last.
__device__ __forceinline__ unsigned long long sort_key(float score, int anchor) {
  return (static_cast<unsigned long long>(__float_as_uint(score)) << 32) |
         (0xFFFFFFFFu - static_cast<unsigned>(anchor));
}

__global__ void score_kernel(const float* __restrict__ preds, int num_anchors, int num_classes,
                             float conf, unsigned long long* __restrict__ keys,
                             int* __restrict__ idx, int* __restrict__ cls,
                             int* __restrict__ count) {
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= num_anchors) return;
  const float* p = preds + static_cast<std::size_t>(a) * (5 + num_classes);
  int best = 0;
  float best_score = p[5];
  for (int c = 1; c < num_classes; ++c) {
    const float s = p[5 + c];
    if (s > best_score) {  // strict: first maximum wins, like std::max_element
      best_score = s;
      best = c;
    }
  }
  const float score = p[4] * best_score;
  const bool keep = score >= conf;
  keys[a] = keep ? sort_key(score, a) : 0ull;
  idx[a] = a;
  cls[a] = best;
  if (keep) atomicAdd(count, 1);
}

// Sorted order -> candidate arrays (boxes as corners, in network-input pixels).
__global__ void gather_kernel(const float* __restrict__ preds, int num_classes,
                              const unsigned long long* __restrict__ keys,
                              const int* __restrict__ idx, const int* __restrict__ cls,
                              const int* __restrict__ count, float* __restrict__ cand,
                              float* __restrict__ cand_score, int* __restrict__ cand_cls) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= *count) return;
  const int a = idx[i];
  const float* p = preds + static_cast<std::size_t>(a) * (5 + num_classes);
  const float hw = p[2] / 2, hh = p[3] / 2;
  cand[4 * i + 0] = p[0] - hw;
  cand[4 * i + 1] = p[1] - hh;
  cand[4 * i + 2] = p[0] + hw;
  cand[4 * i + 3] = p[1] + hh;
  cand_score[i] = __uint_as_float(static_cast<unsigned>(keys[i] >> 32));
  cand_cls[i] = cls[a];
}

__device__ __forceinline__ float iou(const float* a, const float* b) {
  // Same expression and order as vrt::iou() on the CPU.
  const float iw = fmaxf(0.f, fminf(a[2], b[2]) - fmaxf(a[0], b[0]));
  const float ih = fmaxf(0.f, fminf(a[3], b[3]) - fmaxf(a[1], b[1]));
  const float inter = iw * ih;
  const float area_a = (a[2] - a[0]) * (a[3] - a[1]);
  const float area_b = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (area_a + area_b - inter);
}

// Tile (row block r, column block c): thread t sets bit k of mask[r*64+t][c] when candidate
// c*64+k comes later in the order, has the same class, and overlaps r*64+t by more than iou.
// Only the upper triangle of tiles does work.
__global__ void nms_mask_kernel(const float* __restrict__ cand, const int* __restrict__ cand_cls,
                                const int* __restrict__ count, float iou_thresh, int col_blocks,
                                unsigned long long* __restrict__ mask) {
  const int n = *count;
  const int r = blockIdx.y, c = blockIdx.x;
  if (c < r || r * kBlock >= n || c * kBlock >= n) return;

  __shared__ float boxes[kBlock * 4];
  __shared__ int classes[kBlock];
  const int col_n = min(kBlock, n - c * kBlock);
  if (threadIdx.x < col_n) {
    const int j = c * kBlock + threadIdx.x;
    for (int k = 0; k < 4; ++k) boxes[threadIdx.x * 4 + k] = cand[4 * j + k];
    classes[threadIdx.x] = cand_cls[j];
  }
  __syncthreads();

  const int i = r * kBlock + threadIdx.x;
  if (i >= n) return;
  const float* bi = cand + 4 * i;
  const int ci = cand_cls[i];
  unsigned long long bits = 0;
  for (int k = (r == c ? threadIdx.x + 1 : 0); k < col_n; ++k) {
    if (classes[k] == ci && iou(bi, boxes + 4 * k) > iou_thresh) bits |= 1ull << k;
  }
  mask[static_cast<std::size_t>(i) * col_blocks + c] = bits;
}

// One block walks the sorted candidates. A candidate survives if no earlier survivor
// suppressed it; its mask row is then OR-ed into `removed`. This walk is the inherently
// serial part of NMS; the O(n^2) IoU work was done in parallel above.
__global__ void nms_reduce_kernel(const float* __restrict__ cand,
                                  const float* __restrict__ cand_score,
                                  const int* __restrict__ cand_cls, const int* __restrict__ count,
                                  const unsigned long long* __restrict__ mask, int col_blocks,
                                  int max_det, float scale, GpuDetections* __restrict__ out) {
  extern __shared__ unsigned long long removed[];
  const int n = *count;
  for (int w = threadIdx.x; w < col_blocks; w += blockDim.x) removed[w] = 0;
  __syncthreads();

  int kept = 0;
  for (int i = 0; i < n && kept < max_det; ++i) {
    // Every thread reads the same shared word after the last barrier, so the branch is uniform.
    if (removed[i / kBlock] & (1ull << (i % kBlock))) continue;
    if (threadIdx.x == 0) {
      float* d = out->dets + 6 * kept;
      d[0] = cand[4 * i + 0] / scale;
      d[1] = cand[4 * i + 1] / scale;
      d[2] = cand[4 * i + 2] / scale;
      d[3] = cand[4 * i + 3] / scale;
      d[4] = cand_score[i];
      d[5] = static_cast<float>(cand_cls[i]);
    }
    ++kept;
    const unsigned long long* row = mask + static_cast<std::size_t>(i) * col_blocks;
    for (int w = i / kBlock + threadIdx.x; w < col_blocks; w += blockDim.x) {
      // Tiles left of the diagonal were never written; only words at or right of i's are read.
      if (w * kBlock < n) removed[w] |= row[w];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) out->count = kept;
}

}  // namespace

std::size_t gpu_detections_bytes(int max_det) {
  return offsetof(GpuDetections, dets) + sizeof(float) * 6 * static_cast<std::size_t>(max_det);
}

GpuPostprocess::GpuPostprocess(int num_anchors, int num_classes, int max_det)
    : num_anchors_(num_anchors),
      num_classes_(num_classes),
      max_det_(max_det),
      col_blocks_((num_anchors + kBlock - 1) / kBlock) {
  const auto n = static_cast<std::size_t>(num_anchors);
  VRT_CUDA_CHECK(cudaMalloc(&keys_in_, n * sizeof(unsigned long long)));
  VRT_CUDA_CHECK(cudaMalloc(&keys_out_, n * sizeof(unsigned long long)));
  VRT_CUDA_CHECK(cudaMalloc(&idx_in_, n * sizeof(int)));
  VRT_CUDA_CHECK(cudaMalloc(&idx_out_, n * sizeof(int)));
  VRT_CUDA_CHECK(cudaMalloc(&cls_, n * sizeof(int)));
  VRT_CUDA_CHECK(cudaMalloc(&count_, sizeof(int)));
  VRT_CUDA_CHECK(cudaMalloc(&cand_, n * 4 * sizeof(float)));
  VRT_CUDA_CHECK(cudaMalloc(&cand_cls_, n * sizeof(int)));
  VRT_CUDA_CHECK(cudaMalloc(&cand_score_, n * sizeof(float)));
  VRT_CUDA_CHECK(cudaMalloc(&mask_, n * col_blocks_ * sizeof(unsigned long long)));
  VRT_CUDA_CHECK(cudaMalloc(&out_, gpu_detections_bytes(max_det)));
  // The whole block is copied to the host each frame; rows past `count` must be defined.
  VRT_CUDA_CHECK(cudaMemset(out_, 0, gpu_detections_bytes(max_det)));
  VRT_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
      nullptr, sort_temp_bytes_, keys_in_, keys_out_, idx_in_, idx_out_, num_anchors));
  VRT_CUDA_CHECK(cudaMalloc(&sort_temp_, sort_temp_bytes_));
}

GpuPostprocess::~GpuPostprocess() {
  for (void* p :
       {static_cast<void*>(keys_in_), static_cast<void*>(keys_out_), static_cast<void*>(idx_in_),
        static_cast<void*>(idx_out_), static_cast<void*>(cls_), static_cast<void*>(count_),
        static_cast<void*>(cand_), static_cast<void*>(cand_cls_), static_cast<void*>(cand_score_),
        static_cast<void*>(mask_), sort_temp_, out_})
    cudaFree(p);
}

void GpuPostprocess::run(const float* preds, float scale, float conf_thresh, float iou_thresh,
                         cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int anchor_blocks = (num_anchors_ + kThreads - 1) / kThreads;

  VRT_CUDA_CHECK(cudaMemsetAsync(count_, 0, sizeof(int), stream));
  score_kernel<<<anchor_blocks, kThreads, 0, stream>>>(
      preds, num_anchors_, num_classes_, conf_thresh, keys_in_, idx_in_, cls_, count_);
  VRT_CUDA_CHECK(cudaGetLastError());
  VRT_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(sort_temp_, sort_temp_bytes_, keys_in_,
                                                           keys_out_, idx_in_, idx_out_,
                                                           num_anchors_, 0, 64, stream));
  gather_kernel<<<anchor_blocks, kThreads, 0, stream>>>(
      preds, num_classes_, keys_out_, idx_out_, cls_, count_, cand_, cand_score_, cand_cls_);
  VRT_CUDA_CHECK(cudaGetLastError());
  nms_mask_kernel<<<dim3(col_blocks_, col_blocks_), kBlock, 0, stream>>>(
      cand_, cand_cls_, count_, iou_thresh, col_blocks_, mask_);
  VRT_CUDA_CHECK(cudaGetLastError());
  const int reduce_threads = 128;
  nms_reduce_kernel<<<1, reduce_threads, col_blocks_ * sizeof(unsigned long long), stream>>>(
      cand_, cand_score_, cand_cls_, count_, mask_, col_blocks_, max_det_, scale,
      static_cast<GpuDetections*>(out_));
  VRT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vrt
