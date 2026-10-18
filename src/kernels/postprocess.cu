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

// Batched key (see GpuBatchPostprocess): inverted image in bits 48-55, score in 16-47,
// inverted anchor in 0-15. Within one image it orders exactly like sort_key. With
// image < 255 and anchor < 0xFFFF it is never 0, so below-threshold anchors still sort last.
constexpr int kKeyBits = 56;
__device__ __forceinline__ unsigned long long batch_sort_key(float score, int image, int anchor) {
  return (static_cast<unsigned long long>(0xFFu - static_cast<unsigned>(image)) << 48) |
         (static_cast<unsigned long long>(__float_as_uint(score)) << 16) |
         (0xFFFFu - static_cast<unsigned>(anchor));
}
__device__ __forceinline__ int key_image(unsigned long long key) {
  return 0xFF - static_cast<int>(key >> 48);
}

// Best class of one anchor's row; returns its index, writes its score.
__device__ __forceinline__ int best_class(const float* p, int num_classes, float* best_score) {
  int best = 0;
  float s_best = p[5];
  for (int c = 1; c < num_classes; ++c) {
    const float s = p[5 + c];
    if (s > s_best) {  // strict: first maximum wins, like std::max_element
      s_best = s;
      best = c;
    }
  }
  *best_score = s_best;
  return best;
}

// Corners of anchor row p into cand[0..3] (network-input pixels).
__device__ __forceinline__ void write_corners(const float* p, float* cand) {
  const float hw = p[2] / 2, hh = p[3] / 2;
  cand[0] = p[0] - hw;
  cand[1] = p[1] - hh;
  cand[2] = p[0] + hw;
  cand[3] = p[1] + hh;
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

// Tile (row block r, column block c) of one image's mask, with r <= c and both blocks holding
// candidates: thread t sets bit k of mask[r*64+t][c] when candidate c*64+k comes later in the
// order, has the same class, and overlaps r*64+t by more than iou. `boxes` and `classes` are
// the block's shared staging arrays; the caller syncs before reusing them.
__device__ __forceinline__ void nms_mask_tile(const float* __restrict__ cand,
                                              const int* __restrict__ cand_cls, int n,
                                              float iou_thresh, int col_blocks, int r, int c,
                                              unsigned long long* __restrict__ mask, float* boxes,
                                              int* classes) {
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

// One block walks one image's sorted candidates. A candidate survives if no earlier survivor
// suppressed it; its mask row is then OR-ed into `removed` (shared, col_blocks words). This
// walk is the inherently serial part of NMS; the O(n^2) IoU work was done in parallel above.
__device__ __forceinline__ void nms_walk(const float* __restrict__ cand,
                                         const float* __restrict__ cand_score,
                                         const int* __restrict__ cand_cls, int n,
                                         const unsigned long long* __restrict__ mask,
                                         int col_blocks, int max_det, float scale,
                                         GpuDetections* __restrict__ out,
                                         unsigned long long* removed) {
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
    // Everyone has read removed[i / kBlock] before anyone ORs into it. Row i only sets bits
    // after i, so the value read wouldn't change, but without this it is a data race
    // (compute-sanitizer racecheck flags it). At most max_det extra barriers.
    __syncthreads();
    const unsigned long long* row = mask + static_cast<std::size_t>(i) * col_blocks;
    for (int w = i / kBlock + threadIdx.x; w < col_blocks; w += blockDim.x) {
      // Tiles left of the diagonal were never written; only words at or right of i's are read.
      if (w * kBlock < n) removed[w] |= row[w];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) out->count = kept;
}

// --- one image -----------------------------------------------------------------------------

__global__ void score_kernel(const float* __restrict__ preds, int num_anchors, int num_classes,
                             float conf, unsigned long long* __restrict__ keys,
                             int* __restrict__ idx, int* __restrict__ cls,
                             int* __restrict__ count) {
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= num_anchors) return;
  const float* p = preds + static_cast<std::size_t>(a) * (5 + num_classes);
  float best_score;
  const int best = best_class(p, num_classes, &best_score);
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
  write_corners(preds + static_cast<std::size_t>(a) * (5 + num_classes), cand + 4 * i);
  cand_score[i] = __uint_as_float(static_cast<unsigned>(keys[i] >> 32));
  cand_cls[i] = cls[a];
}

// One block per tile; only the upper triangle of tiles does work.
__global__ void nms_mask_kernel(const float* __restrict__ cand, const int* __restrict__ cand_cls,
                                const int* __restrict__ count, float iou_thresh, int col_blocks,
                                unsigned long long* __restrict__ mask) {
  const int n = *count;
  const int r = blockIdx.y, c = blockIdx.x;
  if (c < r || r * kBlock >= n || c * kBlock >= n) return;
  __shared__ float boxes[kBlock * 4];
  __shared__ int classes[kBlock];
  nms_mask_tile(cand, cand_cls, n, iou_thresh, col_blocks, r, c, mask, boxes, classes);
}

__global__ void nms_reduce_kernel(const float* __restrict__ cand,
                                  const float* __restrict__ cand_score,
                                  const int* __restrict__ cand_cls, const int* __restrict__ count,
                                  const unsigned long long* __restrict__ mask, int col_blocks,
                                  int max_det, float scale, GpuDetections* __restrict__ out) {
  extern __shared__ unsigned long long removed[];
  nms_walk(cand, cand_score, cand_cls, *count, mask, col_blocks, max_det, scale, out, removed);
}

// --- a batch of images ---------------------------------------------------------------------
// Image b's anchors are rows [b * num_anchors, (b + 1) * num_anchors) of every per-anchor
// array; its candidates are the first counts[b] rows of its slice of the candidate arrays.

// Grid (anchor blocks, batch).
__global__ void batch_score_kernel(const float* __restrict__ preds, int num_anchors,
                                   int num_classes, float conf,
                                   unsigned long long* __restrict__ keys, int* __restrict__ idx,
                                   int* __restrict__ cls, int* __restrict__ counts) {
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= num_anchors) return;
  const int b = blockIdx.y;
  const int g = b * num_anchors + a;
  const float* p = preds + static_cast<std::size_t>(g) * (5 + num_classes);
  float best_score;
  const int best = best_class(p, num_classes, &best_score);
  const float score = p[4] * best_score;
  const bool keep = score >= conf;
  keys[g] = keep ? batch_sort_key(score, b, a) : 0ull;
  idx[g] = g;
  cls[g] = best;
  if (keep) atomicAdd(counts + b, 1);
}

// Sorted position i -> (image, rank within the image). The sort put image 0's candidates
// first, then image 1's, and so on, so image b's start is the sum of the earlier counts.
__global__ void batch_gather_kernel(const float* __restrict__ preds, int num_anchors,
                                    int num_classes, int batch,
                                    const unsigned long long* __restrict__ keys,
                                    const int* __restrict__ idx, const int* __restrict__ cls,
                                    const int* __restrict__ counts, float* __restrict__ cand,
                                    float* __restrict__ cand_score, int* __restrict__ cand_cls) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = 0;
  for (int b = 0; b < batch; ++b) total += counts[b];
  if (i >= total) return;
  const unsigned long long key = keys[i];
  const int b = key_image(key);
  int start = 0;
  for (int c = 0; c < b; ++c) start += counts[c];
  const int row = b * num_anchors + (i - start);
  const int g = idx[i];
  write_corners(preds + static_cast<std::size_t>(g) * (5 + num_classes), cand + 4 * row);
  cand_score[row] = __uint_as_float(static_cast<unsigned>(key >> 16));
  cand_cls[row] = cls[g];
}

// Grid (blocks per image, batch). The grid is fixed when the graph is captured but the
// candidate counts are only known on the device, so each block strides over its image's
// tiles; launching one block per worst-case tile would be batch * 132^2 mostly empty blocks.
__global__ void batch_nms_mask_kernel(const float* __restrict__ cand,
                                      const int* __restrict__ cand_cls,
                                      const int* __restrict__ counts, int num_anchors,
                                      float iou_thresh, int col_blocks,
                                      unsigned long long* __restrict__ mask) {
  const int b = blockIdx.y;
  const int n = counts[b];
  const int nb = (n + kBlock - 1) / kBlock;
  const auto base = static_cast<std::size_t>(b) * num_anchors;
  __shared__ float boxes[kBlock * 4];
  __shared__ int classes[kBlock];
  for (int t = blockIdx.x; t < nb * nb; t += gridDim.x) {  // uniform across the block
    const int r = t / nb, c = t % nb;
    if (c < r) continue;
    nms_mask_tile(cand + 4 * base, cand_cls + base, n, iou_thresh, col_blocks, r, c,
                  mask + base * col_blocks, boxes, classes);
    __syncthreads();  // before the next tile overwrites boxes/classes
  }
}

// One block per image.
__global__ void batch_nms_walk_kernel(const float* __restrict__ cand,
                                      const float* __restrict__ cand_score,
                                      const int* __restrict__ cand_cls,
                                      const int* __restrict__ counts, int num_anchors,
                                      const unsigned long long* __restrict__ mask, int col_blocks,
                                      int max_det, float scale, std::size_t result_bytes,
                                      void* __restrict__ out) {
  extern __shared__ unsigned long long removed[];
  const int b = blockIdx.x;
  const auto base = static_cast<std::size_t>(b) * num_anchors;
  nms_walk(cand + 4 * base, cand_score + base, cand_cls + base, counts[b], mask + base * col_blocks,
           col_blocks, max_det, scale,
           reinterpret_cast<GpuDetections*>(static_cast<char*>(out) + b * result_bytes), removed);
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

GpuBatchPostprocess::GpuBatchPostprocess(int max_batch, int num_anchors, int num_classes,
                                         int max_det)
    : max_batch_(max_batch),
      num_anchors_(num_anchors),
      num_classes_(num_classes),
      max_det_(max_det),
      col_blocks_((num_anchors + kBlock - 1) / kBlock) {
  // The key has 8 bits of image and 16 of anchor (see batch_sort_key).
  if (max_batch < 1 || max_batch > 254 || num_anchors < 1 || num_anchors > 0xFFFF)
    throw std::invalid_argument("GpuBatchPostprocess: batch must be 1-254, anchors 1-65535");
  const auto n = static_cast<std::size_t>(max_batch) * static_cast<std::size_t>(num_anchors);
  VRT_CUDA_CHECK(cudaMalloc(&keys_in_, n * sizeof(unsigned long long)));
  VRT_CUDA_CHECK(cudaMalloc(&keys_out_, n * sizeof(unsigned long long)));
  VRT_CUDA_CHECK(cudaMalloc(&idx_in_, n * sizeof(int)));
  VRT_CUDA_CHECK(cudaMalloc(&idx_out_, n * sizeof(int)));
  VRT_CUDA_CHECK(cudaMalloc(&cls_, n * sizeof(int)));
  VRT_CUDA_CHECK(cudaMalloc(&counts_, static_cast<std::size_t>(max_batch) * sizeof(int)));
  VRT_CUDA_CHECK(cudaMalloc(&cand_, n * 4 * sizeof(float)));
  VRT_CUDA_CHECK(cudaMalloc(&cand_cls_, n * sizeof(int)));
  VRT_CUDA_CHECK(cudaMalloc(&cand_score_, n * sizeof(float)));
  VRT_CUDA_CHECK(cudaMalloc(&mask_, n * col_blocks_ * sizeof(unsigned long long)));
  const std::size_t out_bytes = result_bytes() * static_cast<std::size_t>(max_batch);
  VRT_CUDA_CHECK(cudaMalloc(&out_, out_bytes));
  VRT_CUDA_CHECK(cudaMemset(out_, 0, out_bytes));  // rows past each count must be defined
  VRT_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(nullptr, sort_temp_bytes_, keys_in_,
                                                           keys_out_, idx_in_, idx_out_,
                                                           static_cast<int>(n), 0, kKeyBits));
  VRT_CUDA_CHECK(cudaMalloc(&sort_temp_, sort_temp_bytes_));
}

GpuBatchPostprocess::~GpuBatchPostprocess() {
  for (void* p :
       {static_cast<void*>(keys_in_), static_cast<void*>(keys_out_), static_cast<void*>(idx_in_),
        static_cast<void*>(idx_out_), static_cast<void*>(cls_), static_cast<void*>(counts_),
        static_cast<void*>(cand_), static_cast<void*>(cand_cls_), static_cast<void*>(cand_score_),
        static_cast<void*>(mask_), sort_temp_, out_})
    cudaFree(p);
}

void GpuBatchPostprocess::run(const float* preds, int batch, float scale, float conf_thresh,
                              float iou_thresh, cudaStream_t stream) {
  if (batch < 1 || batch > max_batch_) throw std::invalid_argument("batch out of range");
  constexpr int kThreads = 256;
  constexpr int kMaskBlocksPerImage = 32;  // plenty for the ~100s of candidates at conf 0.3
  const int anchor_blocks = (num_anchors_ + kThreads - 1) / kThreads;
  const int n = batch * num_anchors_;

  VRT_CUDA_CHECK(
      cudaMemsetAsync(counts_, 0, static_cast<std::size_t>(batch) * sizeof(int), stream));
  batch_score_kernel<<<dim3(anchor_blocks, batch), kThreads, 0, stream>>>(
      preds, num_anchors_, num_classes_, conf_thresh, keys_in_, idx_in_, cls_, counts_);
  VRT_CUDA_CHECK(cudaGetLastError());
  VRT_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(sort_temp_, sort_temp_bytes_, keys_in_,
                                                           keys_out_, idx_in_, idx_out_, n, 0,
                                                           kKeyBits, stream));
  batch_gather_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      preds, num_anchors_, num_classes_, batch, keys_out_, idx_out_, cls_, counts_, cand_,
      cand_score_, cand_cls_);
  VRT_CUDA_CHECK(cudaGetLastError());
  batch_nms_mask_kernel<<<dim3(kMaskBlocksPerImage, batch), kBlock, 0, stream>>>(
      cand_, cand_cls_, counts_, num_anchors_, iou_thresh, col_blocks_, mask_);
  VRT_CUDA_CHECK(cudaGetLastError());
  const int walk_threads = 128;
  batch_nms_walk_kernel<<<batch, walk_threads, col_blocks_ * sizeof(unsigned long long), stream>>>(
      cand_, cand_score_, cand_cls_, counts_, num_anchors_, mask_, col_blocks_, max_det_, scale,
      result_bytes(), out_);
  VRT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vrt
