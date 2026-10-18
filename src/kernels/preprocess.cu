#include <cuda_fp16.h>

#include <stdexcept>
#include <vector>

#include "vrt/cuda_check.h"
#include "vrt/letterbox.h"
#include "vrt/preprocess_gpu.h"

namespace vrt {

YuvCoeffs yuv_coeffs(bool bt709, bool full_range) {
  // Kr, Kb from the standards; same derivation as python/vrt/yuv.py.
  const double kr = bt709 ? 0.2126 : 0.299, kb = bt709 ? 0.0722 : 0.114, kg = 1.0 - kr - kb;
  const double ys = full_range ? 1.0 : 255.0 / 219.0;
  const double cs = full_range ? 1.0 : 255.0 / 224.0;
  return {full_range ? 0.f : 16.f,
          static_cast<float>(ys),
          static_cast<float>(2.0 * (1.0 - kr) * cs),
          static_cast<float>(-2.0 * (1.0 - kb) * kb / kg * cs),
          static_cast<float>(-2.0 * (1.0 - kr) * kr / kg * cs),
          static_cast<float>(2.0 * (1.0 - kb) * cs)};
}

namespace {

__device__ __forceinline__ float to_u8(float v) {
  return fminf(fmaxf(floorf(v + 0.5f), 0.f), 255.f);
}

// BGR (as rounded 0-255 floats) of one source pixel. 4:2:0 chroma, nearest: the UV pair at
// (x/2, y/2) covers this pixel.
__device__ __forceinline__ float3 bgr_at(const Nv12View& s, const YuvCoeffs& c, int x, int y) {
  const float yy = (static_cast<float>(s.luma[y * s.luma_pitch + x]) - c.y_off) * c.y_scale;
  const std::uint8_t* uv = s.chroma + (y >> 1) * s.chroma_pitch + (x & ~1);
  const float u = static_cast<float>(uv[0]) - 128.f;
  const float v = static_cast<float>(uv[1]) - 128.f;
  return make_float3(to_u8(yy + c.b_u * u), to_u8(yy + c.g_u * u + c.g_v * v),
                     to_u8(yy + c.r_v * v));
}

__device__ __forceinline__ float blend(float p00, float p01, float p10, float p11, float wx,
                                       float wy) {
  const float top = p00 + (p01 - p00) * wx;
  const float bot = p10 + (p11 - p10) * wx;
  // Round half up, as the CPU letterbox does (blended values are never negative).
  return floorf(top + (bot - top) * wy + 0.5f);
}

template <typename T>
__device__ __forceinline__ T convert(float v);
template <>
__device__ __forceinline__ float convert<float>(float v) {
  return v;
}
template <>
__device__ __forceinline__ __half convert<__half>(float v) {
  return __float2half_rn(v);  // exact: v is an integer 0-255
}

// One thread per output pixel; the three channel planes are written with coalesced stores.
template <typename T>
__global__ void nv12_letterbox_kernel(Nv12View src, YuvCoeffs c,
                                      const LetterboxTap* __restrict__ tx,
                                      const LetterboxTap* __restrict__ ty, int content_w,
                                      int content_h, int dst_w, int dst_h, T* __restrict__ dst) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= dst_h) return;
  const std::size_t plane = static_cast<std::size_t>(dst_w) * dst_h;
  const std::size_t o = static_cast<std::size_t>(y) * dst_w + x;

  if (x >= content_w || y >= content_h) {
    const T pad = convert<T>(static_cast<float>(kPadValue));
    dst[o] = pad;
    dst[o + plane] = pad;
    dst[o + 2 * plane] = pad;
    return;
  }

  const LetterboxTap a = tx[x], b = ty[y];
  const float3 p00 = bgr_at(src, c, a.i0, b.i0), p01 = bgr_at(src, c, a.i1, b.i0);
  const float3 p10 = bgr_at(src, c, a.i0, b.i1), p11 = bgr_at(src, c, a.i1, b.i1);
  dst[o] = convert<T>(blend(p00.x, p01.x, p10.x, p11.x, a.w1, b.w1));
  dst[o + plane] = convert<T>(blend(p00.y, p01.y, p10.y, p11.y, a.w1, b.w1));
  dst[o + 2 * plane] = convert<T>(blend(p00.z, p01.z, p10.z, p11.z, a.w1, b.w1));
}

// One thread per output pixel, BGR interleaved.
__global__ void nv12_thumbnail_kernel(Nv12View src, YuvCoeffs c, int dst_w, int dst_h,
                                      std::uint8_t* __restrict__ dst) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= dst_h) return;
  const int x0 = x * src.width / dst_w, x1 = max((x + 1) * src.width / dst_w, x0 + 1);
  const int y0 = y * src.height / dst_h, y1 = max((y + 1) * src.height / dst_h, y0 + 1);
  float3 sum = make_float3(0.f, 0.f, 0.f);
  for (int sy = y0; sy < y1; ++sy) {
    for (int sx = x0; sx < x1; ++sx) {
      const float3 p = bgr_at(src, c, sx, sy);
      sum.x += p.x;
      sum.y += p.y;
      sum.z += p.z;
    }
  }
  const float n = static_cast<float>((x1 - x0) * (y1 - y0));
  std::uint8_t* o = dst + (static_cast<std::size_t>(y) * dst_w + x) * 3;
  o[0] = static_cast<std::uint8_t>(to_u8(sum.x / n));
  o[1] = static_cast<std::uint8_t>(to_u8(sum.y / n));
  o[2] = static_cast<std::uint8_t>(to_u8(sum.z / n));
}

void* upload_taps(const std::vector<LetterboxTap>& taps) {
  void* d = nullptr;
  const std::size_t bytes = taps.size() * sizeof(LetterboxTap);
  VRT_CUDA_CHECK(cudaMalloc(&d, bytes));
  VRT_CUDA_CHECK(cudaMemcpy(d, taps.data(), bytes, cudaMemcpyHostToDevice));
  return d;
}

}  // namespace

GpuLetterbox::GpuLetterbox(int src_w, int src_h, int dst_w, int dst_h)
    : src_w_(src_w), src_h_(src_h), dst_w_(dst_w), dst_h_(dst_h) {
  if (src_w % 2 || src_h % 2) throw std::invalid_argument("NV12 needs even width and height");
  const LetterboxGeometry g = letterbox_geometry(src_w, src_h, dst_w, dst_h);
  content_w_ = g.content_w;
  content_h_ = g.content_h;
  scale_ = g.scale;
  taps_x_ = upload_taps(letterbox_taps(src_w, content_w_));
  taps_y_ = upload_taps(letterbox_taps(src_h, content_h_));
}

GpuLetterbox::~GpuLetterbox() {
  cudaFree(taps_x_);
  cudaFree(taps_y_);
}

void GpuLetterbox::run(const Nv12View& src, const YuvCoeffs& coeffs, void* dst, TensorFormat format,
                       cudaStream_t stream) const {
  if (src.width != src_w_ || src.height != src_h_)
    throw std::invalid_argument("frame size differs from the one GpuLetterbox was built for");
  const dim3 block(32, 8);
  const dim3 grid((dst_w_ + block.x - 1) / block.x, (dst_h_ + block.y - 1) / block.y);
  const auto* tx = static_cast<const LetterboxTap*>(taps_x_);
  const auto* ty = static_cast<const LetterboxTap*>(taps_y_);
  if (format == TensorFormat::kFloat16) {
    nv12_letterbox_kernel<<<grid, block, 0, stream>>>(src, coeffs, tx, ty, content_w_, content_h_,
                                                      dst_w_, dst_h_, static_cast<__half*>(dst));
  } else {
    nv12_letterbox_kernel<<<grid, block, 0, stream>>>(src, coeffs, tx, ty, content_w_, content_h_,
                                                      dst_w_, dst_h_, static_cast<float*>(dst));
  }
  VRT_CUDA_CHECK(cudaGetLastError());
}

void nv12_to_bgr_thumbnail(const Nv12View& src, const YuvCoeffs& coeffs, std::uint8_t* dst,
                           int dst_w, int dst_h, cudaStream_t stream) {
  if (dst_w < 1 || dst_h < 1 || dst_w > src.width || dst_h > src.height)
    throw std::invalid_argument("thumbnail must be no larger than the frame");
  const dim3 block(32, 8);
  const dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
  nv12_thumbnail_kernel<<<grid, block, 0, stream>>>(src, coeffs, dst_w, dst_h, dst);
  VRT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vrt
