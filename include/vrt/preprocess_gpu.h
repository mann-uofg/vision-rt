#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace vrt {

// NV12 image in device memory: Y plane, then interleaved UV at half resolution.
struct Nv12View {
  const std::uint8_t* luma = nullptr;
  const std::uint8_t* chroma = nullptr;
  int luma_pitch = 0, chroma_pitch = 0;
  int width = 0, height = 0;
};

// YUV -> RGB matrix, folded with the range expansion.
//   Y' = (Y - y_off) * y_scale;  B = Y' + b_u*U';  G = Y' + g_u*U' + g_v*V';  R = Y' + r_v*V'
// where U' = U - 128, V' = V - 128.
struct YuvCoeffs {
  float y_off, y_scale, r_v, g_u, g_v, b_u;
};

YuvCoeffs yuv_coeffs(bool bt709, bool full_range);

enum class TensorFormat { kFloat32, kFloat16 };

// The fused preprocessing kernel: decoded NV12 frame -> network input, in one pass.
//   per output pixel: 4 bilinear taps, each converted NV12 -> BGR (nearest 4:2:0 chroma) and
//   rounded to uint8, blended, rounded again; pad 114 outside the resized image; written
//   planar CHW in BGR order as FP32 or FP16, values 0-255.
// Uses the same tap tables as the CPU letterbox (letterbox_taps), so the two differ only in
// where the BGR pixels come from.
class GpuLetterbox {
 public:
  GpuLetterbox(int src_w, int src_h, int dst_w, int dst_h);
  ~GpuLetterbox();
  GpuLetterbox(const GpuLetterbox&) = delete;
  GpuLetterbox& operator=(const GpuLetterbox&) = delete;

  void run(const Nv12View& src, const YuvCoeffs& coeffs, void* dst, TensorFormat format,
           cudaStream_t stream) const;

  float scale() const { return scale_; }
  int src_w() const { return src_w_; }
  int src_h() const { return src_h_; }

 private:
  int src_w_, src_h_, dst_w_, dst_h_, content_w_, content_h_;
  float scale_;
  void* taps_x_ = nullptr;  // LetterboxTap[content_w_]
  void* taps_y_ = nullptr;  // LetterboxTap[content_h_]
};

}  // namespace vrt
