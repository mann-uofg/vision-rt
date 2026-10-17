#pragma once

#include <cstdint>
#include <vector>

namespace vrt {

inline constexpr std::uint8_t kPadValue = 114;

struct LetterboxGeometry {
  float scale;               // dst = src * scale
  int content_w, content_h;  // resized image size inside the dst canvas (top-left aligned)
};

// Same rule as YOLOX: scale = min(dst_h / src_h, dst_w / src_w), sizes truncated toward zero.
LetterboxGeometry letterbox_geometry(int src_w, int src_h, int dst_w, int dst_h);

// Bilinear source taps for each destination index along one axis, with OpenCV's mapping:
// src = (dst + 0.5) * (src_size / dst_size) - 0.5, clamped to the image. The CPU letterbox and
// the GPU kernel both use these tables, so they sample exactly the same pixels and weights.
struct LetterboxTap {
  int i0, i1;
  float w1;  // weight of i1; i0 gets 1 - w1
};
std::vector<LetterboxTap> letterbox_taps(int src_size, int dst_size);

// CPU letterbox, the C++ twin of python/vrt/preprocess.py:
//   interleaved BGR uint8 (row stride in bytes) -> planar CHW float, values 0-255, BGR order,
//   bilinear resize with half-pixel centres, image at the top-left, pad 114 elsewhere.
// Each resized pixel is rounded to uint8 before being widened, like cv2.resize on uint8
// input, so values match the Python reference to within 1.
void letterbox_bgr_to_chw(const std::uint8_t* src, int src_w, int src_h, int src_stride, int dst_w,
                          int dst_h, float* dst);

// Same, writing IEEE half-precision bit patterns (for FP16 engines).
void letterbox_bgr_to_chw_half(const std::uint8_t* src, int src_w, int src_h, int src_stride,
                               int dst_w, int dst_h, std::uint16_t* dst);

}  // namespace vrt
