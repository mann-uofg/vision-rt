#pragma once

#include <cstdint>

namespace vrt {

inline constexpr std::uint8_t kPadValue = 114;

struct LetterboxGeometry {
  float scale;               // dst = src * scale
  int content_w, content_h;  // resized image size inside the dst canvas (top-left aligned)
};

// Same rule as YOLOX: scale = min(dst_h / src_h, dst_w / src_w), sizes truncated toward zero.
LetterboxGeometry letterbox_geometry(int src_w, int src_h, int dst_w, int dst_h);

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
