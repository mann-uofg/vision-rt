#include "vrt/letterbox.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "vrt/half.h"

namespace vrt {

LetterboxGeometry letterbox_geometry(int src_w, int src_h, int dst_w, int dst_h) {
  // Computed in double like Python, so the truncated sizes agree exactly.
  const double r = std::min(static_cast<double>(dst_h) / src_h, static_cast<double>(dst_w) / src_w);
  return {static_cast<float>(r), static_cast<int>(src_w * r), static_cast<int>(src_h * r)};
}

std::vector<LetterboxTap> letterbox_taps(int src_size, int dst_size) {
  std::vector<LetterboxTap> taps(static_cast<std::size_t>(dst_size));
  const double ratio = static_cast<double>(src_size) / dst_size;
  for (int d = 0; d < dst_size; ++d) {
    const double f = (d + 0.5) * ratio - 0.5;
    int i0 = static_cast<int>(std::floor(f));
    float w1 = static_cast<float>(f - i0);
    if (i0 < 0) {
      i0 = 0;
      w1 = 0.f;
    }
    if (i0 >= src_size - 1) {
      i0 = src_size - 1;
      w1 = 0.f;
    }
    taps[static_cast<std::size_t>(d)] = {i0, std::min(i0 + 1, src_size - 1), w1};
  }
  return taps;
}

namespace {

// Shared body: resizes into the top-left of the canvas and pads the rest. Every output value
// is an integer 0-255; `convert` turns it into the output type.
template <typename T, typename Convert>
void letterbox_impl(const std::uint8_t* src, int src_w, int src_h, int src_stride, int dst_w,
                    int dst_h, T* dst, Convert convert) {
  const LetterboxGeometry g = letterbox_geometry(src_w, src_h, dst_w, dst_h);
  const std::size_t plane = static_cast<std::size_t>(dst_w) * dst_h;
  const T pad = convert(kPadValue);

  const std::vector<LetterboxTap> xs = letterbox_taps(src_w, g.content_w);
  const std::vector<LetterboxTap> ys = letterbox_taps(src_h, g.content_h);

  for (int c = 0; c < 3; ++c) {
    T* p = dst + c * plane;
    for (int y = 0; y < g.content_h; ++y)  // right-hand pad of each content row
      std::fill(p + static_cast<std::size_t>(y) * dst_w + g.content_w,
                p + static_cast<std::size_t>(y + 1) * dst_w, pad);
    std::fill(p + static_cast<std::size_t>(g.content_h) * dst_w, p + plane, pad);  // bottom pad
  }

  for (int y = 0; y < g.content_h; ++y) {
    const LetterboxTap ty = ys[static_cast<std::size_t>(y)];
    const std::uint8_t* row0 = src + static_cast<std::ptrdiff_t>(ty.i0) * src_stride;
    const std::uint8_t* row1 = src + static_cast<std::ptrdiff_t>(ty.i1) * src_stride;
    T* out_b = dst + static_cast<std::size_t>(y) * dst_w;
    T* out_g = out_b + plane;
    T* out_r = out_g + plane;
    for (int x = 0; x < g.content_w; ++x) {
      const LetterboxTap tx = xs[static_cast<std::size_t>(x)];
      const std::uint8_t* p00 = row0 + tx.i0 * 3;
      const std::uint8_t* p01 = row0 + tx.i1 * 3;
      const std::uint8_t* p10 = row1 + tx.i0 * 3;
      const std::uint8_t* p11 = row1 + tx.i1 * 3;
      std::uint8_t v[3];
      for (int k = 0; k < 3; ++k) {
        const float top = p00[k] + (p01[k] - p00[k]) * tx.w1;
        const float bot = p10[k] + (p11[k] - p10[k]) * tx.w1;
        // Bilinear of uint8 values stays in [0, 255]; round half up like cv2's fixed point.
        v[k] = static_cast<std::uint8_t>(top + (bot - top) * ty.w1 + 0.5f);
      }
      out_b[x] = convert(v[0]);
      out_g[x] = convert(v[1]);
      out_r[x] = convert(v[2]);
    }
  }
}

}  // namespace

void letterbox_bgr_to_chw(const std::uint8_t* src, int src_w, int src_h, int src_stride, int dst_w,
                          int dst_h, float* dst) {
  letterbox_impl(src, src_w, src_h, src_stride, dst_w, dst_h, dst,
                 [](std::uint8_t v) { return static_cast<float>(v); });
}

void letterbox_bgr_to_chw_half(const std::uint8_t* src, int src_w, int src_h, int src_stride,
                               int dst_w, int dst_h, std::uint16_t* dst) {
  // Every output value is an integer 0-255, so the FP16 bit patterns fit in a table.
  static const std::array<std::uint16_t, 256> lut = [] {
    std::array<std::uint16_t, 256> t{};
    for (int i = 0; i < 256; ++i) t[static_cast<std::size_t>(i)] = float_to_half(float(i));
    return t;
  }();
  letterbox_impl(src, src_w, src_h, src_stride, dst_w, dst_h, dst,
                 [](std::uint8_t v) { return lut[v]; });
}

}  // namespace vrt
