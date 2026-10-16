// C entry points into the CPU core, loaded from Python with ctypes so the tests can compare
// the C++ code against the Python reference (tests/test_cpp_core.py).

#include <cstdint>

#include "vrt/letterbox.h"
#include "vrt/postprocess.h"

extern "C" {

void vrt_letterbox(const std::uint8_t* src, int src_w, int src_h, int src_stride, int dst_w,
                   int dst_h, float* dst) {
  vrt::letterbox_bgr_to_chw(src, src_w, src_h, src_stride, dst_w, dst_h, dst);
}

void vrt_letterbox_half(const std::uint8_t* src, int src_w, int src_h, int src_stride, int dst_w,
                        int dst_h, std::uint16_t* dst) {
  vrt::letterbox_bgr_to_chw_half(src, src_w, src_h, src_stride, dst_w, dst_h, dst);
}

// Writes up to max_out detections as rows of 6 floats (x1, y1, x2, y2, score, cls).
// Returns how many there were in total (may exceed max_out).
int vrt_postprocess(const float* preds, int num_anchors, int num_classes, float scale,
                    float conf_thresh, float iou_thresh, float* out, int max_out) {
  const vrt::Detections dets =
      vrt::postprocess(preds, num_anchors, num_classes, scale, conf_thresh, iou_thresh);
  const int n = static_cast<int>(dets.size());
  for (int i = 0; i < n && i < max_out; ++i) {
    const auto& d = dets[static_cast<std::size_t>(i)];
    float* row = out + 6 * i;
    row[0] = d.x1;
    row[1] = d.y1;
    row[2] = d.x2;
    row[3] = d.y2;
    row[4] = d.score;
    row[5] = static_cast<float>(d.cls);
  }
  return n;
}

}  // extern "C"
