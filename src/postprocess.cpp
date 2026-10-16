#include "vrt/postprocess.h"

#include <algorithm>

namespace vrt {

float iou(const Detection& a, const Detection& b) {
  const float iw = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const float ih = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float inter = iw * ih;
  const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (area_a + area_b - inter);
}

std::vector<int> nms_sorted(const Detections& sorted, float iou_thresh) {
  std::vector<int> keep;
  std::vector<char> suppressed(sorted.size(), 0);
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    if (suppressed[i]) continue;
    keep.push_back(static_cast<int>(i));
    for (std::size_t j = i + 1; j < sorted.size(); ++j) {
      if (!suppressed[j] && sorted[j].cls == sorted[i].cls &&
          iou(sorted[i], sorted[j]) > iou_thresh) {
        suppressed[j] = 1;
      }
    }
  }
  return keep;
}

Detections postprocess(const float* preds, int num_anchors, int num_classes, float scale,
                       float conf_thresh, float iou_thresh) {
  const int stride = 5 + num_classes;
  Detections cand;
  for (int a = 0; a < num_anchors; ++a) {
    const float* p = preds + static_cast<std::ptrdiff_t>(a) * stride;
    const float* cls = p + 5;
    const int best = static_cast<int>(std::max_element(cls, cls + num_classes) - cls);
    const float score = p[4] * cls[best];
    if (score < conf_thresh) continue;
    const float hw = p[2] / 2, hh = p[3] / 2;
    cand.push_back({p[0] - hw, p[1] - hh, p[0] + hw, p[1] + hh, score, best});
  }
  // Stable, so equal scores keep anchor order (lower index first), like the reference.
  std::stable_sort(cand.begin(), cand.end(),
                   [](const Detection& a, const Detection& b) { return a.score > b.score; });

  Detections out;
  for (int i : nms_sorted(cand, iou_thresh)) {
    Detection d = cand[static_cast<std::size_t>(i)];
    d.x1 /= scale;
    d.y1 /= scale;
    d.x2 /= scale;
    d.y2 /= scale;
    out.push_back(d);
  }
  return out;
}

}  // namespace vrt
