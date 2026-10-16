#pragma once

#include "vrt/detection.h"

namespace vrt {

// Thresholds, same as python/vrt/postprocess.py.
inline constexpr float kEvalConf = 0.01f, kEvalNmsIou = 0.65f;
inline constexpr float kDeployConf = 0.3f, kDeployNmsIou = 0.45f;

// CPU postprocessing for one image, the C++ twin of python/vrt/postprocess.py.
// preds: [num_anchors, 5 + num_classes] floats (cx, cy, w, h in input pixels, objectness,
// class scores). Keeps anchors with objectness * best class score >= conf_thresh, runs greedy
// class-aware NMS in input space, then divides boxes by `scale` to get frame pixels.
// Output is sorted by score, highest first.
Detections postprocess(const float* preds, int num_anchors, int num_classes, float scale,
                       float conf_thresh, float iou_thresh);

// Plain greedy NMS over boxes that are already sorted by descending score. Boxes of different
// classes never suppress each other. Returns kept indices in input order.
std::vector<int> nms_sorted(const Detections& sorted, float iou_thresh);

float iou(const Detection& a, const Detection& b);

}  // namespace vrt
