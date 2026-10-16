#pragma once

#include <vector>

namespace vrt {

// One detected box in original-frame pixels.
struct Detection {
  float x1, y1, x2, y2;
  float score;  // objectness * best class score
  int cls;      // 0..79, COCO class index (not category id)
};

using Detections = std::vector<Detection>;

}  // namespace vrt
