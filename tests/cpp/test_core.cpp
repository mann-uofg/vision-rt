// Unit tests for the CPU core. Plain asserts, no framework: run by CTest.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "vrt/half.h"
#include "vrt/letterbox.h"
#include "vrt/postprocess.h"
#include "vrt/stats.h"

namespace {

int failures = 0;

#define CHECK(cond)                                                                 \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                                   \
    }                                                                               \
  } while (0)

bool near(double a, double b, double tol = 1e-6) { return std::fabs(a - b) <= tol; }

void test_geometry() {
  const auto g = vrt::letterbox_geometry(1920, 1080, 640, 640);
  CHECK(g.content_w == 640 && g.content_h == 360);
  CHECK(near(g.scale, 1.0 / 3.0));
  const auto g2 = vrt::letterbox_geometry(517, 333, 640, 640);  // upscale, width-bound
  CHECK(g2.content_w == 640 && g2.content_h == static_cast<int>(333 * (640.0 / 517)));
}

void test_letterbox_padding_and_order() {
  // Uniform 64x32 image: B=10, G=20, R=200. Resized to 64x32 inside a 64x64 canvas.
  const int w = 64, h = 32;
  std::vector<std::uint8_t> img(static_cast<std::size_t>(w * h * 3));
  for (int i = 0; i < w * h; ++i) {
    img[static_cast<std::size_t>(3 * i)] = 10;
    img[static_cast<std::size_t>(3 * i + 1)] = 20;
    img[static_cast<std::size_t>(3 * i + 2)] = 200;
  }
  std::vector<float> out(3 * 64 * 64);
  vrt::letterbox_bgr_to_chw(img.data(), w, h, w * 3, 64, 64, out.data());
  const std::size_t plane = 64 * 64;
  CHECK(out[0] == 10 && out[plane] == 20 && out[2 * plane] == 200);  // top-left: image
  CHECK(out[31 * 64 + 63] == 10);                                    // last image row
  CHECK(out[32 * 64] == vrt::kPadValue && out[2 * plane + 63 * 64 + 63] == vrt::kPadValue);
}

void test_letterbox_identity() {
  // Same size in and out: bilinear taps land exactly on pixels, output equals input.
  const int w = 8, h = 8;
  std::vector<std::uint8_t> img(static_cast<std::size_t>(w * h * 3));
  for (std::size_t i = 0; i < img.size(); ++i) img[i] = static_cast<std::uint8_t>(i * 7 % 251);
  std::vector<float> out(3 * 8 * 8);
  vrt::letterbox_bgr_to_chw(img.data(), w, h, w * 3, 8, 8, out.data());
  bool same = true;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < 3; ++c)
        same &= out[static_cast<std::size_t>(c * 64 + y * 8 + x)] ==
                img[static_cast<std::size_t>((y * w + x) * 3 + c)];
  CHECK(same);
}

void test_half() {
  CHECK(vrt::float_to_half(0.f) == 0x0000);
  CHECK(vrt::float_to_half(1.f) == 0x3C00);
  CHECK(vrt::float_to_half(-2.f) == 0xC000);
  CHECK(vrt::float_to_half(255.f) == 0x5BF8);
  CHECK(vrt::float_to_half(65504.f) == 0x7BFF);
  CHECK(vrt::float_to_half(1e6f) == 0x7C00);              // overflow -> inf
  CHECK(vrt::float_to_half(5.9604645e-8f) == 0x0001);     // smallest subnormal
  CHECK(vrt::float_to_half(1.f + 1.f / 2048) == 0x3C00);  // halfway, rounds to even
  CHECK(vrt::float_to_half(1.f + 3.f / 2048) == 0x3C02);  // halfway, rounds to even (up)
  for (int i = 0; i <= 255; ++i) CHECK(vrt::half_to_float(vrt::float_to_half(float(i))) == i);
}

std::vector<float> anchor(float cx, float cy, float w, float h, float obj, int cls, float p) {
  std::vector<float> a(85, 0.f);
  a[0] = cx, a[1] = cy, a[2] = w, a[3] = h, a[4] = obj;
  a[static_cast<std::size_t>(5 + cls)] = p;
  return a;
}

void test_postprocess() {
  std::vector<float> preds;
  for (const auto& a : {anchor(100, 100, 50, 50, 0.9f, 3, 1.f),   // kept
                        anchor(101, 101, 50, 50, 0.8f, 3, 1.f),   // same class, overlaps: dropped
                        anchor(100, 100, 50, 50, 0.7f, 7, 1.f),   // other class: kept
                        anchor(300, 300, 50, 50, 0.1f, 1, 1.f)})  // below threshold
    preds.insert(preds.end(), a.begin(), a.end());
  const auto dets = vrt::postprocess(preds.data(), 4, 80, 0.5f, 0.3f, 0.45f);
  CHECK(dets.size() == 2);
  CHECK(dets[0].cls == 3 && near(dets[0].score, 0.9, 1e-6));
  CHECK(near(dets[0].x1, 150) && near(dets[0].y2, 250));  // (75..125) / 0.5
  CHECK(dets[1].cls == 7);
}

void test_percentile() {
  CHECK(near(vrt::percentile({1, 2, 3, 4}, 50), 2.5));
  CHECK(near(vrt::percentile({1, 2, 3, 4}, 99), 3.97));
  CHECK(near(vrt::percentile({5}, 99), 5));
}

}  // namespace

int main() {
  test_geometry();
  test_letterbox_padding_and_order();
  test_letterbox_identity();
  test_half();
  test_postprocess();
  test_percentile();
  if (failures) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return EXIT_FAILURE;
  }
  std::puts("all core tests passed");
  return EXIT_SUCCESS;
}
