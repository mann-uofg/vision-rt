// Run a TensorRT engine over COCO val2017 images through the C++ pipeline and write raw
// detections as CSV, for tools/eval_coco.py --detections to score.
//
//   vrt_coco --engine models/yolox_s_fp16.engine --images data/coco/val2017
//            --out bench/results/detections/cpp_yolox_s_fp16.csv [--limit N]
//
// image_id comes from the file name (COCO names are the zero-padded id). Classes are written
// as indices 0..79; the Python side maps them to COCO category ids.

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include "args.h"
#include "vrt/postprocess.h"
#include "vrt/sync_pipeline.h"
#include "vrt/video_source.h"

namespace fs = std::filesystem;

namespace {

int run(const vrt::Args& args) {
  const std::string engine_path = args.required("engine");
  const fs::path dir = args.required("images");
  const long long limit = args.integer("limit", -1);
  const float conf = args.real("conf", vrt::kEvalConf);
  const float nms = args.real("nms", vrt::kEvalNmsIou);

  std::vector<fs::path> images;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.path().extension() == ".jpg") images.push_back(e.path());
  std::sort(images.begin(), images.end());  // zero-padded names: sorts by id
  if (limit >= 0 && static_cast<std::size_t>(limit) < images.size())
    images.resize(static_cast<std::size_t>(limit));

  vrt::SyncPipeline pipe(engine_path);
  std::ofstream out(args.required("out"));
  out << "image_id,x1,y1,x2,y2,score,cls\n";
  char line[160];

  vrt::BgrImage img;
  for (std::size_t n = 0; n < images.size(); ++n) {
    const long long image_id = std::stoll(images[n].stem().string());
    vrt::VideoSource src(images[n].string(), 1);
    if (!src.read(img)) throw std::runtime_error("cannot decode " + images[n].string());
    for (const auto& d : pipe.run(img, conf, nms, nullptr)) {
      std::snprintf(line, sizeof line, "%lld,%.3f,%.3f,%.3f,%.3f,%.6f,%d\n", image_id, d.x1, d.y1,
                    d.x2, d.y2, d.score, d.cls);
      out << line;
    }
    if ((n + 1) % 1000 == 0) std::printf("%zu/%zu images\n", n + 1, images.size());
  }
  std::printf("%zu images -> %s\n", images.size(), args.str("out").c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(vrt::Args(argc, argv));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vrt_coco: %s\n", e.what());
    return 1;
  }
}
