// Run a TensorRT engine over a video with per-stage timing (M1: synchronous, CPU pre/post).
//
//   vrt_detect --engine models/yolox_s_fp16.engine --video data/footage/barn022.mp4
//              [--warmup 100] [--frames N] [--conf 0.3] [--nms 0.45]
//              [--json out.json] [--dets out.csv] [--dump-input FRAME:path.bin]
//              [--cuda-graph]
//
// Output schema matches tools/baseline.py so the two can sit in one table.

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>

#include "args.h"
#include "vrt/nvtx.h"
#include "vrt/postprocess.h"
#include "vrt/stats.h"
#include "vrt/sync_pipeline.h"
#include "vrt/video_source.h"

namespace {

std::string stem(const std::string& path) {
  const auto slash = path.find_last_of('/');
  std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
  return name.substr(0, name.find_last_of('.'));
}

std::string basename(const std::string& path) {
  const auto slash = path.find_last_of('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

int run(const vrt::Args& args) {
  using clock = std::chrono::steady_clock;
  const std::string engine_path = args.required("engine");
  const std::string video_path = args.required("video");
  const long long warmup = args.integer("warmup", 100);
  const long long max_frames = args.integer("frames", -1);
  const float conf = args.real("conf", vrt::kDeployConf);
  const float nms = args.real("nms", vrt::kDeployNmsIou);

  // "--dump-input 150:path" writes the preprocessed tensor of frame 150 as raw float32 CHW.
  long long dump_frame = -1;
  std::string dump_path;
  if (args.has("dump-input")) {
    const std::string v = args.str("dump-input");
    dump_frame = std::stoll(v.substr(0, v.find(':')));
    dump_path = v.substr(v.find(':') + 1);
  }

  const bool cuda_graph = args.has("cuda-graph");
  vrt::SyncPipeline pipe(engine_path, cuda_graph);
  vrt::VideoSource video(video_path);
  const vrt::VideoInfo& vi = video.info();

  std::ofstream dets_csv;
  if (args.has("dets")) {
    dets_csv.open(args.str("dets"));
    dets_csv << "frame,x1,y1,x2,y2,score,cls\n";
  }

  std::vector<double> decode, pre, gpu, post, e2e, h2d, infer, d2h;
  std::vector<double> ndets;
  vrt::BgrImage frame;
  vrt::StageTimes st;
  for (long long i = 0; max_frames < 0 || i < warmup + max_frames; ++i) {
    const auto t0 = clock::now();
    vrt::NvtxRange frame_range("frame");
    nvtxRangePushA("decode (cpu)");
    const bool got = video.read(frame);
    nvtxRangePop();
    if (!got) break;
    const double decode_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    const vrt::Detections dets = pipe.run(frame, conf, nms, &st);
    const double e2e_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();

    if (i == dump_frame) {
      const std::vector<float> input = pipe.last_input();
      std::ofstream f(dump_path, std::ios::binary);
      f.write(reinterpret_cast<const char*>(input.data()),
              static_cast<std::streamsize>(input.size() * sizeof(float)));
    }
    if (i < warmup) continue;
    decode.push_back(decode_ms);
    pre.push_back(st.preprocess_ms);
    gpu.push_back(st.gpu_ms);
    post.push_back(st.postprocess_ms);
    e2e.push_back(e2e_ms);
    h2d.push_back(st.h2d_ms);
    infer.push_back(st.infer_ms);
    d2h.push_back(st.d2h_ms);
    ndets.push_back(static_cast<double>(dets.size()));
    if (dets_csv) {
      for (const auto& d : dets) {
        dets_csv << i << ',' << d.x1 << ',' << d.y1 << ',' << d.x2 << ',' << d.y2 << ',' << d.score
                 << ',' << d.cls << '\n';
      }
    }
  }
  if (e2e.empty()) throw std::runtime_error("no frames timed; video shorter than --warmup?");

  const std::string name = stem(engine_path);  // yolox_s_fp16
  const std::string model = name.substr(0, name.find_last_of('_'));
  const std::string precision = name.substr(name.find_last_of('_') + 1);
  const vrt::Summary e2e_s = vrt::summarize(e2e);

  vrt::JsonWriter j;
  j.begin_object()
      .field("pipeline", cuda_graph ? "cpp-trt-sync-graph" : "cpp-trt-sync")
      .field("cuda_graph", cuda_graph)
      .field("model", model)
      .field("precision", precision)
      .field("engine", basename(engine_path));
  j.field("batch", 1).field("streams", 1).field("conf_thresh", conf).field("nms_iou", nms);
  j.begin_object("video")
      .field("file", basename(video_path))
      .field("width", vi.width)
      .field("height", vi.height)
      .field("fps", vi.fps)
      .field("frames", vi.frames)
      .field("codec", vi.codec)
      .field("color_space", vi.color_space)
      .field("color_range", vi.color_range)
      .end_object();
  j.field("warmup_frames", warmup).field("timed_frames", static_cast<long long>(e2e.size()));
  j.field("throughput_fps", 1000.0 / e2e_s.mean);
  j.begin_object("host_ms")
      .field("decode", vrt::summarize(decode))
      .field("preprocess", vrt::summarize(pre))
      .field("gpu", vrt::summarize(gpu))
      .field("postprocess", vrt::summarize(post))
      .field("e2e", e2e_s)
      .end_object();
  j.begin_object("gpu_ms")
      .field("h2d", vrt::summarize(h2d))
      .field("infer", vrt::summarize(infer))
      .field("d2h", vrt::summarize(d2h))
      .end_object();
  j.field("detections_per_frame",
          std::accumulate(ndets.begin(), ndets.end(), 0.0) / static_cast<double>(ndets.size()));
  j.begin_object("environment");
  vrt::write_environment(j);
  j.end_object().end_object();

  if (args.has("json")) {
    std::ofstream(args.str("json")) << j.str();
  }

  const vrt::Summary s_dec = vrt::summarize(decode), s_pre = vrt::summarize(pre),
                     s_gpu = vrt::summarize(gpu), s_post = vrt::summarize(post),
                     s_h2d = vrt::summarize(h2d), s_inf = vrt::summarize(infer),
                     s_d2h = vrt::summarize(d2h);
  std::printf("%s on %s: %zu frames timed, %dx%d %s\n", name.c_str(), basename(video_path).c_str(),
              e2e.size(), vi.width, vi.height, vi.codec.c_str());
  std::printf("%-14s%9s%9s%9s  (ms)\n", "stage", "mean", "p50", "p99");
  const std::pair<const char*, const vrt::Summary*> rows[] = {
      {"host decode", &s_dec},    {"host preproc", &s_pre}, {"host gpu", &s_gpu},
      {"host postproc", &s_post}, {"host e2e", &e2e_s},     {"  gpu h2d", &s_h2d},
      {"  gpu infer", &s_inf},    {"  gpu d2h", &s_d2h}};
  for (const auto& [label, s] : rows)
    std::printf("%-14s%9.2f%9.2f%9.2f\n", label, s->mean, s->p50, s->p99);
  std::printf("throughput %.1f fps\n", 1000.0 / e2e_s.mean);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(vrt::Args(argc, argv));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vrt_detect: %s\n", e.what());
    return 1;
  }
}
