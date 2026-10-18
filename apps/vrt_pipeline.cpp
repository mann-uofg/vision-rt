// The M2 GPU pipeline over a video: NVDEC decode, fused preprocessing kernel, TensorRT from a
// CUDA graph, GPU NMS, results copied to pinned memory.
//
//   vrt_pipeline --engine models/yolox_s_fp16.engine --video data/footage/barn022.mp4
//                [--mode pipelined|sync] [--warmup 100] [--frames N] [--conf 0.3] [--nms 0.45]
//                [--max-det 100] [--json out.json] [--dets out.csv]
//                [--dump-input FRAME:path.bin]
//
// pipelined: decode frame n+1 on the CPU/NVDEC while the GPU runs frame n, then collect n.
//            Best throughput; each frame's latency includes the next frame's decode.
// sync:      decode, run, collect, one frame at a time. Lowest per-frame latency.
//
// Latency is measured per frame from the start of its decode to its detections being in
// host memory. Throughput is timed frames / wall time over the timed frames.

#include <chrono>
#include <cstdio>
#include <fstream>
#include <numeric>

#include "args.h"
#include "vrt/cuda_check.h"
#include "vrt/gpu_pipeline.h"
#include "vrt/nvdec_source.h"
#include "vrt/nvtx.h"
#include "vrt/postprocess.h"
#include "vrt/stats.h"
#include "vrt/sync_pipeline.h"

namespace {

using clock_type = std::chrono::steady_clock;

double ms_between(clock_type::time_point a, clock_type::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

std::string basename(const std::string& path) {
  const auto slash = path.find_last_of('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

struct Record {
  std::vector<double> decode, latency;
  std::vector<double> pre, infer, post, d2h, gpu_total;
  std::vector<double> ndets;
};

int run(const vrt::Args& args) {
  const std::string engine_path = args.required("engine");
  const std::string video_path = args.required("video");
  const std::string mode = args.str("mode", "pipelined");
  if (mode != "pipelined" && mode != "sync") throw std::invalid_argument("--mode pipelined|sync");
  const long long warmup = args.integer("warmup", 100);
  const long long max_frames = args.integer("frames", -1);
  const float conf = args.real("conf", vrt::kDeployConf);
  const float nms = args.real("nms", vrt::kDeployNmsIou);
  const int max_det = static_cast<int>(args.integer("max-det", 100));
  long long dump_frame = -1;
  std::string dump_path;
  if (args.has("dump-input")) {
    const std::string v = args.str("dump-input");
    dump_frame = std::stoll(v.substr(0, v.find(':')));
    dump_path = v.substr(v.find(':') + 1);
  }

  // Declared first so it outlives the pipeline and the decoder, which both use it on teardown.
  vrt::CudaStream stream;
  vrt::GpuPipeline pipe(engine_path, stream, max_det);
  vrt::NvdecSource video(video_path, stream, vrt::GpuPipeline::kSlots);
  const vrt::VideoInfo& vi = video.info();
  pipe.prepare(vi.width, vi.height, conf, nms);

  std::ofstream dets_csv;
  if (args.has("dets")) {
    dets_csv.open(args.str("dets"));
    dets_csv << "frame,x1,y1,x2,y2,score,cls\n";
  }

  Record rec;
  vrt::GpuFrame frames[vrt::GpuPipeline::kSlots];
  clock_type::time_point started[vrt::GpuPipeline::kSlots];
  long long index[vrt::GpuPipeline::kSlots] = {};
  clock_type::time_point timed_begin, timed_end;
  double cpu_begin = 0, cpu_end = 0;

  // Collect slot `s`: wait for its results, record times, release its frame.
  auto finish = [&](int s) {
    vrt::GpuStageTimes t;
    const vrt::Detections dets = pipe.collect(s, &t);
    const auto now = clock_type::now();
    frames[s].release();
    const long long i = index[s];
    if (i < warmup) return;
    rec.latency.push_back(ms_between(started[s], now));
    rec.pre.push_back(t.preprocess_ms);
    rec.infer.push_back(t.infer_ms);
    rec.post.push_back(t.postprocess_ms);
    rec.d2h.push_back(t.d2h_ms);
    rec.gpu_total.push_back(t.total_ms);
    rec.ndets.push_back(static_cast<double>(dets.size()));
    timed_end = now;
    cpu_end = vrt::process_cpu_ms();
    if (dets_csv) {
      for (const auto& d : dets)
        dets_csv << i << ',' << d.x1 << ',' << d.y1 << ',' << d.x2 << ',' << d.y2 << ',' << d.score
                 << ',' << d.cls << '\n';
    }
  };

  int pending = -1;  // slot submitted but not yet collected (pipelined mode)
  for (long long i = 0; max_frames < 0 || i < warmup + max_frames; ++i) {
    const int s = static_cast<int>(i % vrt::GpuPipeline::kSlots);
    const auto t0 = clock_type::now();
    if (i == warmup) {
      timed_begin = t0;
      cpu_begin = vrt::process_cpu_ms();
    }
    nvtxRangePushA("decode (nvdec)");
    const bool got = video.read(frames[s]);
    nvtxRangePop();
    if (!got) break;
    if (i >= warmup) rec.decode.push_back(ms_between(t0, clock_type::now()));
    started[s] = t0;
    index[s] = i;

    pipe.submit(frames[s], s, conf, nms);
    if (i == dump_frame) {
      const std::vector<float> input = pipe.input_tensor();  // synchronises; dump a warm-up frame
      std::ofstream(dump_path, std::ios::binary)
          .write(reinterpret_cast<const char*>(input.data()),
                 static_cast<std::streamsize>(input.size() * sizeof(float)));
    }

    if (mode == "sync") {
      finish(s);
    } else {
      if (pending >= 0) finish(pending);
      pending = s;
    }
  }
  if (pending >= 0) finish(pending);
  VRT_CUDA_CHECK(cudaStreamSynchronize(stream));
  if (rec.latency.empty())
    throw std::runtime_error("no frames timed; video shorter than --warmup?");

  const std::string name = basename(engine_path).substr(0, basename(engine_path).find('.'));
  const std::string model = name.substr(0, name.find_last_of('_'));
  const std::string precision = name.substr(name.find_last_of('_') + 1);
  const double wall_ms = ms_between(timed_begin, timed_end);
  const double fps = 1000.0 * static_cast<double>(rec.latency.size()) / wall_ms;

  vrt::JsonWriter j;
  j.begin_object()
      .field("pipeline", "gpu-nvdec")
      .field("mode", mode)
      .field("model", model)
      .field("precision", precision)
      .field("engine", basename(engine_path))
      .field("cuda_graph", true);
  j.field("batch", 1).field("streams", 1).field("conf_thresh", conf).field("nms_iou", nms);
  j.field("max_det", max_det);
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
  j.field("warmup_frames", warmup)
      .field("timed_frames", static_cast<long long>(rec.latency.size()));
  j.field("throughput_fps", fps);
  j.field("cpu_ms_per_frame", (cpu_end - cpu_begin) / static_cast<double>(rec.latency.size()));
  j.begin_object("host_ms")
      .field("decode", vrt::summarize(rec.decode))
      .field("latency", vrt::summarize(rec.latency))
      .end_object();
  j.begin_object("gpu_ms")
      .field("preprocess", vrt::summarize(rec.pre))
      .field("infer", vrt::summarize(rec.infer))
      .field("postprocess", vrt::summarize(rec.post))
      .field("d2h", vrt::summarize(rec.d2h))
      .field("total", vrt::summarize(rec.gpu_total))
      .end_object();
  j.field("detections_per_frame", std::accumulate(rec.ndets.begin(), rec.ndets.end(), 0.0) /
                                      static_cast<double>(rec.ndets.size()));
  j.begin_object("environment");
  vrt::write_environment(j);
  j.end_object().end_object();
  if (args.has("json")) std::ofstream(args.str("json")) << j.str();

  const vrt::Summary lat = vrt::summarize(rec.latency);
  std::printf("%s on %s (%s): %zu frames timed\n", name.c_str(), basename(video_path).c_str(),
              mode.c_str(), rec.latency.size());
  std::printf("%-18s%9s%9s%9s  (ms)\n", "stage", "mean", "p50", "p99");
  const std::pair<const char*, const std::vector<double>*> rows[] = {
      {"decode (host)", &rec.decode},  {"preprocess (gpu)", &rec.pre},
      {"infer (gpu)", &rec.infer},     {"nms (gpu)", &rec.post},
      {"d2h (gpu)", &rec.d2h},         {"gpu total", &rec.gpu_total},
      {"latency (host)", &rec.latency}};
  for (const auto& [label, v] : rows) {
    const vrt::Summary s = vrt::summarize(*v);
    std::printf("%-18s%9.3f%9.3f%9.3f\n", label, s.mean, s.p50, s.p99);
  }
  std::printf("throughput %.1f fps (latency p50 %.2f ms)\n", fps, lat.p50);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(vrt::Args(argc, argv));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vrt_pipeline: %s\n", e.what());
    return 1;
  }
}
