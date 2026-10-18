// N camera streams at once on one GPU: the scaling experiment.
//
//   vrt_multistream --engine models/yolox_s_fp16.engine --streams 8
//                   --videos data/footage/barn004.mp4,data/footage/barn007.mp4,...
//                   [--mode pipelined|sync] [--fps 0] [--warmup 100] [--frames 1000]
//                   [--json out.json]
//
// Each stream is one worker thread with its own NVDEC decoder, CUDA stream, TensorRT execution
// context and buffers (the M2 GpuPipeline). The engine itself is loaded once and shared.
// Videos are assigned to streams round-robin.
//
// --fps 0 runs every stream as fast as it can (capacity). --fps 30 paces each stream like a
// real camera: frame i of stream k is not read before start + (i + k/N)/30 s, and latency is
// measured from that release time, so queueing shows up as latency. The k/N phase offset
// spreads the streams across the frame interval like unsynchronised cameras; without it all N
// streams release a frame at the same instant and the GPU sees N-frame bursts.
//
// While the timed frames run, NVML samples GPU and decoder (NVDEC) utilisation.

#include <nvml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <latch>
#include <sstream>
#include <thread>

#include "args.h"
#include "vrt/cuda_check.h"
#include "vrt/gpu_pipeline.h"
#include "vrt/nvdec_source.h"
#include "vrt/postprocess.h"
#include "vrt/stats.h"
#include "vrt/sync_pipeline.h"

namespace {

using clock_type = std::chrono::steady_clock;

double ms_between(clock_type::time_point a, clock_type::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, sep);)
    if (!item.empty()) out.push_back(item);
  return out;
}

struct Config {
  std::string mode;
  double fps;
  long long warmup, frames;
  float conf, nms;
};

struct Stream {
  std::string video;
  std::unique_ptr<vrt::CudaStream> cuda;
  std::unique_ptr<vrt::GpuPipeline> pipe;
  std::unique_ptr<vrt::NvdecSource> src;
  std::vector<double> latency;
  clock_type::time_point timed_begin, timed_end;
  long long timed = 0;
  std::string error;
};

// One camera's loop; same structure as vrt_pipeline, plus optional pacing.
// `warm` counts streams that are past warm-up, or have failed: main waits for all of them
// before it starts sampling, so a stream that dies early must still be counted.
void run_stream(Stream& s, const Config& c, double phase, std::latch& start,
                std::atomic<int>& warm) {
  bool counted = false;
  try {
    vrt::GpuFrame frames[vrt::GpuPipeline::kSlots];
    clock_type::time_point released[vrt::GpuPipeline::kSlots];
    long long index[vrt::GpuPipeline::kSlots] = {};
    start.arrive_and_wait();
    const auto t_start = clock_type::now();

    auto finish = [&](int slot) {
      s.pipe->collect(slot, nullptr);
      const auto now = clock_type::now();
      frames[slot].release();
      if (index[slot] < c.warmup) return;
      s.latency.push_back(ms_between(released[slot], now));
      s.timed_end = now;
      ++s.timed;
    };

    int pending = -1;
    for (long long i = 0; i < c.warmup + c.frames; ++i) {
      const int slot = static_cast<int>(i % vrt::GpuPipeline::kSlots);
      auto release = clock_type::now();
      if (c.fps > 0) {
        release =
            t_start + std::chrono::duration_cast<clock_type::duration>(
                          std::chrono::duration<double>((static_cast<double>(i) + phase) / c.fps));
        std::this_thread::sleep_until(release);
      }
      if (i == c.warmup) {
        s.timed_begin = c.fps > 0 ? release : clock_type::now();
        warm.fetch_add(1);
        counted = true;
      }
      if (!s.src->read(frames[slot])) throw std::runtime_error("video ended: " + s.video);
      released[slot] = release;
      index[slot] = i;
      s.pipe->submit(frames[slot], slot, c.conf, c.nms);
      if (c.mode == "sync") {
        finish(slot);
      } else {
        if (pending >= 0) finish(pending);
        pending = slot;
      }
    }
    if (pending >= 0) finish(pending);
  } catch (const std::exception& e) {
    s.error = e.what();
    if (!counted) warm.fetch_add(1);
  }
}

// Mean GPU and decoder utilisation (%) while `running` is set.
class UtilSampler {
 public:
  UtilSampler() {
    if (nvmlInit_v2() == NVML_SUCCESS && nvmlDeviceGetHandleByIndex_v2(0, &dev_) == NVML_SUCCESS)
      ok_ = true;
  }
  ~UtilSampler() {
    stop();
    if (ok_) nvmlShutdown();
  }
  void start() {
    if (!ok_) return;
    running_ = true;
    thread_ = std::thread([this] {
      while (running_) {
        nvmlUtilization_t u{};
        unsigned dec = 0, period = 0;
        if (nvmlDeviceGetUtilizationRates(dev_, &u) == NVML_SUCCESS) gpu_.push_back(u.gpu);
        if (nvmlDeviceGetDecoderUtilization(dev_, &dec, &period) == NVML_SUCCESS)
          dec_.push_back(dec);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });
  }
  void stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
  }
  double gpu_mean() const { return mean(gpu_); }
  double decoder_mean() const { return mean(dec_); }

 private:
  static double mean(const std::vector<double>& v) {
    if (v.empty()) return -1;
    double s = 0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
  }
  nvmlDevice_t dev_{};
  bool ok_ = false;
  std::atomic<bool> running_{false};
  std::thread thread_;
  std::vector<double> gpu_, dec_;  // written by the sampling thread, read after it joins
};

int run(const vrt::Args& args) {
  const std::string engine_path = args.required("engine");
  const std::vector<std::string> videos = split(args.required("videos"), ',');
  const int n = static_cast<int>(args.integer("streams", 1));
  Config c{args.str("mode", "pipelined"),       args.real("fps", 0.f),
           args.integer("warmup", 100),         args.integer("frames", 1000),
           args.real("conf", vrt::kDeployConf), args.real("nms", vrt::kDeployNmsIou)};
  if (c.mode != "pipelined" && c.mode != "sync") throw std::invalid_argument("--mode");
  if (videos.empty() || n < 1) throw std::invalid_argument("need --videos and --streams >= 1");

  // Setup is serial: loading the model, creating contexts and capturing graphs.
  auto model = vrt::TrtModel::load(engine_path);
  std::vector<Stream> streams(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    Stream& s = streams[static_cast<std::size_t>(i)];
    s.video = videos[static_cast<std::size_t>(i) % videos.size()];
    s.cuda = std::make_unique<vrt::CudaStream>();
    s.pipe = std::make_unique<vrt::GpuPipeline>(model, *s.cuda);
    s.src = std::make_unique<vrt::NvdecSource>(s.video, *s.cuda, vrt::GpuPipeline::kSlots);
  }

  UtilSampler sampler;
  std::latch start(n + 1);
  std::atomic<int> warm{0};
  std::vector<std::thread> threads;
  for (int k = 0; k < n; ++k) {
    const double phase = static_cast<double>(k) / n;  // fraction of a frame interval
    threads.emplace_back(run_stream, std::ref(streams[static_cast<std::size_t>(k)]), std::cref(c),
                         phase, std::ref(start), std::ref(warm));
  }
  const double cpu0 = vrt::process_cpu_ms();
  start.arrive_and_wait();
  while (warm.load() < n) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  sampler.start();  // all streams are past warm-up
  for (auto& t : threads) t.join();
  sampler.stop();
  const double cpu1 = vrt::process_cpu_ms();

  std::vector<double> all, per_stream_fps, per_stream_p99;
  clock_type::time_point begin = clock_type::time_point::max(), end{};
  long long total = 0;
  for (const auto& s : streams) {
    if (!s.error.empty()) throw std::runtime_error("stream on " + s.video + ": " + s.error);
    all.insert(all.end(), s.latency.begin(), s.latency.end());
    per_stream_fps.push_back(1000.0 * static_cast<double>(s.timed) /
                             ms_between(s.timed_begin, s.timed_end));
    per_stream_p99.push_back(vrt::percentile(s.latency, 99));
    begin = std::min(begin, s.timed_begin);
    end = std::max(end, s.timed_end);
    total += s.timed;
  }
  const double wall = ms_between(begin, end);
  const double agg_fps = 1000.0 * static_cast<double>(total) / wall;
  const vrt::Summary lat = vrt::summarize(all);
  const vrt::Summary sfps = vrt::summarize(per_stream_fps);
  const double worst_p99 = *std::max_element(per_stream_p99.begin(), per_stream_p99.end());

  const auto slash = engine_path.find_last_of('/');
  const std::string ename = engine_path.substr(slash == std::string::npos ? 0 : slash + 1);
  vrt::JsonWriter j;
  j.begin_object().field("pipeline", "gpu-nvdec-multistream").field("engine", ename);
  j.field("mode", c.mode).field("streams", n).field("fps_per_stream_limit", c.fps);
  j.field("phase_staggered", c.fps > 0);
  j.field("warmup_frames", c.warmup).field("timed_frames_per_stream", c.frames);
  j.field("aggregate_fps", agg_fps).field("per_stream_fps", sfps);
  j.field("latency_ms", lat).field("worst_stream_p99_ms", worst_p99);
  j.field("gpu_util_pct", sampler.gpu_mean()).field("decoder_util_pct", sampler.decoder_mean());
  // Process CPU time over every frame of the run, warm-up included.
  j.field("cpu_ms_per_frame", (cpu1 - cpu0) / static_cast<double>(n * (c.warmup + c.frames)));
  j.field("videos", static_cast<int>(videos.size()));
  j.begin_object("environment");
  vrt::write_environment(j);
  j.end_object().end_object();
  if (args.has("json")) std::ofstream(args.str("json")) << j.str();

  std::printf(
      "%2d streams (%s, fps limit %g): %7.1f fps total, %6.1f per stream (min %.1f), latency p50 "
      "%.2f p99 %.2f worst-stream p99 %.2f ms, GPU %.0f%%, NVDEC %.0f%%\n",
      n, c.mode.c_str(), c.fps, agg_fps, sfps.mean,
      *std::min_element(per_stream_fps.begin(), per_stream_fps.end()), lat.p50, lat.p99, worst_p99,
      sampler.gpu_mean(), sampler.decoder_mean());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(vrt::Args(argc, argv));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vrt_multistream: %s\n", e.what());
    return 1;
  }
}
