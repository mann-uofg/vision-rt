// Cross-camera batching: N camera streams, one batcher that runs their frames together.
//
//   vrt_batched --engines models/yolox_s_fp16.engine,models/yolox_s_fp16_b2.engine,...
//               --videos data/footage/barn004.mp4,... --streams 32 --max-batch 8
//               [--in-flight 2] [--fps 0] [--warmup 100] [--frames 1000] [--decode-threads N]
//               [--json out.json] [--dets-dir DIR]
//
// Each camera has its own NVDEC decoder and CUDA stream. Decoder threads (--decode-threads,
// default one per camera) each drive a share of the cameras: submit the next frame of every
// camera that is due and has a free slot, collect them, record each frame's ready event on its
// camera's stream, and queue it. FFmpeg's nvdec decodes synchronously inside send(), so decode
// parallelism needs threads: one thread for all cameras tops out around 850-1,100 fps. A camera
// may have at most two frames waiting, which gives backpressure. One batcher thread takes
// whatever is queued, up to --max-batch frames, oldest first, and runs them as one batch
// (vrt::BatchInference). This is dynamic batching with no added delay: when the GPU is busy,
// frames pile up and the next batch is bigger; when it's idle, batches are small. Up to
// --in-flight batches (default 2) are queued on the GPU at once, so the next batch is already
// launched while the batcher waits for the previous one's results.
//
// All CUDA graphs are captured before the threads start (BatchInference::prepare): capturing
// one while decoder threads run can be invalidated by their device activity (INTERVIEW H3).
//
// --fps paces each camera like vrt_multistream (phases staggered by k/N of a frame);
// latency is from a frame's release to its detections being back on the host.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <latch>
#include <mutex>
#include <sstream>
#include <thread>

#include "args.h"
#include "vrt/batch_inference.h"
#include "vrt/cuda_check.h"
#include "vrt/nvdec_source.h"
#include "vrt/nvml_sampler.h"
#include "vrt/postprocess.h"
#include "vrt/stats.h"
#include "vrt/sync_pipeline.h"

namespace {

using clock_type = std::chrono::steady_clock;
constexpr int kRing = 3;      // decoded frames a camera owns (held until the slot is reused)
constexpr int kInFlight = 2;  // of which at most this many wait for the batcher

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

struct Camera {
  std::string video;
  std::unique_ptr<vrt::CudaStream> stream;
  std::unique_ptr<vrt::NvdecSource> src;
  vrt::GpuFrame frames[kRing];  // written by the decoder thread only
  cudaEvent_t ready[kRing] = {};
  bool busy[kRing] = {};
  int in_flight = 0;
  std::mutex m;  // guards busy[] and in_flight
  // Written by the batcher only.
  std::vector<double> latency;
  clock_type::time_point timed_begin = clock_type::time_point::max(), timed_end{};
  long long timed = 0;
  std::ofstream dets;
};

struct Request {
  int cam, slot;
  long long index;
  clock_type::time_point release;
};

struct Shared {
  std::mutex m;
  std::condition_variable cv;  // the batcher waits here for queued frames
  std::deque<Request> queue;
  bool decoder_done = false;  // set when the last decoder thread finishes
  std::atomic<int> decoders_running{0};
  std::mutex dm;
  std::condition_variable decoder_cv;  // the decoder waits here for free slots
  std::atomic<bool> abort{false};
};

// Drives the decoders of the cameras in `ids`; each camera belongs to exactly one decoder
// thread, so its FFmpeg objects are only ever used (and its frames only freed) by that thread.
void decoder_loop(std::vector<std::unique_ptr<Camera>>& cams, std::vector<int> ids, double fps,
                  long long total, std::latch& start, Shared& sh, std::string& error) {
  using Recv = vrt::NvdecSource::Recv;
  const int n = static_cast<int>(cams.size());
  std::vector<long long> next(static_cast<std::size_t>(n), 0);  // next frame index per camera
  struct Sent {
    int cam, slot;
    clock_type::time_point release;
  };
  try {
    start.arrive_and_wait();
    const auto t0 = clock_type::now();
    int finished = 0;
    const int mine = static_cast<int>(ids.size());
    while (finished < mine && !sh.abort) {
      const auto now = clock_type::now();
      auto wake = now + std::chrono::milliseconds(1);
      std::vector<Sent> sent;
      // Submit: every camera that is due and has a free slot.
      for (int k : ids) {
        Camera& c = *cams[static_cast<std::size_t>(k)];
        const long long i = next[static_cast<std::size_t>(k)];
        if (i >= total) continue;
        auto release = now;
        if (fps > 0) {
          const double t = (static_cast<double>(i) + static_cast<double>(k) / n) / fps;
          release = t0 + std::chrono::duration_cast<clock_type::duration>(
                             std::chrono::duration<double>(t));
          if (release > now) {
            wake = std::min(wake, release);
            continue;
          }
        }
        int slot = -1;
        {
          std::lock_guard lock(c.m);
          if (c.in_flight < kInFlight) {
            for (int s = 0; s < kRing; ++s)
              if (!c.busy[s]) slot = s;
            // in_flight < kInFlight < kRing guarantees a free slot. An earlier error path broke
            // that invariant and wrote frames[-1] (INTERVIEW H3), so check rather than trust it.
            if (slot < 0) throw std::logic_error("no free frame slot for " + c.video);
            c.busy[slot] = true;
            ++c.in_flight;
          }
        }
        if (slot < 0) continue;  // the batcher still holds both of this camera's frames
        c.src->send();
        sent.push_back({k, slot, release});
      }
      // Collect, in the same order; NVDEC has been decoding all of them meanwhile.
      for (const Sent& s : sent) {
        Camera& c = *cams[static_cast<std::size_t>(s.cam)];
        Recv r;
        while ((r = c.src->receive(c.frames[s.slot])) == Recv::kNeedInput) c.src->send();
        if (r == Recv::kEnd) throw std::runtime_error("video ended: " + c.video);
        VRT_CUDA_CHECK(cudaEventRecord(c.ready[s.slot], *c.stream));
        {
          std::lock_guard lock(sh.m);
          sh.queue.push_back({s.cam, s.slot, next[static_cast<std::size_t>(s.cam)], s.release});
        }
        sh.cv.notify_one();
        if (++next[static_cast<std::size_t>(s.cam)] == total) ++finished;
      }
      if (sent.empty()) {
        // Nothing due, or every due camera is waiting for the batcher: sleep until the next
        // release time or until the batcher frees a slot (at most 1 ms).
        std::unique_lock lock(sh.dm);
        sh.decoder_cv.wait_until(lock, wake);
      }
    }
  } catch (const std::exception& e) {
    error = e.what();
    sh.abort = true;
  }
  if (--sh.decoders_running == 0) {
    std::lock_guard lock(sh.m);
    sh.decoder_done = true;
  }
  sh.cv.notify_one();
}

struct BatchStats {
  std::vector<double> batch_sizes, preprocess_ms, infer_ms, postprocess_ms, total_ms;
};

// Takes whatever is queued (up to max_batch frames, oldest first) and submits it as a batch.
// Up to `batches_in_flight` batches are on the GPU at once: with two, batch n+1 is queued
// before the thread blocks on batch n, so the batch stream doesn't sit empty while this thread
// unpacks results and gathers the next batch. When nothing new is queued, it finishes the
// oldest batch, which hands frame slots back to the decoders.
void batcher_loop(vrt::BatchInference& inf, std::vector<std::unique_ptr<Camera>>& cams,
                  int max_batch, int batches_in_flight, long long warmup, Shared& sh,
                  BatchStats& stats, std::string& error) {
  struct Pending {
    int ticket;
    std::vector<Request> reqs;
  };
  std::deque<Pending> pending;

  const auto finish = [&](const Pending& p) {
    vrt::BatchTimes t;
    const std::vector<vrt::Detections> dets = inf.wait(p.ticket, &t);
    const auto now = clock_type::now();
    stats.batch_sizes.push_back(t.batch);
    stats.preprocess_ms.push_back(t.preprocess_ms);
    stats.infer_ms.push_back(t.infer_ms);
    stats.postprocess_ms.push_back(t.postprocess_ms);
    stats.total_ms.push_back(t.total_ms);

    for (std::size_t j = 0; j < p.reqs.size(); ++j) {
      const Request& r = p.reqs[j];
      Camera& c = *cams[static_cast<std::size_t>(r.cam)];
      if (r.index >= warmup) {
        c.latency.push_back(ms_between(r.release, now));
        c.timed_begin = std::min(c.timed_begin, r.release);
        c.timed_end = now;
        ++c.timed;
      }
      if (c.dets.is_open()) {
        for (const auto& d : dets[j])
          c.dets << r.index << ',' << d.x1 << ',' << d.y1 << ',' << d.x2 << ',' << d.y2 << ','
                 << d.score << ',' << d.cls << '\n';
      }
      {
        // Hand the slot back without touching the frame: FFmpeg frames are only freed on the
        // decoder thread, when it reuses the slot.
        std::lock_guard lock(c.m);
        c.busy[r.slot] = false;
        --c.in_flight;
      }
      sh.decoder_cv.notify_one();
    }
  };

  try {
    while (true) {
      std::vector<Request> reqs;
      {
        std::unique_lock lock(sh.m);
        // Only block for new frames when nothing is in flight; otherwise take what's there.
        if (pending.empty()) sh.cv.wait(lock, [&] { return !sh.queue.empty() || sh.decoder_done; });
        if (sh.queue.empty() && pending.empty()) break;  // decoders done, nothing left
        while (!sh.queue.empty() && static_cast<int>(reqs.size()) < max_batch) {
          reqs.push_back(sh.queue.front());
          sh.queue.pop_front();
        }
      }
      if (!reqs.empty()) {
        std::vector<vrt::BatchItem> items;
        for (const Request& r : reqs) {
          Camera& c = *cams[static_cast<std::size_t>(r.cam)];
          items.push_back({&c.frames[r.slot], c.ready[r.slot]});
        }
        pending.push_back({inf.submit(items), std::move(reqs)});
      }
      if (static_cast<int>(pending.size()) == batches_in_flight ||
          (reqs.empty() && !pending.empty())) {
        finish(pending.front());
        pending.pop_front();
      }
    }
  } catch (const std::exception& e) {
    error = e.what();
    sh.abort = true;  // the decoder stops at its next round
    sh.decoder_cv.notify_one();
  }
}

int run(const vrt::Args& args) {
  const std::vector<std::string> engines = split(args.required("engines"), ',');
  const std::vector<std::string> videos = split(args.required("videos"), ',');
  const int n = static_cast<int>(args.integer("streams", 1));
  const double fps = args.real("fps", 0.f);
  const long long warmup = args.integer("warmup", 100);
  const long long frames = args.integer("frames", 1000);
  const float conf = args.real("conf", vrt::kDeployConf);
  const float nms = args.real("nms", vrt::kDeployNmsIou);
  if (videos.empty() || n < 1) throw std::invalid_argument("need --videos and --streams >= 1");
  if (frames < 1 || warmup < 0) throw std::invalid_argument("need --frames >= 1, --warmup >= 0");

  vrt::CudaStream batch_stream;
  const int in_flight = static_cast<int>(args.integer("in-flight", 2));
  if (in_flight < 1) throw std::invalid_argument("--in-flight must be at least 1");
  vrt::BatchInference inf(engines, batch_stream, conf, nms, 100, in_flight);
  const int max_batch = static_cast<int>(args.integer("max-batch", inf.max_batch()));
  if (max_batch < 1 || max_batch > inf.max_batch())
    throw std::invalid_argument("--max-batch must be between 1 and the largest engine's batch");

  std::vector<std::unique_ptr<Camera>> cams;
  for (int i = 0; i < n; ++i) {
    auto c = std::make_unique<Camera>();
    c->video = videos[static_cast<std::size_t>(i) % videos.size()];
    c->stream = std::make_unique<vrt::CudaStream>();
    c->src = std::make_unique<vrt::NvdecSource>(c->video, *c->stream, kRing);
    for (auto& e : c->ready) VRT_CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    if (args.has("dets-dir"))
      c->dets.open(args.str("dets-dir") + "/cam" + std::to_string(i) + ".csv");
    if (c->dets.is_open()) c->dets << "frame,x1,y1,x2,y2,score,cls\n";
    cams.push_back(std::move(c));
  }

  // Capture every graph now, before the decoder and batcher threads start.
  for (const auto& c : cams)
    if (c->src->info().width != cams[0]->src->info().width ||
        c->src->info().height != cams[0]->src->info().height)
      throw std::invalid_argument("all videos must have the same frame size");
  inf.prepare(cams[0]->src->info().width, cams[0]->src->info().height);

  Shared sh;
  BatchStats stats;
  const int decode_threads = std::clamp(static_cast<int>(args.integer("decode-threads", n)), 1, n);
  std::vector<std::string> decode_errors(static_cast<std::size_t>(decode_threads));
  std::string batch_error;
  vrt::UtilSampler sampler;
  std::latch start(decode_threads + 1);
  sh.decoders_running = decode_threads;
  std::vector<std::thread> decoders;
  for (int d = 0; d < decode_threads; ++d) {
    std::vector<int> ids;
    for (int k = d; k < n; k += decode_threads) ids.push_back(k);
    decoders.emplace_back(decoder_loop, std::ref(cams), ids, fps, warmup + frames, std::ref(start),
                          std::ref(sh), std::ref(decode_errors[static_cast<std::size_t>(d)]));
  }
  std::thread batcher(batcher_loop, std::ref(inf), std::ref(cams), max_batch, in_flight, warmup,
                      std::ref(sh), std::ref(stats), std::ref(batch_error));
  const double cpu0 = vrt::process_cpu_ms();
  start.arrive_and_wait();
  sampler.start();
  for (auto& d : decoders) d.join();
  batcher.join();
  sampler.stop();
  const double cpu1 = vrt::process_cpu_ms();
  for (const auto& e : decode_errors)
    if (!e.empty()) throw std::runtime_error("decoder: " + e);
  if (!batch_error.empty()) throw std::runtime_error("batcher: " + batch_error);

  std::vector<double> all, per_stream_p99;
  clock_type::time_point begin = clock_type::time_point::max(), end{};
  long long total = 0;
  for (auto& c : cams) {
    for (auto& e : c->ready) cudaEventDestroy(e);
    all.insert(all.end(), c->latency.begin(), c->latency.end());
    per_stream_p99.push_back(vrt::percentile(c->latency, 99));
    begin = std::min(begin, c->timed_begin);
    end = std::max(end, c->timed_end);
    total += c->timed;
  }
  const double agg_fps = 1000.0 * static_cast<double>(total) / ms_between(begin, end);
  const vrt::Summary lat = vrt::summarize(all);
  const vrt::Summary bs = vrt::summarize(stats.batch_sizes);
  const double worst_p99 = *std::max_element(per_stream_p99.begin(), per_stream_p99.end());

  vrt::JsonWriter j;
  j.begin_object().field("pipeline", "gpu-nvdec-batched").field("engines", args.str("engines"));
  j.field("streams", n).field("max_batch", max_batch).field("batches_in_flight", in_flight);
  j.field("nms", "batched").field("fps_per_stream_limit", fps);
  j.field("phase_staggered", fps > 0).field("decode_threads", decode_threads);
  j.field("warmup_frames", warmup).field("timed_frames_per_stream", frames);
  j.field("aggregate_fps", agg_fps).field("per_stream_fps", agg_fps / n);
  j.field("latency_ms", lat).field("worst_stream_p99_ms", worst_p99);
  j.field("batch_size", bs).field("batch_preprocess_ms", vrt::summarize(stats.preprocess_ms));
  j.field("batch_infer_ms", vrt::summarize(stats.infer_ms));
  j.field("batch_postprocess_ms", vrt::summarize(stats.postprocess_ms));
  j.field("batch_gpu_total_ms", vrt::summarize(stats.total_ms));
  j.field("gpu_util_pct", sampler.gpu_mean()).field("decoder_util_pct", sampler.decoder_mean());
  j.field("cpu_ms_per_frame", (cpu1 - cpu0) / static_cast<double>(n * (warmup + frames)));
  j.begin_object("environment");
  vrt::write_environment(j);
  j.end_object().end_object();
  if (args.has("json")) std::ofstream(args.str("json")) << j.str();

  std::printf(
      "%2d streams, max batch %2d, %d in flight (fps limit %g): %7.1f fps total, mean batch %.1f, "
      "latency p50 %.2f p99 %.2f worst-stream p99 %.2f ms, GPU %.0f%%, NVDEC %.0f%%\n",
      n, max_batch, in_flight, fps, agg_fps, bs.mean, lat.p50, lat.p99, worst_p99,
      sampler.gpu_mean(), sampler.decoder_mean());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(vrt::Args(argc, argv));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vrt_batched: %s\n", e.what());
    return 1;
  }
}
