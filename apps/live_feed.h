#pragma once

// vrt_multistream --show: each camera's detections plus a small BGR copy of its frame, streamed
// on stdout to a viewer process (tools/live_view.py) that draws the 2x2 camera wall.
//
// Per frame, after the pipeline's own work, the camera thread queues one more kernel on its
// stream (nv12_to_bgr_thumbnail, 1080p -> 640x360) and a 0.7 MB copy into pinned memory. The
// detections are the ones the pipeline already copied back. Full frames never leave the GPU.
//
// Camera threads never wait for the viewer. A finished frame replaces that camera's previous one
// in a one-deep mailbox and a writer thread sends whatever is newest, so a slow viewer skips
// frames instead of stalling the pipeline. Without --show none of this exists and the camera
// loop is the plain benchmark's.
//
// Message, little-endian, all fields 4 bytes:
//   header  "VRT1", int32 cam, width, height, ndet, frame,
//           float32 latency_ms, fps, p50_ms, p99_ms
//   ndet x  float32 x1, y1, x2, y2, score, cls        (thumbnail pixels)
//   pixels  height x width x 3 bytes, BGR
// latency_ms is this frame's (release to detections on the host, as in the benchmark); fps,
// p50 and p99 are this camera's over its last kWindow timed frames (-1 during warm-up).

#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vrt/cuda_check.h"
#include "vrt/detection.h"
#include "vrt/gpu_pipeline.h"
#include "vrt/nvdec_source.h"
#include "vrt/preprocess_gpu.h"
#include "vrt/stats.h"

namespace vrt {

class LiveFeed {
 public:
  using clock_type = std::chrono::steady_clock;
  static constexpr int kWidth = 640;             // thumbnail width; height keeps the aspect ratio
  static constexpr std::size_t kWindow = 300;    // frames in the rolling percentiles (10 s)
  static constexpr std::size_t kFpsFrames = 30;  // frames in the rolling fps (1 s)

  // One {width, height} per camera. Allocates everything now: construct it before the camera
  // threads start, like the CUDA graphs.
  explicit LiveFeed(const std::vector<std::pair<int, int>>& frame_sizes)
      : cams_(frame_sizes.size()) {
    std::signal(SIGPIPE, SIG_IGN);  // a closed viewer shows up as a failed write instead
    for (std::size_t k = 0; k < cams_.size(); ++k) {
      Cam& c = cams_[k];
      c.frame_w = frame_sizes[k].first;
      c.frame_h = frame_sizes[k].second;
      c.w = kWidth;
      c.h = (kWidth * c.frame_h + c.frame_w / 2) / c.frame_w;
      c.bytes = static_cast<std::size_t>(c.w) * c.h * 3;
      VRT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c.d_bgr), c.bytes));
      for (int s = 0; s < GpuPipeline::kSlots; ++s) {
        VRT_CUDA_CHECK(
            cudaHostAlloc(reinterpret_cast<void**>(&c.h_bgr[s]), c.bytes, cudaHostAllocDefault));
        VRT_CUDA_CHECK(cudaEventCreateWithFlags(&c.copied[s], cudaEventDisableTiming));
      }
    }
  }

  ~LiveFeed() {
    cudaDeviceSynchronize();  // a camera that failed mid-frame may still have a copy queued
    for (Cam& c : cams_) {
      cudaFree(c.d_bgr);
      for (int s = 0; s < GpuPipeline::kSlots; ++s) {
        cudaFreeHost(c.h_bgr[s]);
        cudaEventDestroy(c.copied[s]);
      }
    }
  }

  LiveFeed(const LiveFeed&) = delete;
  LiveFeed& operator=(const LiveFeed&) = delete;

  // Camera thread, right after pipe.submit(frame, slot): queue the thumbnail and its copy
  // behind the pipeline's work on the same stream.
  void queue(int cam, int slot, const GpuFrame& f, cudaStream_t stream) {
    Cam& c = cams_[static_cast<std::size_t>(cam)];
    if (f.width != c.frame_w || f.height != c.frame_h)
      throw std::runtime_error("frame size changed mid-stream");
    Nv12View v;
    v.luma = f.luma;
    v.chroma = f.chroma;
    v.luma_pitch = f.luma_pitch;
    v.chroma_pitch = f.chroma_pitch;
    v.width = f.width;
    v.height = f.height;
    nv12_to_bgr_thumbnail(v, yuv_coeffs(f.matrix == YuvMatrix::kBt709, f.full_range), c.d_bgr, c.w,
                          c.h, stream);
    VRT_CUDA_CHECK(
        cudaMemcpyAsync(c.h_bgr[slot], c.d_bgr, c.bytes, cudaMemcpyDeviceToHost, stream));
    VRT_CUDA_CHECK(cudaEventRecord(c.copied[slot], stream));
  }

  // Camera thread, after pipe.collect(slot) and before the frame is released: wait for the
  // thumbnail, update the rolling stats and hand the frame to the writer. Warm-up frames
  // (`timed` false) are shown but, as in the benchmark, kept out of the percentiles.
  void publish(int cam, int slot, long long frame, double latency_ms, const Detections& dets,
               clock_type::time_point done, bool timed) {
    Cam& c = cams_[static_cast<std::size_t>(cam)];
    VRT_CUDA_CHECK(cudaEventSynchronize(c.copied[slot]));

    if (timed && c.latency.size() < kWindow) {
      c.latency.push_back(latency_ms);
    } else if (timed) {
      c.latency[c.next] = latency_ms;
      c.next = (c.next + 1) % kWindow;
    }
    const bool warm = !c.latency.empty();
    c.done.push_back(done);
    if (c.done.size() > kFpsFrames + 1) c.done.pop_front();
    const double span = std::chrono::duration<double>(c.done.back() - c.done.front()).count();
    const double fps = span > 0 ? static_cast<double>(c.done.size() - 1) / span : 0.0;

    Message& m = c.staging;
    m.head = Header{{'V', 'R', 'T', '1'},
                    cam,
                    c.w,
                    c.h,
                    static_cast<std::int32_t>(dets.size()),
                    static_cast<std::int32_t>(frame),
                    static_cast<float>(latency_ms),
                    static_cast<float>(fps),
                    warm ? static_cast<float>(percentile(c.latency, 50)) : -1.f,
                    warm ? static_cast<float>(percentile(c.latency, 99)) : -1.f};
    const float sx = static_cast<float>(c.w) / static_cast<float>(c.frame_w);
    const float sy = static_cast<float>(c.h) / static_cast<float>(c.frame_h);
    m.dets.clear();
    for (const Detection& d : dets) {
      const float row[6] = {d.x1 * sx, d.y1 * sy, d.x2 * sx,
                            d.y2 * sy, d.score,   static_cast<float>(d.cls)};
      m.dets.insert(m.dets.end(), row, row + 6);
    }
    m.pixels.assign(c.h_bgr[slot], c.h_bgr[slot] + c.bytes);
    {
      std::lock_guard lock(mu_);
      std::swap(c.staging, c.box);
      c.fresh = true;
    }
    cv_.notify_one();
  }

  // The writer loop, on its own thread. Returns after stop(), or once the viewer has gone.
  void run() {
    std::vector<Message> out(cams_.size());
    std::vector<std::size_t> ready;
    std::unique_lock lock(mu_);
    while (true) {
      cv_.wait(lock, [&] { return stop_ || any_fresh(); });
      if (stop_) return;
      ready.clear();
      for (std::size_t k = 0; k < cams_.size(); ++k) {
        if (!cams_[k].fresh) continue;
        std::swap(out[k], cams_[k].box);
        cams_[k].fresh = false;
        ready.push_back(k);
      }
      lock.unlock();
      bool ok = true;
      for (std::size_t k : ready) ok = ok && send(out[k]);
      lock.lock();
      if (!ok) {
        closed_ = true;
        return;
      }
    }
  }

  void stop() {
    {
      std::lock_guard lock(mu_);
      stop_ = true;
    }
    cv_.notify_all();
  }

  // True once the viewer has closed its end; the camera threads then finish up.
  bool closed() const { return closed_.load(); }

 private:
  struct Header {
    char magic[4];
    std::int32_t cam, width, height, ndet, frame;
    float latency_ms, fps, p50_ms, p99_ms;
  };
  static_assert(sizeof(Header) == 40, "the viewer reads a 40-byte header");

  struct Message {
    Header head{};
    std::vector<float> dets;
    std::vector<std::uint8_t> pixels;
  };

  struct Cam {
    int frame_w = 0, frame_h = 0, w = 0, h = 0;
    std::size_t bytes = 0;
    std::uint8_t* d_bgr = nullptr;
    std::uint8_t* h_bgr[GpuPipeline::kSlots] = {};
    cudaEvent_t copied[GpuPipeline::kSlots] = {};
    // Camera thread only.
    std::vector<double> latency;  // ring of the last kWindow latencies
    std::size_t next = 0;
    std::deque<clock_type::time_point> done;
    Message staging;
    // Guarded by mu_.
    Message box;
    bool fresh = false;
  };

  bool any_fresh() const {
    for (const Cam& c : cams_)
      if (c.fresh) return true;
    return false;
  }

  static bool write_all(const void* data, std::size_t n) {
    const auto* p = static_cast<const char*>(data);
    while (n > 0) {
      const ssize_t w = ::write(STDOUT_FILENO, p, n);
      if (w < 0 && errno == EINTR) continue;
      if (w <= 0) return false;
      p += w;
      n -= static_cast<std::size_t>(w);
    }
    return true;
  }

  static bool send(const Message& m) {
    return write_all(&m.head, sizeof m.head) &&
           write_all(m.dets.data(), m.dets.size() * sizeof(float)) &&
           write_all(m.pixels.data(), m.pixels.size());
  }

  std::vector<Cam> cams_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::atomic<bool> closed_{false};
};

}  // namespace vrt
