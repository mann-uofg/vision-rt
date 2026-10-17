// Decode a video with the CPU (FFmpeg software decode + swscale to BGR) or with NVDEC
// (frames stay on the GPU as NV12) and report per-frame decode time.
//
//   vrt_decode_bench --video data/footage/barn022.mp4 --decoder nvdec|cpu
//                    [--warmup 30] [--json out.json] [--dump-nv12 FRAME:path.bin]
//
// --dump-nv12 copies one decoded NV12 frame to the host (Y plane, then interleaved UV, both
// tightly packed) so its colour conversion can be checked offline.

#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>

#include "args.h"
#include "vrt/cuda_check.h"
#include "vrt/nvdec_source.h"
#include "vrt/stats.h"
#include "vrt/sync_pipeline.h"
#include "vrt/video_source.h"

namespace {

using clock_type = std::chrono::steady_clock;

double ms_since(clock_type::time_point t) {
  return std::chrono::duration<double, std::milli>(clock_type::now() - t).count();
}

void dump_nv12(const vrt::GpuFrame& f, const std::string& path) {
  std::vector<std::uint8_t> host(static_cast<std::size_t>(f.width) * f.height * 3 / 2);
  VRT_CUDA_CHECK(cudaMemcpy2D(host.data(), static_cast<std::size_t>(f.width), f.luma,
                              static_cast<std::size_t>(f.luma_pitch),
                              static_cast<std::size_t>(f.width), static_cast<std::size_t>(f.height),
                              cudaMemcpyDeviceToHost));
  VRT_CUDA_CHECK(cudaMemcpy2D(
      host.data() + static_cast<std::size_t>(f.width) * f.height, static_cast<std::size_t>(f.width),
      f.chroma, static_cast<std::size_t>(f.chroma_pitch), static_cast<std::size_t>(f.width),
      static_cast<std::size_t>(f.height / 2), cudaMemcpyDeviceToHost));
  std::ofstream(path, std::ios::binary)
      .write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(host.size()));
}

int run(const vrt::Args& args) {
  const std::string video = args.required("video");
  const std::string decoder = args.str("decoder", "nvdec");
  const long long warmup = args.integer("warmup", 30);
  long long dump_frame = -1;
  std::string dump_path;
  if (args.has("dump-nv12")) {
    const std::string v = args.str("dump-nv12");
    dump_frame = std::stoll(v.substr(0, v.find(':')));
    dump_path = v.substr(v.find(':') + 1);
  }

  std::vector<double> times;
  vrt::VideoInfo info;
  long long n = 0;
  if (decoder == "nvdec") {
    cudaStream_t stream;
    VRT_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    vrt::NvdecSource src(video, stream);
    info = src.info();
    vrt::GpuFrame frame;
    for (;; ++n) {
      const auto t = clock_type::now();
      if (!src.read(frame)) break;
      // The copy out of the decoder is queued on `stream`; wait for it so the time is the
      // full cost of getting a usable frame.
      VRT_CUDA_CHECK(cudaStreamSynchronize(stream));
      if (n >= warmup) times.push_back(ms_since(t));
      if (n == dump_frame) dump_nv12(frame, dump_path);
    }
    frame.release();
    VRT_CUDA_CHECK(cudaStreamDestroy(stream));
  } else if (decoder == "cpu") {
    vrt::VideoSource src(video);
    info = src.info();
    vrt::BgrImage img;
    for (;; ++n) {
      const auto t = clock_type::now();
      if (!src.read(img)) break;
      if (n >= warmup) times.push_back(ms_since(t));
    }
  } else {
    throw std::invalid_argument("--decoder must be nvdec or cpu");
  }
  if (times.empty()) throw std::runtime_error("no frames timed");

  const vrt::Summary s = vrt::summarize(times);
  std::printf("%s: %lld frames, %dx%d %s, decode mean %.3f p50 %.3f p99 %.3f ms (%.0f fps)\n",
              decoder.c_str(), n, info.width, info.height, info.codec.c_str(), s.mean, s.p50, s.p99,
              1000.0 / s.mean);

  if (args.has("json")) {
    vrt::JsonWriter j;
    j.begin_object().field("decoder", decoder).field("video", video).field("codec", info.codec);
    j.field("width", info.width).field("height", info.height).field("frames", n);
    j.field("warmup_frames", warmup).field("decode_ms", s);
    j.field("output", decoder == "nvdec" ? "NV12 in GPU memory" : "BGR24 in host memory");
    j.begin_object("environment");
    vrt::write_environment(j);
    j.end_object().end_object();
    std::ofstream(args.str("json")) << j.str();
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(vrt::Args(argc, argv));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vrt_decode_bench: %s\n", e.what());
    return 1;
  }
}
