#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>

#include "vrt/video_source.h"

struct AVFrame;

namespace vrt {

enum class YuvMatrix { kBt601, kBt709 };

// A decoded NV12 frame that lives in GPU memory. Holds a reference to the decoder's surface,
// so the decoder won't reuse it until this object is released. Keep it alive until every
// GPU operation that reads it has finished.
class GpuFrame {
 public:
  GpuFrame() = default;
  ~GpuFrame();
  GpuFrame(GpuFrame&& other) noexcept;
  GpuFrame& operator=(GpuFrame&& other) noexcept;
  GpuFrame(const GpuFrame&) = delete;
  GpuFrame& operator=(const GpuFrame&) = delete;

  void release();
  bool empty() const { return frame_ == nullptr; }

  const std::uint8_t* luma = nullptr;    // width x height, `luma_pitch` bytes per row
  const std::uint8_t* chroma = nullptr;  // interleaved UV, width/2 x height/2 pairs
  int luma_pitch = 0, chroma_pitch = 0;
  int width = 0, height = 0;
  YuvMatrix matrix = YuvMatrix::kBt709;
  bool full_range = false;

 private:
  friend class NvdecSource;
  AVFrame* frame_ = nullptr;
};

// Hardware decode on the GPU (NVDEC) through FFmpeg's CUDA hwaccel. Frames never leave the
// GPU: read() hands back device pointers to the NV12 surface.
//
// FFmpeg shares the CUDA runtime's primary context, and its copy out of the decoder is
// issued on `stream`. Work queued on the same stream afterwards is ordered behind it.
//
// Threads: separate NvdecSources can be driven from separate threads (tested under ASAN with 16
// concurrent decoders), but each one, and the GpuFrames it returns, should stay on the thread
// that owns it. send()/receive() expose the two halves of read(); note that FFmpeg's nvdec path
// decodes synchronously inside send(), so one thread can't overlap many decoders this way.
//
// VRT_DECODER=cuvid switches to FFmpeg's <codec>_cuvid decoder (NVIDIA's parser and its own
// surface queue): ~30% faster per frame here, but not what the reported results used.
class NvdecSource {
 public:
  // in_flight: frames the caller may hold at once; the decoder's surface pool grows by this.
  NvdecSource(const std::string& path, cudaStream_t stream, int in_flight = 2);
  ~NvdecSource();
  NvdecSource(const NvdecSource&) = delete;
  NvdecSource& operator=(const NvdecSource&) = delete;

  // Next frame, or false at end of stream. Equivalent to receive(), with send() as needed.
  bool read(GpuFrame& out);

  // Feed the next packet to the decoder; NVDEC starts on it asynchronously. False once the
  // input is exhausted and the decoder has been told to flush.
  bool send();

  enum class Recv { kFrame, kNeedInput, kEnd };
  // A decoded frame if one is ready (waits for NVDEC to finish it), kNeedInput if the decoder
  // needs another send() first, kEnd once it is fully drained.
  Recv receive(GpuFrame& out);
  const VideoInfo& info() const { return info_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  VideoInfo info_;
};

}  // namespace vrt
