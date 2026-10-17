#include "vrt/nvdec_source.h"

#include <cuda.h>

#include <stdexcept>
#include <utility>

#include "vrt/cuda_check.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace vrt {

namespace {

std::string av_error(int err) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(err, buf, sizeof buf);
  return buf;
}

void check(int err, const std::string& what) {
  if (err < 0) throw std::runtime_error(what + ": " + av_error(err));
}

AVPixelFormat pick_cuda_format(AVCodecContext*, const AVPixelFormat* formats) {
  for (const AVPixelFormat* f = formats; *f != AV_PIX_FMT_NONE; ++f)
    if (*f == AV_PIX_FMT_CUDA) return *f;
  return AV_PIX_FMT_NONE;  // decoder can't use NVDEC for this stream: fail loudly
}

YuvMatrix matrix_for(const AVFrame* f) {
  switch (f->colorspace) {
    case AVCOL_SPC_BT709:
      return YuvMatrix::kBt709;
    case AVCOL_SPC_BT470BG:
    case AVCOL_SPC_SMPTE170M:
      return YuvMatrix::kBt601;
    default:
      return f->height >= 720 ? YuvMatrix::kBt709 : YuvMatrix::kBt601;  // same rule as the CPU path
  }
}

}  // namespace

GpuFrame::~GpuFrame() { release(); }

GpuFrame::GpuFrame(GpuFrame&& o) noexcept { *this = std::move(o); }

GpuFrame& GpuFrame::operator=(GpuFrame&& o) noexcept {
  if (this != &o) {
    release();
    luma = o.luma;
    chroma = o.chroma;
    luma_pitch = o.luma_pitch;
    chroma_pitch = o.chroma_pitch;
    width = o.width;
    height = o.height;
    matrix = o.matrix;
    full_range = o.full_range;
    frame_ = std::exchange(o.frame_, nullptr);
  }
  return *this;
}

void GpuFrame::release() {
  if (frame_) av_frame_free(&frame_);
  luma = chroma = nullptr;
}

struct NvdecSource::Impl {
  AVFormatContext* fmt = nullptr;
  AVCodecContext* dec = nullptr;
  AVBufferRef* device = nullptr;
  AVPacket* pkt = nullptr;
  int stream = -1;
  bool draining = false;

  ~Impl() {
    av_packet_free(&pkt);
    avcodec_free_context(&dec);
    av_buffer_unref(&device);
    avformat_close_input(&fmt);
  }
};

NvdecSource::NvdecSource(const std::string& path, cudaStream_t stream, int in_flight)
    : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;

  // CUDA device context that uses the runtime's primary context and our stream.
  VRT_CUDA_CHECK(cudaFree(nullptr));  // make sure the runtime has made the primary context current
  CUcontext ctx = nullptr;
  if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx)
    throw std::runtime_error("no current CUDA context");
  s.device = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
  if (!s.device) throw std::runtime_error("av_hwdevice_ctx_alloc failed");
  auto* cuda = static_cast<AVCUDADeviceContext*>(
      reinterpret_cast<AVHWDeviceContext*>(s.device->data)->hwctx);
  cuda->cuda_ctx = ctx;
  cuda->stream = stream;
  check(av_hwdevice_ctx_init(s.device), "init CUDA device context");

  check(avformat_open_input(&s.fmt, path.c_str(), nullptr, nullptr), "open " + path);
  check(avformat_find_stream_info(s.fmt, nullptr), "find stream info");
  const AVCodec* codec = nullptr;
  s.stream = av_find_best_stream(s.fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
  check(s.stream, "no video stream in " + path);
  const AVStream* st = s.fmt->streams[s.stream];

  s.dec = avcodec_alloc_context3(codec);
  if (!s.dec) throw std::runtime_error("avcodec_alloc_context3 failed");
  check(avcodec_parameters_to_context(s.dec, st->codecpar), "codec parameters");
  s.dec->hw_device_ctx = av_buffer_ref(s.device);
  s.dec->get_format = pick_cuda_format;
  s.dec->extra_hw_frames = in_flight + 1;
  check(avcodec_open2(s.dec, codec, nullptr), "open decoder with NVDEC");
  s.pkt = av_packet_alloc();
  if (!s.pkt) throw std::runtime_error("av_packet_alloc failed");

  info_.codec = std::string(codec->name) + " (nvdec)";
  info_.width = st->codecpar->width;
  info_.height = st->codecpar->height;
  info_.fps = st->avg_frame_rate.den ? av_q2d(st->avg_frame_rate) : 0.0;
  info_.frames = st->nb_frames;
  const char* cs = av_color_space_name(st->codecpar->color_space);
  const char* cr = av_color_range_name(st->codecpar->color_range);
  info_.color_space = cs ? cs : "unknown";
  info_.color_range = cr ? cr : "unknown";
}

NvdecSource::~NvdecSource() = default;

bool NvdecSource::read(GpuFrame& out) {
  Impl& s = *impl_;
  AVFrame* frame = av_frame_alloc();
  if (!frame) throw std::runtime_error("av_frame_alloc failed");
  while (true) {
    const int r = avcodec_receive_frame(s.dec, frame);
    if (r == 0) break;
    if (r == AVERROR_EOF || (r == AVERROR(EAGAIN) && s.draining)) {
      av_frame_free(&frame);
      return false;
    }
    if (r != AVERROR(EAGAIN)) {
      av_frame_free(&frame);
      check(r, "avcodec_receive_frame");
    }
    const int rd = av_read_frame(s.fmt, s.pkt);
    if (rd == AVERROR_EOF) {
      s.draining = true;
      check(avcodec_send_packet(s.dec, nullptr), "flush decoder");
      continue;
    }
    check(rd, "av_read_frame");
    if (s.pkt->stream_index == s.stream) check(avcodec_send_packet(s.dec, s.pkt), "send packet");
    av_packet_unref(s.pkt);
  }

  if (frame->format != AV_PIX_FMT_CUDA) {
    av_frame_free(&frame);
    throw std::runtime_error("decoder returned a non-CUDA frame (NVDEC not used)");
  }
  const auto* frames_ctx = reinterpret_cast<AVHWFramesContext*>(frame->hw_frames_ctx->data);
  if (frames_ctx->sw_format != AV_PIX_FMT_NV12) {
    const char* name = av_get_pix_fmt_name(frames_ctx->sw_format);
    av_frame_free(&frame);
    throw std::runtime_error(std::string("expected NV12 surfaces, got ") + (name ? name : "?"));
  }

  out.release();
  out.frame_ = frame;
  out.luma = frame->data[0];
  out.chroma = frame->data[1];
  out.luma_pitch = frame->linesize[0];
  out.chroma_pitch = frame->linesize[1];
  out.width = frame->width;
  out.height = frame->height;
  out.matrix = matrix_for(frame);
  out.full_range = frame->color_range == AVCOL_RANGE_JPEG;
  return true;
}

}  // namespace vrt
