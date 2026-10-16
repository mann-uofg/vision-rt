#include "vrt/video_source.h"

#include <stdexcept>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
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

// swscale colourspace constant for the frame's tagged matrix.
int sws_matrix(AVColorSpace cs, int height) {
  switch (cs) {
    case AVCOL_SPC_BT709:
      return SWS_CS_ITU709;
    case AVCOL_SPC_BT470BG:
    case AVCOL_SPC_SMPTE170M:
      return SWS_CS_ITU601;
    case AVCOL_SPC_BT2020_NCL:
    case AVCOL_SPC_BT2020_CL:
      return SWS_CS_BT2020;
    default:
      return height >= 720 ? SWS_CS_ITU709 : SWS_CS_ITU601;
  }
}

bool is_full_range(const AVFrame* f) {
  if (f->color_range == AVCOL_RANGE_JPEG) return true;
  if (f->color_range == AVCOL_RANGE_MPEG) return false;
  // Untagged: the "J" pixel formats are full range by definition.
  const auto fmt = static_cast<AVPixelFormat>(f->format);
  return fmt == AV_PIX_FMT_YUVJ420P || fmt == AV_PIX_FMT_YUVJ422P || fmt == AV_PIX_FMT_YUVJ444P;
}

// Deprecated J formats make swscale warn; map them to the plain ones (range is set separately).
AVPixelFormat dejpeg(AVPixelFormat f) {
  switch (f) {
    case AV_PIX_FMT_YUVJ420P:
      return AV_PIX_FMT_YUV420P;
    case AV_PIX_FMT_YUVJ422P:
      return AV_PIX_FMT_YUV422P;
    case AV_PIX_FMT_YUVJ444P:
      return AV_PIX_FMT_YUV444P;
    default:
      return f;
  }
}

}  // namespace

struct VideoSource::Impl {
  AVFormatContext* fmt = nullptr;
  AVCodecContext* dec = nullptr;
  SwsContext* sws = nullptr;
  AVPacket* pkt = nullptr;
  AVFrame* frame = nullptr;
  int stream = -1;
  bool draining = false;
  // Key for the cached swscale context.
  int sws_w = 0, sws_h = 0, sws_fmt = -1;

  ~Impl() {
    sws_freeContext(sws);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&dec);
    avformat_close_input(&fmt);
  }

  // Next decoded frame into `frame`; false at end of stream.
  bool decode_next() {
    while (true) {
      const int r = avcodec_receive_frame(dec, frame);
      if (r == 0) return true;
      if (r == AVERROR_EOF) return false;
      if (r != AVERROR(EAGAIN)) check(r, "avcodec_receive_frame");

      if (draining) return false;
      const int rd = av_read_frame(fmt, pkt);
      if (rd == AVERROR_EOF) {
        draining = true;
        check(avcodec_send_packet(dec, nullptr), "flush decoder");
        continue;
      }
      check(rd, "av_read_frame");
      if (pkt->stream_index == stream) check(avcodec_send_packet(dec, pkt), "avcodec_send_packet");
      av_packet_unref(pkt);
    }
  }

  void to_bgr(BgrImage& out) {
    const int w = frame->width, h = frame->height;
    if (!sws || w != sws_w || h != sws_h || frame->format != sws_fmt) {
      sws_freeContext(sws);
      const auto src_fmt = dejpeg(static_cast<AVPixelFormat>(frame->format));
      // Plain SWS_BILINEAR keeps swscale's SIMD yuv2rgb path. With ACCURATE_RND and
      // FULL_CHR_H_INT, 1080p decode + convert was several times slower here.
      sws = sws_getContext(w, h, src_fmt, w, h, AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr, nullptr,
                           nullptr);
      if (!sws) throw std::runtime_error("sws_getContext failed");
      const int* coefs = sws_getCoefficients(sws_matrix(frame->colorspace, h));
      // Output BGR is always full range; input range comes from the frame.
      sws_setColorspaceDetails(sws, coefs, is_full_range(frame) ? 1 : 0, coefs, 1, 0, 1 << 16,
                               1 << 16);
      sws_w = w;
      sws_h = h;
      sws_fmt = frame->format;
    }
    out.width = w;
    out.height = h;
    out.stride = w * 3;
    out.data.resize(static_cast<std::size_t>(out.stride) * h);
    std::uint8_t* dst[1] = {out.data.data()};
    const int dst_stride[1] = {out.stride};
    sws_scale(sws, frame->data, frame->linesize, 0, h, dst, dst_stride);
  }
};

VideoSource::VideoSource(const std::string& path, int threads) : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  check(avformat_open_input(&s.fmt, path.c_str(), nullptr, nullptr), "open " + path);
  check(avformat_find_stream_info(s.fmt, nullptr), "find stream info");
  const AVCodec* codec = nullptr;
  s.stream = av_find_best_stream(s.fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
  check(s.stream, "no video stream in " + path);

  const AVStream* st = s.fmt->streams[s.stream];
  s.dec = avcodec_alloc_context3(codec);
  if (!s.dec) throw std::runtime_error("avcodec_alloc_context3 failed");
  check(avcodec_parameters_to_context(s.dec, st->codecpar), "codec parameters");
  s.dec->thread_count = threads;
  check(avcodec_open2(s.dec, codec, nullptr), "open decoder");
  s.pkt = av_packet_alloc();
  s.frame = av_frame_alloc();
  if (!s.pkt || !s.frame) throw std::runtime_error("av alloc failed");

  info_.codec = codec->name;
  info_.width = st->codecpar->width;
  info_.height = st->codecpar->height;
  info_.fps = st->avg_frame_rate.den ? av_q2d(st->avg_frame_rate) : 0.0;
  info_.frames = st->nb_frames;
  const char* cs = av_color_space_name(st->codecpar->color_space);
  const char* cr = av_color_range_name(st->codecpar->color_range);
  info_.color_space = cs ? cs : "unknown";
  info_.color_range = cr ? cr : "unknown";
}

VideoSource::~VideoSource() = default;

bool VideoSource::read(BgrImage& out) {
  if (!impl_->decode_next()) return false;
  impl_->to_bgr(out);
  av_frame_unref(impl_->frame);
  return true;
}

std::string ffmpeg_version() { return av_version_info(); }

}  // namespace vrt
