#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vrt {

// Interleaved 8-bit BGR image, rows `stride` bytes apart.
struct BgrImage {
  std::vector<std::uint8_t> data;
  int width = 0, height = 0, stride = 0;
};

struct VideoInfo {
  std::string codec;
  int width = 0, height = 0;
  double fps = 0;
  long long frames = 0;  // from the container; 0 if unknown
  std::string color_space, color_range;
};

// CPU decode with FFmpeg (libavformat + libavcodec), converted to BGR with libswscale.
// Opens anything FFmpeg can: video files, and single images (COCO JPEGs) via image2.
//
// YUV -> BGR uses the matrix and range the stream says it has (BT.709 limited for the barn
// clips, BT.601 full range for JPEG). Untagged streams fall back to BT.709 for HD and
// BT.601 for SD, which is what most players do.
class VideoSource {
 public:
  // threads: decoder threads; 0 lets FFmpeg pick (frame threading on all cores).
  explicit VideoSource(const std::string& path, int threads = 0);
  ~VideoSource();
  VideoSource(const VideoSource&) = delete;
  VideoSource& operator=(const VideoSource&) = delete;

  // Decodes the next frame into `out`. Returns false at end of stream.
  bool read(BgrImage& out);
  const VideoInfo& info() const { return info_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  VideoInfo info_;
};

std::string ffmpeg_version();

}  // namespace vrt
