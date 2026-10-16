#include "vrt/sync_pipeline.h"

#include <chrono>
#include <stdexcept>

#include "vrt/cuda_check.h"
#include "vrt/half.h"
#include "vrt/letterbox.h"
#include "vrt/postprocess.h"

namespace vrt {

namespace {

double ms_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

SyncPipeline::SyncPipeline(const std::string& engine_path) : engine_(engine_path) {
  const TensorInfo& in = engine_.input();
  const TensorInfo& out = engine_.output();
  if (in.dims.nbDims != 4 || in.dims.d[0] != 1 || in.dims.d[1] != 3)
    throw std::runtime_error("expected input [1, 3, H, W]");
  if (out.dims.nbDims != 3 || out.dims.d[0] != 1 || out.dtype != nvinfer1::DataType::kFLOAT)
    throw std::runtime_error("expected float32 output [1, anchors, 5 + classes]");
  if (in.dtype != nvinfer1::DataType::kFLOAT && in.dtype != nvinfer1::DataType::kHALF)
    throw std::runtime_error("input must be float32 or float16");

  in_h_ = static_cast<int>(in.dims.d[2]);
  in_w_ = static_cast<int>(in.dims.d[3]);
  num_anchors_ = static_cast<int>(out.dims.d[1]);
  num_classes_ = static_cast<int>(out.dims.d[2]) - 5;
  fp16_input_ = in.dtype == nvinfer1::DataType::kHALF;

  if (fp16_input_) {
    chw_half_.resize(in.count);
  } else {
    chw_.resize(in.count);
  }
  preds_.resize(out.count);

  VRT_CUDA_CHECK(cudaMalloc(&d_in_, in.bytes));
  VRT_CUDA_CHECK(cudaMalloc(&d_out_, out.bytes));
  engine_.bind(in.name, d_in_);
  engine_.bind(out.name, d_out_);
  VRT_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  for (auto& e : ev_) VRT_CUDA_CHECK(cudaEventCreate(&e));
}

SyncPipeline::~SyncPipeline() {
  for (auto& e : ev_) cudaEventDestroy(e);
  cudaStreamDestroy(stream_);
  cudaFree(d_out_);
  cudaFree(d_in_);
}

Detections SyncPipeline::run(const BgrImage& frame, float conf_thresh, float iou_thresh,
                             StageTimes* times) {
  using clock = std::chrono::steady_clock;

  auto t = clock::now();
  const void* host_in;
  if (fp16_input_) {
    letterbox_bgr_to_chw_half(frame.data.data(), frame.width, frame.height, frame.stride, in_w_,
                              in_h_, chw_half_.data());
    host_in = chw_half_.data();
  } else {
    letterbox_bgr_to_chw(frame.data.data(), frame.width, frame.height, frame.stride, in_w_, in_h_,
                         chw_.data());
    host_in = chw_.data();
  }
  const double pre_ms = ms_since(t);

  t = clock::now();
  // Pageable host memory on purpose: same as the baseline. Pinned buffers come in M2.
  VRT_CUDA_CHECK(cudaEventRecord(ev_[0], stream_));
  VRT_CUDA_CHECK(
      cudaMemcpyAsync(d_in_, host_in, engine_.input().bytes, cudaMemcpyHostToDevice, stream_));
  VRT_CUDA_CHECK(cudaEventRecord(ev_[1], stream_));
  engine_.enqueue(stream_);
  VRT_CUDA_CHECK(cudaEventRecord(ev_[2], stream_));
  VRT_CUDA_CHECK(cudaMemcpyAsync(preds_.data(), d_out_, engine_.output().bytes,
                                 cudaMemcpyDeviceToHost, stream_));
  VRT_CUDA_CHECK(cudaEventRecord(ev_[3], stream_));
  VRT_CUDA_CHECK(cudaStreamSynchronize(stream_));
  const double gpu_ms = ms_since(t);

  t = clock::now();
  const float scale = letterbox_geometry(frame.width, frame.height, in_w_, in_h_).scale;
  Detections dets =
      postprocess(preds_.data(), num_anchors_, num_classes_, scale, conf_thresh, iou_thresh);
  const double post_ms = ms_since(t);

  if (times) {
    times->preprocess_ms = pre_ms;
    times->gpu_ms = gpu_ms;
    times->postprocess_ms = post_ms;
    VRT_CUDA_CHECK(cudaEventElapsedTime(&times->h2d_ms, ev_[0], ev_[1]));
    VRT_CUDA_CHECK(cudaEventElapsedTime(&times->infer_ms, ev_[1], ev_[2]));
    VRT_CUDA_CHECK(cudaEventElapsedTime(&times->d2h_ms, ev_[2], ev_[3]));
  }
  return dets;
}

std::vector<float> SyncPipeline::last_input() const {
  if (!fp16_input_) return chw_;
  std::vector<float> out(chw_half_.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = half_to_float(chw_half_[i]);
  return out;
}

void write_environment(JsonWriter& j) {
  int dev = 0, runtime = 0, driver = 0;
  cudaDeviceProp prop{};
  VRT_CUDA_CHECK(cudaGetDevice(&dev));
  VRT_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  VRT_CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  VRT_CUDA_CHECK(cudaDriverGetVersion(&driver));
  auto ver = [](int v) { return std::to_string(v / 1000) + "." + std::to_string(v % 1000 / 10); };
  j.field("gpu", prop.name)
      .field("compute_capability", std::to_string(prop.major) + "." + std::to_string(prop.minor))
      .field("cuda_runtime", ver(runtime))
      .field("cuda_driver_api", ver(driver))
      .field("tensorrt", tensorrt_version())
      .field("ffmpeg", ffmpeg_version());
}

}  // namespace vrt
