#include "vrt/gpu_pipeline.h"

#include <stdexcept>

#include "vrt/cuda_check.h"
#include "vrt/half.h"
#include "vrt/nvtx.h"

namespace vrt {

GpuPipeline::GpuPipeline(const std::string& engine_path, cudaStream_t stream, int max_det)
    : engine_(engine_path), stream_(stream) {
  const TensorInfo& in = engine_.input();
  const TensorInfo& out = engine_.output();
  if (in.dims.nbDims != 4 || in.dims.d[0] != 1 || in.dims.d[1] != 3)
    throw std::runtime_error("expected input [1, 3, H, W]");
  if (out.dims.nbDims != 3 || out.dims.d[0] != 1 || out.dtype != nvinfer1::DataType::kFLOAT)
    throw std::runtime_error("expected float32 output [1, anchors, 5 + classes]");
  if (in.dtype == nvinfer1::DataType::kHALF) {
    format_ = TensorFormat::kFloat16;
  } else if (in.dtype != nvinfer1::DataType::kFLOAT) {
    throw std::runtime_error("input must be float32 or float16");
  }
  in_h_ = static_cast<int>(in.dims.d[2]);
  in_w_ = static_cast<int>(in.dims.d[3]);

  VRT_CUDA_CHECK(cudaMalloc(&d_in_, in.bytes));
  VRT_CUDA_CHECK(cudaMalloc(&d_out_, out.bytes));
  engine_.bind(in.name, d_in_);
  engine_.bind(out.name, d_out_);
  engine_.capture_graph(stream_);

  post_ = std::make_unique<GpuPostprocess>(static_cast<int>(out.dims.d[1]),
                                           static_cast<int>(out.dims.d[2]) - 5, max_det);
  for (Slot& s : slots_) {
    VRT_CUDA_CHECK(cudaHostAlloc(&s.host, post_->output_bytes(), cudaHostAllocDefault));
    for (auto& e : s.ev) VRT_CUDA_CHECK(cudaEventCreate(&e));
  }
}

GpuPipeline::~GpuPipeline() {
  cudaStreamSynchronize(stream_);
  if (post_graph_) cudaGraphExecDestroy(post_graph_);
  for (Slot& s : slots_) {
    for (auto& e : s.ev) cudaEventDestroy(e);
    cudaFreeHost(s.host);
  }
  cudaFree(d_out_);
  cudaFree(d_in_);
}

void GpuPipeline::submit(const GpuFrame& frame, int slot, float conf_thresh, float iou_thresh) {
  NvtxRange range("submit");
  Slot& s = slots_[slot];
  if (!letterbox_ || letterbox_->src_w() != frame.width || letterbox_->src_h() != frame.height)
    letterbox_ = std::make_unique<GpuLetterbox>(frame.width, frame.height, in_w_, in_h_);
  s.scale = letterbox_->scale();

  Nv12View view;
  view.luma = frame.luma;
  view.chroma = frame.chroma;
  view.luma_pitch = frame.luma_pitch;
  view.chroma_pitch = frame.chroma_pitch;
  view.width = frame.width;
  view.height = frame.height;

  VRT_CUDA_CHECK(cudaEventRecord(s.ev[0], stream_));
  letterbox_->run(view, yuv_coeffs(frame.matrix == YuvMatrix::kBt709, frame.full_range), d_in_,
                  format_, stream_);
  VRT_CUDA_CHECK(cudaEventRecord(s.ev[1], stream_));
  engine_.enqueue(stream_);
  VRT_CUDA_CHECK(cudaEventRecord(s.ev[2], stream_));
  run_postprocess(s.scale, conf_thresh, iou_thresh);
  VRT_CUDA_CHECK(cudaEventRecord(s.ev[3], stream_));
  VRT_CUDA_CHECK(cudaMemcpyAsync(s.host, post_->output(), post_->output_bytes(),
                                 cudaMemcpyDeviceToHost, stream_));
  VRT_CUDA_CHECK(cudaEventRecord(s.ev[4], stream_));
}

void GpuPipeline::run_postprocess(float scale, float conf_thresh, float iou_thresh) {
  if (!post_graph_ || scale != graph_scale_ || conf_thresh != graph_conf_ ||
      iou_thresh != graph_iou_) {
    if (post_graph_) {
      VRT_CUDA_CHECK(cudaStreamSynchronize(stream_));  // the old graph may still be running
      VRT_CUDA_CHECK(cudaGraphExecDestroy(post_graph_));
      post_graph_ = nullptr;
    }
    // Capture records the launches instead of running them; work already queued on the
    // stream is not part of the graph.
    VRT_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    post_->run(static_cast<const float*>(d_out_), scale, conf_thresh, iou_thresh, stream_);
    cudaGraph_t graph = nullptr;
    VRT_CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
    const cudaError_t err = cudaGraphInstantiate(&post_graph_, graph, 0);
    cudaGraphDestroy(graph);
    VRT_CUDA_CHECK(err);
    graph_scale_ = scale;
    graph_conf_ = conf_thresh;
    graph_iou_ = iou_thresh;
  }
  VRT_CUDA_CHECK(cudaGraphLaunch(post_graph_, stream_));
}

Detections GpuPipeline::collect(int slot, GpuStageTimes* times) {
  NvtxRange range("collect");
  Slot& s = slots_[slot];
  VRT_CUDA_CHECK(cudaEventSynchronize(s.ev[4]));
  const auto* res = static_cast<const GpuDetections*>(s.host);
  Detections dets(static_cast<std::size_t>(res->count));
  for (int i = 0; i < res->count; ++i) {
    const float* d = res->dets + 6 * i;
    dets[static_cast<std::size_t>(i)] = {d[0], d[1], d[2], d[3], d[4], static_cast<int>(d[5])};
  }
  if (times) {
    VRT_CUDA_CHECK(cudaEventElapsedTime(&times->preprocess_ms, s.ev[0], s.ev[1]));
    VRT_CUDA_CHECK(cudaEventElapsedTime(&times->infer_ms, s.ev[1], s.ev[2]));
    VRT_CUDA_CHECK(cudaEventElapsedTime(&times->postprocess_ms, s.ev[2], s.ev[3]));
    VRT_CUDA_CHECK(cudaEventElapsedTime(&times->d2h_ms, s.ev[3], s.ev[4]));
    VRT_CUDA_CHECK(cudaEventElapsedTime(&times->total_ms, s.ev[0], s.ev[4]));
  }
  return dets;
}

std::vector<float> GpuPipeline::input_tensor() {
  VRT_CUDA_CHECK(cudaStreamSynchronize(stream_));
  const TensorInfo& in = engine_.input();
  std::vector<float> out(in.count);
  if (format_ == TensorFormat::kFloat32) {
    VRT_CUDA_CHECK(cudaMemcpy(out.data(), d_in_, in.bytes, cudaMemcpyDeviceToHost));
  } else {
    std::vector<std::uint16_t> half(in.count);
    VRT_CUDA_CHECK(cudaMemcpy(half.data(), d_in_, in.bytes, cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = half_to_float(half[i]);
  }
  return out;
}

}  // namespace vrt
