#include "vrt/batch_inference.h"

#include <algorithm>
#include <stdexcept>

#include "vrt/cuda_check.h"
#include "vrt/nvtx.h"

namespace vrt {

BatchInference::BatchInference(const std::vector<std::string>& engine_paths, cudaStream_t stream,
                               float conf, float iou, int max_det)
    : stream_(stream), conf_(conf), iou_(iou), max_det_(max_det) {
  if (engine_paths.empty()) throw std::invalid_argument("no engines");
  for (const auto& path : engine_paths) {
    auto ctx = std::make_unique<TrtEngine>(TrtModel::load(path));
    const TensorInfo& in = ctx->input();
    const TensorInfo& out = ctx->output();
    if (in.dims.nbDims != 4 || in.dims.d[1] != 3 || out.dims.nbDims != 3)
      throw std::runtime_error(path + ": expected input [B, 3, H, W] and output [B, anchors, C]");
    if (out.dims.d[0] != in.dims.d[0]) throw std::runtime_error(path + ": batch mismatch");
    engines_.push_back({static_cast<int>(in.dims.d[0]), std::move(ctx)});
  }
  std::sort(engines_.begin(), engines_.end(),
            [](const Engine& a, const Engine& b) { return a.batch < b.batch; });
  max_batch_ = engines_.back().batch;

  // All engines share one input and one output buffer sized for the largest batch; a batch-k
  // engine uses the first k slots. That only works if every engine has the same per-image
  // shapes and types, so check.
  const TensorInfo& big_in = engines_.back().ctx->input();
  const TensorInfo& big_out = engines_.back().ctx->output();
  for (const Engine& e : engines_) {
    const TensorInfo& in = e.ctx->input();
    const TensorInfo& out = e.ctx->output();
    if (in.dtype != big_in.dtype ||
        in.bytes / static_cast<std::size_t>(e.batch) !=
            big_in.bytes / static_cast<std::size_t>(max_batch_) ||
        out.dtype != nvinfer1::DataType::kFLOAT ||
        out.count / static_cast<std::size_t>(e.batch) !=
            big_out.count / static_cast<std::size_t>(max_batch_))
      throw std::runtime_error("engines differ in per-image shape or type");
  }
  if (big_in.dtype == nvinfer1::DataType::kFLOAT) {
    format_ = TensorFormat::kFloat32;
  } else if (big_in.dtype != nvinfer1::DataType::kHALF) {
    throw std::runtime_error("input must be float16 or float32");
  }
  in_slot_bytes_ = big_in.bytes / static_cast<std::size_t>(max_batch_);
  out_slot_floats_ = big_out.count / static_cast<std::size_t>(max_batch_);
  num_anchors_ = static_cast<int>(big_out.dims.d[1]);
  num_classes_ = static_cast<int>(big_out.dims.d[2]) - 5;

  VRT_CUDA_CHECK(cudaMalloc(&d_in_, big_in.bytes));
  VRT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out_), big_out.bytes));
  for (Engine& e : engines_) {
    e.ctx->bind(e.ctx->input().name, d_in_);
    e.ctx->bind(e.ctx->output().name, d_out_);
    e.ctx->capture_graph(stream_);
  }

  for (int i = 0; i < max_batch_; ++i)
    post_.push_back(std::make_unique<GpuPostprocess>(num_anchors_, num_classes_, max_det_));
  result_bytes_ = post_[0]->output_bytes();
  VRT_CUDA_CHECK(cudaHostAlloc(&host_, result_bytes_ * static_cast<std::size_t>(max_batch_),
                               cudaHostAllocDefault));
  for (auto& e : ev_) VRT_CUDA_CHECK(cudaEventCreate(&e));
  VRT_CUDA_CHECK(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming));
  side_.resize(static_cast<std::size_t>(max_batch_));
  join_.resize(static_cast<std::size_t>(max_batch_));
  for (int i = 0; i < max_batch_; ++i) {
    VRT_CUDA_CHECK(
        cudaStreamCreateWithFlags(&side_[static_cast<std::size_t>(i)], cudaStreamNonBlocking));
    VRT_CUDA_CHECK(
        cudaEventCreateWithFlags(&join_[static_cast<std::size_t>(i)], cudaEventDisableTiming));
  }
}

BatchInference::~BatchInference() {
  cudaStreamSynchronize(stream_);
  for (auto& [k, g] : post_graphs_) cudaGraphExecDestroy(g);
  for (auto& e : ev_) cudaEventDestroy(e);
  for (auto& e : join_) cudaEventDestroy(e);
  for (auto& s : side_) cudaStreamDestroy(s);
  cudaEventDestroy(fork_);
  cudaFreeHost(host_);
  cudaFree(d_out_);
  cudaFree(d_in_);
}

const BatchInference::Engine& BatchInference::pick(int k) const {
  for (const Engine& e : engines_)
    if (e.batch >= k) return e;
  throw std::invalid_argument("batch larger than the largest engine");
}

// NMS for the first k slots plus their result copies, as one graph. Captured on first use for
// each k; recaptured for all k if the letterbox scale changes (different frame size).
cudaGraphExec_t BatchInference::post_graph(int k, float scale) {
  if (scale != graph_scale_) {
    VRT_CUDA_CHECK(cudaStreamSynchronize(stream_));
    for (auto& [n, g] : post_graphs_) VRT_CUDA_CHECK(cudaGraphExecDestroy(g));
    post_graphs_.clear();
    graph_scale_ = scale;
  }
  auto it = post_graphs_.find(k);
  if (it != post_graphs_.end()) return it->second;

  // Fork: each slot's NMS + result copy goes on its own side stream; the capture follows the
  // event dependencies, so the graph gets k parallel branches that join back on stream_.
  VRT_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
  VRT_CUDA_CHECK(cudaEventRecord(fork_, stream_));
  for (int i = 0; i < k; ++i) {
    const auto s = static_cast<std::size_t>(i);
    VRT_CUDA_CHECK(cudaStreamWaitEvent(side_[s], fork_, 0));
    post_[s]->run(d_out_ + s * out_slot_floats_, scale, conf_, iou_, side_[s]);
    VRT_CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(host_) + s * result_bytes_,
                                   post_[s]->output(), result_bytes_, cudaMemcpyDeviceToHost,
                                   side_[s]));
    VRT_CUDA_CHECK(cudaEventRecord(join_[s], side_[s]));
    VRT_CUDA_CHECK(cudaStreamWaitEvent(stream_, join_[s], 0));
  }
  cudaGraph_t graph = nullptr;
  VRT_CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
  cudaGraphExec_t exec = nullptr;
  const cudaError_t err = cudaGraphInstantiate(&exec, graph, 0);
  cudaGraphDestroy(graph);
  VRT_CUDA_CHECK(err);
  post_graphs_[k] = exec;
  return exec;
}

void BatchInference::prepare(int frame_w, int frame_h) {
  const TensorInfo& in = engines_.back().ctx->input();
  letterbox_ = std::make_unique<GpuLetterbox>(frame_w, frame_h, static_cast<int>(in.dims.d[3]),
                                              static_cast<int>(in.dims.d[2]));
  for (int k = 1; k <= max_batch_; ++k) post_graph(k, letterbox_->scale());
  VRT_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

std::vector<Detections> BatchInference::run(const std::vector<BatchItem>& items,
                                            BatchTimes* times) {
  NvtxRange range("batch");
  const int k = static_cast<int>(items.size());
  if (k < 1) return {};
  const Engine& e = pick(k);
  const GpuFrame& first = *items[0].frame;
  if (!letterbox_ || letterbox_->src_w() != first.width || letterbox_->src_h() != first.height)
    letterbox_ = std::make_unique<GpuLetterbox>(first.width, first.height,
                                                static_cast<int>(e.ctx->input().dims.d[3]),
                                                static_cast<int>(e.ctx->input().dims.d[2]));

  VRT_CUDA_CHECK(cudaEventRecord(ev_[0], stream_));
  for (int i = 0; i < k; ++i) {
    const GpuFrame& f = *items[static_cast<std::size_t>(i)].frame;
    if (f.width != first.width || f.height != first.height)
      throw std::invalid_argument("all frames in a batch must have the same size");
    // The frame's surface copy was queued on its camera's decode stream.
    VRT_CUDA_CHECK(cudaStreamWaitEvent(stream_, items[static_cast<std::size_t>(i)].ready, 0));
    Nv12View v;
    v.luma = f.luma;
    v.chroma = f.chroma;
    v.luma_pitch = f.luma_pitch;
    v.chroma_pitch = f.chroma_pitch;
    v.width = f.width;
    v.height = f.height;
    letterbox_->run(v, yuv_coeffs(f.matrix == YuvMatrix::kBt709, f.full_range),
                    static_cast<char*>(d_in_) + static_cast<std::size_t>(i) * in_slot_bytes_,
                    format_, stream_);
  }
  VRT_CUDA_CHECK(cudaEventRecord(ev_[1], stream_));
  e.ctx->enqueue(stream_);
  VRT_CUDA_CHECK(cudaEventRecord(ev_[2], stream_));
  VRT_CUDA_CHECK(cudaGraphLaunch(post_graph(k, letterbox_->scale()), stream_));
  VRT_CUDA_CHECK(cudaEventRecord(ev_[3], stream_));
  VRT_CUDA_CHECK(cudaEventSynchronize(ev_[3]));

  std::vector<Detections> out(static_cast<std::size_t>(k));
  for (int i = 0; i < k; ++i) {
    const auto* res = reinterpret_cast<const GpuDetections*>(
        static_cast<const char*>(host_) + static_cast<std::size_t>(i) * result_bytes_);
    Detections& d = out[static_cast<std::size_t>(i)];
    d.resize(static_cast<std::size_t>(res->count));
    for (int j = 0; j < res->count; ++j) {
      const float* r = res->dets + 6 * j;
      d[static_cast<std::size_t>(j)] = {r[0], r[1], r[2], r[3], r[4], static_cast<int>(r[5])};
    }
  }
  if (times) {
    times->batch = k;
    times->engine = e.batch;
    VRT_CUDA_CHECK(cudaEventElapsedTime(&times->preprocess_ms, ev_[0], ev_[1]));
    VRT_CUDA_CHECK(cudaEventElapsedTime(&times->infer_ms, ev_[1], ev_[2]));
    VRT_CUDA_CHECK(cudaEventElapsedTime(&times->postprocess_ms, ev_[2], ev_[3]));
    VRT_CUDA_CHECK(cudaEventElapsedTime(&times->total_ms, ev_[0], ev_[3]));
  }
  return out;
}

}  // namespace vrt
