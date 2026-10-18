#include "vrt/batch_inference.h"

#include <algorithm>
#include <stdexcept>

#include "vrt/cuda_check.h"
#include "vrt/nvtx.h"

namespace vrt {

BatchInference::BatchInference(const std::vector<std::string>& engine_paths, cudaStream_t stream,
                               float conf, float iou, int max_det, int sets)
    : stream_(stream), conf_(conf), iou_(iou), max_det_(max_det) {
  if (engine_paths.empty()) throw std::invalid_argument("no engines");
  if (sets < 1) throw std::invalid_argument("need at least one result set");
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
  num_anchors_ = static_cast<int>(big_out.dims.d[1]);
  num_classes_ = static_cast<int>(big_out.dims.d[2]) - 5;

  VRT_CUDA_CHECK(cudaMalloc(&d_in_, big_in.bytes));
  VRT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out_), big_out.bytes));
  for (Engine& e : engines_) {
    e.ctx->bind(e.ctx->input().name, d_in_);
    e.ctx->bind(e.ctx->output().name, d_out_);
    e.ctx->capture_graph(stream_);
  }

  post_ = std::make_unique<GpuBatchPostprocess>(max_batch_, num_anchors_, num_classes_, max_det_);
  result_bytes_ = post_->result_bytes();
  sets_.resize(static_cast<std::size_t>(sets));
  for (ResultSet& s : sets_) {
    VRT_CUDA_CHECK(cudaHostAlloc(&s.host, result_bytes_ * static_cast<std::size_t>(max_batch_),
                                 cudaHostAllocDefault));
    for (auto& e : s.ev) VRT_CUDA_CHECK(cudaEventCreate(&e));
  }
}

BatchInference::~BatchInference() {
  cudaStreamSynchronize(stream_);
  for (auto& [k, g] : post_graphs_) cudaGraphExecDestroy(g);
  for (ResultSet& s : sets_) {
    for (auto& e : s.ev) cudaEventDestroy(e);
    cudaFreeHost(s.host);
  }
  cudaFree(d_out_);
  cudaFree(d_in_);
}

const BatchInference::Engine& BatchInference::pick(int k) const {
  for (const Engine& e : engines_)
    if (e.batch >= k) return e;
  throw std::invalid_argument("batch larger than the largest engine");
}

// Batched NMS over the first k slots, as one graph. Captured on first use for each k;
// recaptured for all k if the letterbox scale changes (different frame size).
cudaGraphExec_t BatchInference::post_graph(int k, float scale) {
  if (scale != graph_scale_) {
    VRT_CUDA_CHECK(cudaStreamSynchronize(stream_));
    for (auto& [n, g] : post_graphs_) VRT_CUDA_CHECK(cudaGraphExecDestroy(g));
    post_graphs_.clear();
    graph_scale_ = scale;
  }
  auto it = post_graphs_.find(k);
  if (it != post_graphs_.end()) return it->second;

  VRT_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
  post_->run(d_out_, k, scale, conf_, iou_, stream_);
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

int BatchInference::submit(const std::vector<BatchItem>& items) {
  NvtxRange range("batch submit");
  const int k = static_cast<int>(items.size());
  if (k < 1) throw std::invalid_argument("empty batch");
  const int ticket = next_set_;
  ResultSet& set = sets_[static_cast<std::size_t>(ticket)];
  if (set.pending) throw std::logic_error("submit: every result set is still in flight");
  const Engine& e = pick(k);
  const GpuFrame& first = *items[0].frame;
  if (!letterbox_ || letterbox_->src_w() != first.width || letterbox_->src_h() != first.height) {
    VRT_CUDA_CHECK(cudaStreamSynchronize(stream_));  // queued batches still use the old tables
    letterbox_ = std::make_unique<GpuLetterbox>(first.width, first.height,
                                                static_cast<int>(e.ctx->input().dims.d[3]),
                                                static_cast<int>(e.ctx->input().dims.d[2]));
  }
  for (const BatchItem& it : items)
    if (it.frame->width != first.width || it.frame->height != first.height)
      throw std::invalid_argument("all frames in a batch must have the same size");
  // Capture (if this k is new) before queuing anything of this batch on the stream.
  const cudaGraphExec_t post = post_graph(k, letterbox_->scale());

  VRT_CUDA_CHECK(cudaEventRecord(set.ev[0], stream_));
  for (int i = 0; i < k; ++i) {
    const BatchItem& it = items[static_cast<std::size_t>(i)];
    const GpuFrame& f = *it.frame;
    // The frame's surface copy was queued on its camera's decode stream.
    VRT_CUDA_CHECK(cudaStreamWaitEvent(stream_, it.ready, 0));
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
  VRT_CUDA_CHECK(cudaEventRecord(set.ev[1], stream_));
  e.ctx->enqueue(stream_);
  VRT_CUDA_CHECK(cudaEventRecord(set.ev[2], stream_));
  VRT_CUDA_CHECK(cudaGraphLaunch(post, stream_));
  VRT_CUDA_CHECK(cudaMemcpyAsync(set.host, post_->output(),
                                 result_bytes_ * static_cast<std::size_t>(k),
                                 cudaMemcpyDeviceToHost, stream_));
  VRT_CUDA_CHECK(cudaEventRecord(set.ev[3], stream_));
  set.batch = k;
  set.engine = e.batch;
  set.pending = true;
  next_set_ = (next_set_ + 1) % static_cast<int>(sets_.size());
  return ticket;
}

std::vector<Detections> BatchInference::wait(int ticket, BatchTimes* times) {
  NvtxRange range("batch wait");
  ResultSet& set = sets_.at(static_cast<std::size_t>(ticket));
  if (!set.pending) throw std::logic_error("wait: nothing in flight for this ticket");
  VRT_CUDA_CHECK(cudaEventSynchronize(set.ev[3]));
  set.pending = false;

  const int k = set.batch;
  std::vector<Detections> out(static_cast<std::size_t>(k));
  for (int i = 0; i < k; ++i) {
    const auto* res = reinterpret_cast<const GpuDetections*>(
        static_cast<const char*>(set.host) + static_cast<std::size_t>(i) * result_bytes_);
    Detections& d = out[static_cast<std::size_t>(i)];
    d.resize(static_cast<std::size_t>(res->count));
    for (int j = 0; j < res->count; ++j) {
      const float* r = res->dets + 6 * j;
      d[static_cast<std::size_t>(j)] = {r[0], r[1], r[2], r[3], r[4], static_cast<int>(r[5])};
    }
  }
  if (times) {
    times->batch = k;
    times->engine = set.engine;
    VRT_CUDA_CHECK(cudaEventElapsedTime(&times->preprocess_ms, set.ev[0], set.ev[1]));
    VRT_CUDA_CHECK(cudaEventElapsedTime(&times->infer_ms, set.ev[1], set.ev[2]));
    VRT_CUDA_CHECK(cudaEventElapsedTime(&times->postprocess_ms, set.ev[2], set.ev[3]));
    VRT_CUDA_CHECK(cudaEventElapsedTime(&times->total_ms, set.ev[0], set.ev[3]));
  }
  return out;
}

}  // namespace vrt
