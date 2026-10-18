#pragma once

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

namespace vrt {

struct TensorInfo {
  std::string name;
  nvinfer1::DataType dtype;
  nvinfer1::Dims dims;
  std::size_t count = 0;  // elements
  std::size_t bytes = 0;
  bool is_input = false;
};

namespace detail {
struct TrtDeleter {
  template <typename T>
  void operator()(T* p) const {
    delete p;
  }
};
}  // namespace detail

// A deserialized engine: weights and chosen kernels, loaded once and shared by any number of
// execution contexts (one per concurrent stream). Static shapes only.
class TrtModel {
 public:
  static std::shared_ptr<TrtModel> load(const std::string& engine_path);

  nvinfer1::ICudaEngine& engine() { return *engine_; }
  const std::vector<TensorInfo>& tensors() const { return tensors_; }

 private:
  TrtModel() = default;
  std::unique_ptr<nvinfer1::IRuntime, detail::TrtDeleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, detail::TrtDeleter> engine_;
  std::vector<TensorInfo> tensors_;
};

// One execution context on a (possibly shared) model. The context holds the per-stream state
// (activation memory, tensor addresses), so each concurrent stream needs its own. The caller
// owns the device buffers and the stream.
class TrtEngine {
 public:
  explicit TrtEngine(const std::string& engine_path);  // loads a model just for this context
  explicit TrtEngine(std::shared_ptr<TrtModel> model);
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  const std::vector<TensorInfo>& tensors() const { return model_->tensors(); }
  const TensorInfo& input() const;   // the single input
  const TensorInfo& output() const;  // the single output

  // Bind a device pointer to a tensor by name. Must be called before enqueue().
  void bind(const std::string& name, void* device_ptr);

  // Enqueue inference on `stream`. Returns immediately; synchronize the stream (or an
  // event recorded after this) before reading outputs. Replays the CUDA graph if one has
  // been captured, otherwise calls enqueueV3.
  void enqueue(cudaStream_t stream);

  // Record one enqueueV3 into a CUDA graph, so later enqueue() calls are a single
  // cudaGraphLaunch instead of ~160 separate kernel launches. Bind all tensors first: the
  // graph bakes in their addresses, so they must not change afterwards.
  void capture_graph(cudaStream_t stream);
  bool has_graph() const { return graph_exec_ != nullptr; }

 private:
  std::shared_ptr<TrtModel> model_;
  std::unique_ptr<nvinfer1::IExecutionContext, detail::TrtDeleter> context_;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
};

std::size_t dtype_size(nvinfer1::DataType t);
const char* dtype_name(nvinfer1::DataType t);
std::string tensorrt_version();

}  // namespace vrt
