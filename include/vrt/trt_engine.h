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

// A deserialized engine with one execution context. Static shapes only: the engine's I/O
// shapes are fixed at build time. The caller owns the device buffers and the stream.
class TrtEngine {
 public:
  explicit TrtEngine(const std::string& engine_path);
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const TensorInfo& input() const;   // the single input
  const TensorInfo& output() const;  // the single output

  // Bind a device pointer to a tensor by name. Must be called before enqueue().
  void bind(const std::string& name, void* device_ptr);

  // Enqueue inference on `stream`. Returns immediately; synchronize the stream (or an
  // event recorded after this) before reading outputs.
  void enqueue(cudaStream_t stream);

 private:
  struct Deleter {
    template <typename T>
    void operator()(T* p) const {
      delete p;
    }
  };
  std::unique_ptr<nvinfer1::IRuntime, Deleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, Deleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, Deleter> context_;
  std::vector<TensorInfo> tensors_;
};

std::size_t dtype_size(nvinfer1::DataType t);
const char* dtype_name(nvinfer1::DataType t);
std::string tensorrt_version();

}  // namespace vrt
