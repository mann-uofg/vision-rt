#include "vrt/trt_engine.h"

#include <NvInferVersion.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "vrt/cuda_check.h"

namespace vrt {

namespace {

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::fprintf(stderr, "[TensorRT] %s\n", msg);
  }
};

Logger& logger() {
  static Logger instance;
  return instance;
}

std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

}  // namespace

std::size_t dtype_size(nvinfer1::DataType t) {
  using nvinfer1::DataType;
  switch (t) {
    case DataType::kFLOAT:
      return 4;
    case DataType::kHALF:
      return 2;
    case DataType::kBF16:
      return 2;
    case DataType::kINT8:
      return 1;
    case DataType::kUINT8:
      return 1;
    case DataType::kINT32:
      return 4;
    case DataType::kINT64:
      return 8;
    case DataType::kBOOL:
      return 1;
    default:
      throw std::runtime_error("unsupported tensor data type");
  }
}

const char* dtype_name(nvinfer1::DataType t) {
  using nvinfer1::DataType;
  switch (t) {
    case DataType::kFLOAT:
      return "float32";
    case DataType::kHALF:
      return "float16";
    case DataType::kBF16:
      return "bfloat16";
    case DataType::kINT8:
      return "int8";
    case DataType::kUINT8:
      return "uint8";
    case DataType::kINT32:
      return "int32";
    case DataType::kINT64:
      return "int64";
    case DataType::kBOOL:
      return "bool";
    default:
      return "other";
  }
}

std::string tensorrt_version() {
  return std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR) + "." +
         std::to_string(NV_TENSORRT_PATCH);
}

TrtEngine::TrtEngine(const std::string& engine_path) {
  runtime_.reset(nvinfer1::createInferRuntime(logger()));
  if (!runtime_) throw std::runtime_error("createInferRuntime failed");
  const std::vector<char> blob = read_file(engine_path);
  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine_) throw std::runtime_error("failed to deserialize " + engine_path);
  context_.reset(engine_->createExecutionContext());
  if (!context_) throw std::runtime_error("createExecutionContext failed");

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    TensorInfo t;
    t.name = engine_->getIOTensorName(i);
    t.dtype = engine_->getTensorDataType(t.name.c_str());
    t.dims = engine_->getTensorShape(t.name.c_str());
    t.is_input = engine_->getTensorIOMode(t.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
    t.count = 1;
    for (int d = 0; d < t.dims.nbDims; ++d) {
      if (t.dims.d[d] < 0)
        throw std::runtime_error("dynamic shape on " + t.name + " not supported");
      t.count *= static_cast<std::size_t>(t.dims.d[d]);
    }
    t.bytes = t.count * dtype_size(t.dtype);
    tensors_.push_back(t);
  }
}

TrtEngine::~TrtEngine() {
  if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
  if (graph_) cudaGraphDestroy(graph_);
}

const TensorInfo& TrtEngine::input() const {
  for (const auto& t : tensors_)
    if (t.is_input) return t;
  throw std::runtime_error("engine has no input");
}

const TensorInfo& TrtEngine::output() const {
  for (const auto& t : tensors_)
    if (!t.is_input) return t;
  throw std::runtime_error("engine has no output");
}

void TrtEngine::bind(const std::string& name, void* device_ptr) {
  if (!context_->setTensorAddress(name.c_str(), device_ptr))
    throw std::runtime_error("setTensorAddress failed for " + name);
}

void TrtEngine::enqueue(cudaStream_t stream) {
  if (graph_exec_) {
    VRT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
  } else if (!context_->enqueueV3(stream)) {
    throw std::runtime_error("enqueueV3 failed");
  }
}

void TrtEngine::capture_graph(cudaStream_t stream) {
  if (graph_exec_) return;
  // One ordinary run first: TensorRT does lazy setup on the first enqueue (allocations,
  // stream/event creation) that is not allowed inside a capture.
  if (!context_->enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed");
  VRT_CUDA_CHECK(cudaStreamSynchronize(stream));

  VRT_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  const bool ok = context_->enqueueV3(stream);
  cudaGraph_t graph = nullptr;
  const cudaError_t end = cudaStreamEndCapture(stream, &graph);
  if (!ok || end != cudaSuccess) {
    if (graph) cudaGraphDestroy(graph);
    throw std::runtime_error(std::string("capturing enqueueV3 into a CUDA graph failed: ") +
                             (ok ? cudaGetErrorString(end) : "enqueueV3 returned false"));
  }
  graph_ = graph;
  VRT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, 0));
}

}  // namespace vrt
