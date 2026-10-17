#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#define VRT_CUDA_CHECK(expr)                                                                     \
  do {                                                                                           \
    const cudaError_t vrt_err_ = (expr);                                                         \
    if (vrt_err_ != cudaSuccess) {                                                               \
      throw std::runtime_error(std::string(#expr) + " failed: " + cudaGetErrorString(vrt_err_) + \
                               " (" + __FILE__ + ":" + std::to_string(__LINE__) + ")");          \
    }                                                                                            \
  } while (0)

namespace vrt {

// Owns a non-blocking CUDA stream. Declare it before anything that queues work on it, so it
// is destroyed last.
class CudaStream {
 public:
  CudaStream() { VRT_CUDA_CHECK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking)); }
  ~CudaStream() {
    cudaStreamSynchronize(s_);
    cudaStreamDestroy(s_);
  }
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  operator cudaStream_t() const { return s_; }

 private:
  cudaStream_t s_ = nullptr;
};

}  // namespace vrt
