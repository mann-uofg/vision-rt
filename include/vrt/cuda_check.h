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
