cmake_minimum_required(VERSION 3.28)
project(vision_rt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)  # no C++20 modules; skips the clang-scan-deps requirement
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# GPU parts (TensorRT engine, apps) need CUDA and TensorRT. CI runners have neither, so they
# build the CPU core, the FFmpeg decoder and the tests only.
option(VRT_GPU "Build the CUDA/TensorRT parts" ON)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall$<SEMICOLON>-Wextra$<SEMICOLON>-Wpedantic$<SEMICOLON>-Werror>)

# --- FFmpeg (decode) ---------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libswscale)

# --- CPU core ----------------------------------------------------------------------------
add_library(vrt_core STATIC src/letterbox.cpp src/half.cpp src/postprocess.cpp src/stats.cpp)
target_include_directories(vrt_core PUBLIC include)
set_target_properties(vrt_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(vrt_video STATIC src/video_source.cpp)
target_link_libraries(vrt_video PUBLIC vrt_core PkgConfig::FFMPEG)

# C entry points for the Python tests (ctypes).
add_library(vrt_capi SHARED src/capi.cpp)
target_link_libraries(vrt_capi PRIVATE vrt_core)

enable_testing()
add_executable(test_core tests/cpp/test_core.cpp)
target_link_libraries(test_core PRIVATE vrt_core)
add_test(NAME core COMMAND test_core)

# --- GPU ---------------------------------------------------------------------------------
if(VRT_GPU)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)  # RTX 50-series (Blackwell, sm_120)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)

  set(TRT_ROOT "$ENV{TRT_ROOT}" CACHE PATH "TensorRT install directory")
  find_path(TRT_INCLUDE_DIR NvInfer.h HINTS "${TRT_ROOT}/include" REQUIRED)
  find_library(TRT_NVINFER nvinfer HINTS "${TRT_ROOT}/lib" REQUIRED)
  add_library(TensorRT::nvinfer SHARED IMPORTED)
  set_target_properties(TensorRT::nvinfer PROPERTIES
    IMPORTED_LOCATION "${TRT_NVINFER}"
    INTERFACE_INCLUDE_DIRECTORIES "${TRT_INCLUDE_DIR}")

  # Kernels: warnings are errors, and no fused multiply-add, so their arithmetic rounds exactly
  # like the CPU reference (x86-64 builds without FMA) and the outputs can be compared bit for bit.
  add_library(vrt_kernels STATIC src/kernels/preprocess.cu src/kernels/postprocess.cu)
  target_include_directories(vrt_kernels PUBLIC include)
  target_link_libraries(vrt_kernels PUBLIC vrt_core CUDA::cudart)
  target_compile_options(vrt_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-Werror=all-warnings;-Xcompiler=-Wall,-Wextra>)
  set_target_properties(vrt_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

  add_library(vrt_gpu_capi SHARED src/gpu_capi.cpp)
  target_link_libraries(vrt_gpu_capi PRIVATE vrt_kernels)

  add_library(vrt_gpu STATIC src/trt_engine.cpp src/sync_pipeline.cpp src/nvdec_source.cpp
                      src/gpu_pipeline.cpp)
  target_link_libraries(vrt_gpu PUBLIC vrt_core vrt_video vrt_kernels TensorRT::nvinfer
                                       CUDA::cudart CUDA::cuda_driver)
  # TensorRT headers trip -Wpedantic/-Wextra; treat them as system headers.
  target_include_directories(vrt_gpu SYSTEM PUBLIC "${TRT_INCLUDE_DIR}")

  foreach(app vrt_detect vrt_coco vrt_decode_bench vrt_pipeline)
    add_executable(${app} apps/${app}.cpp)
    target_link_libraries(${app} PRIVATE vrt_gpu)
  endforeach()
endif()
