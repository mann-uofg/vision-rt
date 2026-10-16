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

add_compile_options(-Wall -Wextra -Wpedantic -Werror)

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
  find_package(CUDAToolkit REQUIRED)

  set(TRT_ROOT "$ENV{TRT_ROOT}" CACHE PATH "TensorRT install directory")
  find_path(TRT_INCLUDE_DIR NvInfer.h HINTS "${TRT_ROOT}/include" REQUIRED)
  find_library(TRT_NVINFER nvinfer HINTS "${TRT_ROOT}/lib" REQUIRED)
  add_library(TensorRT::nvinfer SHARED IMPORTED)
  set_target_properties(TensorRT::nvinfer PROPERTIES
    IMPORTED_LOCATION "${TRT_NVINFER}"
    INTERFACE_INCLUDE_DIRECTORIES "${TRT_INCLUDE_DIR}")

  add_library(vrt_gpu STATIC src/trt_engine.cpp src/sync_pipeline.cpp)
  target_link_libraries(vrt_gpu PUBLIC vrt_core vrt_video TensorRT::nvinfer CUDA::cudart)
  # TensorRT headers trip -Wpedantic/-Wextra; treat them as system headers.
  target_include_directories(vrt_gpu SYSTEM PUBLIC "${TRT_INCLUDE_DIR}")

  foreach(app vrt_detect vrt_coco)
    add_executable(${app} apps/${app}.cpp)
    target_link_libraries(${app} PRIVATE vrt_gpu)
  endforeach()
endif()
