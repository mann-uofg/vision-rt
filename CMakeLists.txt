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
