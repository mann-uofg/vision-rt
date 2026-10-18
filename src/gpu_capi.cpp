// C entry points into the GPU kernels for the Python tests (tests/test_gpu_kernels.py).
// Each call copies host inputs to the device, runs one kernel path, and copies results back.
// Errors are reported as a negative return value plus a message from vrt_gpu_last_error().

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "vrt/cuda_check.h"
#include "vrt/postprocess_gpu.h"
#include "vrt/preprocess_gpu.h"

namespace {

std::string g_last_error;

template <typename F>
int guarded(F&& f) {
  try {
    return f();
  } catch (const std::exception& e) {
    g_last_error = e.what();
    return -1;
  }
}

struct DeviceBuffer {
  void* p = nullptr;
  explicit DeviceBuffer(std::size_t bytes) { VRT_CUDA_CHECK(cudaMalloc(&p, bytes)); }
  ~DeviceBuffer() { cudaFree(p); }
};

}  // namespace

extern "C" {

const char* vrt_gpu_last_error() { return g_last_error.c_str(); }

// nv12: tightly packed Y plane (w*h) then interleaved UV (w*h/2). out: 3*dst_h*dst_w values,
// float32 or float16 bit patterns depending on `fp16`.
int vrt_gpu_nv12_letterbox(const std::uint8_t* nv12, int w, int h, int bt709, int full_range,
                           int dst_w, int dst_h, int fp16, void* out) {
  return guarded([&] {
    const std::size_t luma = static_cast<std::size_t>(w) * h;
    DeviceBuffer d_src(luma * 3 / 2);
    const std::size_t out_bytes = 3ull * dst_w * dst_h * (fp16 ? 2 : 4);
    DeviceBuffer d_out(out_bytes);
    VRT_CUDA_CHECK(cudaMemcpy(d_src.p, nv12, luma * 3 / 2, cudaMemcpyHostToDevice));

    vrt::Nv12View v;
    v.luma = static_cast<const std::uint8_t*>(d_src.p);
    v.chroma = v.luma + luma;
    v.luma_pitch = v.chroma_pitch = w;
    v.width = w;
    v.height = h;
    vrt::GpuLetterbox lb(w, h, dst_w, dst_h);
    lb.run(v, vrt::yuv_coeffs(bt709 != 0, full_range != 0), d_out.p,
           fp16 ? vrt::TensorFormat::kFloat16 : vrt::TensorFormat::kFloat32, nullptr);
    VRT_CUDA_CHECK(cudaMemcpy(out, d_out.p, out_bytes, cudaMemcpyDeviceToHost));
    return 0;
  });
}

// preds: host [num_anchors, 5 + num_classes]. out: max_det rows of 6 floats. Returns the
// number of detections kept (<= max_det).
int vrt_gpu_postprocess(const float* preds, int num_anchors, int num_classes, float scale,
                        float conf_thresh, float iou_thresh, int max_det, float* out) {
  return guarded([&] {
    const std::size_t bytes = sizeof(float) * num_anchors * (5ull + num_classes);
    DeviceBuffer d_preds(bytes);
    VRT_CUDA_CHECK(cudaMemcpy(d_preds.p, preds, bytes, cudaMemcpyHostToDevice));
    vrt::GpuPostprocess post(num_anchors, num_classes, max_det);
    post.run(static_cast<const float*>(d_preds.p), scale, conf_thresh, iou_thresh, nullptr);
    std::string host(post.output_bytes(), '\0');
    VRT_CUDA_CHECK(cudaMemcpy(host.data(), post.output(), host.size(), cudaMemcpyDeviceToHost));
    const auto* res = reinterpret_cast<const vrt::GpuDetections*>(host.data());
    std::memcpy(out, res->dets, sizeof(float) * 6 * static_cast<std::size_t>(res->count));
    return res->count;
  });
}

// The batched path. preds: host [batch, num_anchors, 5 + num_classes]. out: batch * max_det
// rows of 6 floats (image i's rows start at i * max_det); counts: batch ints.
int vrt_gpu_postprocess_batched(const float* preds, int batch, int num_anchors, int num_classes,
                                float scale, float conf_thresh, float iou_thresh, int max_det,
                                float* out, int* counts) {
  return guarded([&] {
    const std::size_t bytes = sizeof(float) * batch * num_anchors * (5ull + num_classes);
    DeviceBuffer d_preds(bytes);
    VRT_CUDA_CHECK(cudaMemcpy(d_preds.p, preds, bytes, cudaMemcpyHostToDevice));
    vrt::GpuBatchPostprocess post(batch, num_anchors, num_classes, max_det);
    post.run(static_cast<const float*>(d_preds.p), batch, scale, conf_thresh, iou_thresh, nullptr);
    std::string host(post.result_bytes() * static_cast<std::size_t>(batch), '\0');
    VRT_CUDA_CHECK(cudaMemcpy(host.data(), post.output(), host.size(), cudaMemcpyDeviceToHost));
    for (int i = 0; i < batch; ++i) {
      const auto* res = reinterpret_cast<const vrt::GpuDetections*>(
          host.data() + static_cast<std::size_t>(i) * post.result_bytes());
      counts[i] = res->count;
      std::memcpy(out + static_cast<std::size_t>(i) * max_det * 6, res->dets,
                  sizeof(float) * 6 * static_cast<std::size_t>(res->count));
    }
    return 0;
  });
}

}  // extern "C"
