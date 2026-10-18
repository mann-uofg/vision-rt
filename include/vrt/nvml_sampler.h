#pragma once

// Samples GPU and decoder (NVDEC) utilisation through NVML on a background thread, so a
// benchmark can report how busy each engine was while it ran.

#include <nvml.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace vrt {

class UtilSampler {
 public:
  UtilSampler() {
    if (nvmlInit_v2() == NVML_SUCCESS && nvmlDeviceGetHandleByIndex_v2(0, &dev_) == NVML_SUCCESS)
      ok_ = true;
  }
  ~UtilSampler() {
    stop();
    if (ok_) nvmlShutdown();
  }
  UtilSampler(const UtilSampler&) = delete;
  UtilSampler& operator=(const UtilSampler&) = delete;

  void start() {
    if (!ok_ || running_) return;
    running_ = true;
    thread_ = std::thread([this] {
      while (running_) {
        nvmlUtilization_t u{};
        unsigned dec = 0, period = 0;
        if (nvmlDeviceGetUtilizationRates(dev_, &u) == NVML_SUCCESS) gpu_.push_back(u.gpu);
        if (nvmlDeviceGetDecoderUtilization(dev_, &dec, &period) == NVML_SUCCESS)
          dec_.push_back(dec);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });
  }
  void stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
  }
  // Mean utilisation in percent, or -1 if NVML wasn't available. Call after stop().
  double gpu_mean() const { return mean(gpu_); }
  double decoder_mean() const { return mean(dec_); }

 private:
  static double mean(const std::vector<double>& v) {
    if (v.empty()) return -1;
    double s = 0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
  }
  nvmlDevice_t dev_{};
  bool ok_ = false;
  std::atomic<bool> running_{false};
  std::thread thread_;
  std::vector<double> gpu_, dec_;  // written by the sampling thread, read after it joins
};

}  // namespace vrt
