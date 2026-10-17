#pragma once

// Named ranges for Nsight Systems timelines. NVTX v3 is header-only and costs next to
// nothing when no profiler is attached.

#include <nvtx3/nvToolsExt.h>

namespace vrt {

class NvtxRange {
 public:
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace vrt
