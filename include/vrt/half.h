#pragma once

#include <cstddef>
#include <cstdint>

namespace vrt {

// IEEE 754 binary16 bit pattern from a float, round to nearest even. Host-side only;
// used to feed FP16 engines from the CPU preprocessing path.
std::uint16_t float_to_half(float f);
float half_to_float(std::uint16_t h);

void floats_to_halves(const float* src, std::uint16_t* dst, std::size_t n);

}  // namespace vrt
