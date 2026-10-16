#include "vrt/half.h"

#include <bit>

namespace vrt {

std::uint16_t float_to_half(float f) {
  const std::uint32_t x = std::bit_cast<std::uint32_t>(f);
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  const std::uint32_t exp = (x >> 23) & 0xFFu;
  std::uint32_t mant = x & 0x7FFFFFu;

  if (exp == 0xFF) {  // inf or NaN (keep NaN quiet and non-zero)
    return static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? 0x200u | (mant >> 13) : 0u));
  }
  const int e = static_cast<int>(exp) - 127 + 15;
  if (e >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);  // overflow -> inf
  if (e <= 0) {
    // Subnormal half (or zero). Shift the mantissa, with its implicit 1, into place.
    if (e < -10) return static_cast<std::uint16_t>(sign);
    mant |= 0x800000u;
    const int shift = 14 - e;
    std::uint32_t h = mant >> shift;
    const std::uint32_t rem = mant & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (h & 1u))) ++h;
    return static_cast<std::uint16_t>(sign | h);
  }
  std::uint32_t h = (static_cast<std::uint32_t>(e) << 10) | (mant >> 13);
  const std::uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;  // may carry into the exponent: fine
  return static_cast<std::uint16_t>(sign | h);
}

float half_to_float(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  std::uint32_t mant = h & 0x3FFu;
  std::uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {  // subnormal: normalise
      int e = -1;
      do {
        mant <<= 1;
        ++e;
      } while (!(mant & 0x400u));
      bits = sign | (static_cast<std::uint32_t>(127 - 15 - e) << 23) | ((mant & 0x3FFu) << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  return std::bit_cast<float>(bits);
}

void floats_to_halves(const float* src, std::uint16_t* dst, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) dst[i] = float_to_half(src[i]);
}

}  // namespace vrt
