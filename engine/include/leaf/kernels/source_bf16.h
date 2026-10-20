#pragma once
#include "leaf/kernels/gemv_pair.h"
#include <cstdint>
#include <cstring>

namespace leaf::kernels {
inline float source_bf16_value(std::uint16_t value) noexcept {
    const std::uint32_t bits = std::uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline void expand_source_bf16(const std::uint16_t* source, float* output, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) output[i] = source_bf16_value(source[i]);
}

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
namespace source_bf16_detail {
__attribute__((target("avx2,fma"), always_inline)) inline __m256 load8(const std::uint16_t* values) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(values))), 16));
}
}

// Same accumulator lanes, FMA sequence, reduction and signed-zero addition as
// gemv_pair_f32. Callers check AVX2/FMA, two rows and K >= 8 divisible by eight.
// Storage expansion discards no source bits and performs no arithmetic rounding.
__attribute__((target("avx2,fma"))) inline void gemv_pair_source_bf16(
        const std::uint16_t* weights, const float* input, std::size_t columns, float* output) {
    const auto* second = weights + columns;
    auto a0 = _mm256_setzero_ps(), b0 = a0, c0 = a0, d0 = a0;
    auto a1 = a0, b1 = a0, c1 = a0, d1 = a0;
    std::size_t col = 0;
    for (; columns - col >= 32; col += 32) {
        auto x = _mm256_loadu_ps(input + col);
        a0 = _mm256_fmadd_ps(source_bf16_detail::load8(weights + col), x, a0);
        a1 = _mm256_fmadd_ps(source_bf16_detail::load8(second + col), x, a1);
        x = _mm256_loadu_ps(input + col + 8);
        b0 = _mm256_fmadd_ps(source_bf16_detail::load8(weights + col + 8), x, b0);
        b1 = _mm256_fmadd_ps(source_bf16_detail::load8(second + col + 8), x, b1);
        x = _mm256_loadu_ps(input + col + 16);
        c0 = _mm256_fmadd_ps(source_bf16_detail::load8(weights + col + 16), x, c0);
        c1 = _mm256_fmadd_ps(source_bf16_detail::load8(second + col + 16), x, c1);
        x = _mm256_loadu_ps(input + col + 24);
        d0 = _mm256_fmadd_ps(source_bf16_detail::load8(weights + col + 24), x, d0);
        d1 = _mm256_fmadd_ps(source_bf16_detail::load8(second + col + 24), x, d1);
    }
    a0 = _mm256_add_ps(_mm256_add_ps(a0, b0), _mm256_add_ps(c0, d0));
    a1 = _mm256_add_ps(_mm256_add_ps(a1, b1), _mm256_add_ps(c1, d1));
    for (; columns - col >= 8; col += 8) {
        const auto x = _mm256_loadu_ps(input + col);
        a0 = _mm256_fmadd_ps(source_bf16_detail::load8(weights + col), x, a0);
        a1 = _mm256_fmadd_ps(source_bf16_detail::load8(second + col), x, a1);
    }
    output[0] = 0.0f + gemv_pair_detail::reduce(a0);
    output[1] = 0.0f + gemv_pair_detail::reduce(a1);
}
#endif
}
