#pragma once
#include <cstddef>

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#include <immintrin.h>

namespace leaf::kernels {
namespace gemv_pair_detail {
__attribute__((target("avx2,fma"))) inline float reduce(__m256 value) {
    const auto halves = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    const auto pairs = _mm_add_ps(halves, _mm_movehl_ps(halves, halves));
    return _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 1)));
}
} // namespace gemv_pair_detail

// Caller checks AVX2/FMA and supplies two contiguous FP32 weight rows and
// nonoverlapping output, columns >= 8 divisible by 8. Scalar-tail dimensions
// retain dot_avx to avoid compiler-dependent scalar FMA contraction differences.
__attribute__((target("avx2,fma"))) inline void gemv_pair_f32(
        const float* weights, const float* input, std::size_t columns, float* output) {
    const auto* second = weights + columns;
    auto a0 = _mm256_setzero_ps(), b0 = a0, c0 = a0, d0 = a0;
    auto a1 = a0, b1 = a0, c1 = a0, d1 = a0;
    std::size_t col = 0;
    for (; columns - col >= 32; col += 32) {
        auto x = _mm256_loadu_ps(input + col);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(weights + col), x, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(second + col), x, a1);
        x = _mm256_loadu_ps(input + col + 8);
        b0 = _mm256_fmadd_ps(_mm256_loadu_ps(weights + col + 8), x, b0);
        b1 = _mm256_fmadd_ps(_mm256_loadu_ps(second + col + 8), x, b1);
        x = _mm256_loadu_ps(input + col + 16);
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(weights + col + 16), x, c0);
        c1 = _mm256_fmadd_ps(_mm256_loadu_ps(second + col + 16), x, c1);
        x = _mm256_loadu_ps(input + col + 24);
        d0 = _mm256_fmadd_ps(_mm256_loadu_ps(weights + col + 24), x, d0);
        d1 = _mm256_fmadd_ps(_mm256_loadu_ps(second + col + 24), x, d1);
    }
    a0 = _mm256_add_ps(_mm256_add_ps(a0, b0), _mm256_add_ps(c0, d0));
    a1 = _mm256_add_ps(_mm256_add_ps(a1, b1), _mm256_add_ps(c1, d1));
    for (; columns - col >= 8; col += 8) {
        const auto x = _mm256_loadu_ps(input + col);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(weights + col), x, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(second + col), x, a1);
    }
    float first_sum = gemv_pair_detail::reduce(a0), second_sum = gemv_pair_detail::reduce(a1);
    output[0] = 0.0f + first_sum;
    output[1] = 0.0f + second_sum;
}
} // namespace leaf::kernels
#endif
