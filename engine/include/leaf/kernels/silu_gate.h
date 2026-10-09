#pragma once
#include <cmath>
#include <cstddef>
#include <stdexcept>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#endif

namespace leaf::kernels {
inline float silu_gate_scalar(float up, float gate) {
    // Keep the decoder's operation order and float exp rounding.
    return up * (gate / (1.0f + std::exp(-gate)));
}

#if defined(__x86_64__) && defined(__GNUC__)
namespace silu_gate_detail {
__attribute__((target("avx2,no-fma"))) inline void vector(
        float* up, const float* gate, std::size_t size) {
    std::size_t i = 0;
    for (; size - i >= 4; i += 4) {
        const auto g = _mm_loadu_ps(gate + i);
        const auto x32 = _mm_xor_ps(g, _mm_set1_ps(-0.0f));
        const auto ordinary = _mm_and_ps(_mm_cmpge_ps(x32, _mm_set1_ps(-80.0f)),
                                         _mm_cmple_ps(x32, _mm_set1_ps(80.0f)));
        // Keep libm overflow, underflow, NaN, infinity and signed-zero behavior
        // for extreme inputs. No global fast-math or denormal-mode changes.
        if (_mm_movemask_ps(ordinary) != 15) {
            for (unsigned j = 0; j < 4; ++j)
                up[i + j] = silu_gate_scalar(up[i + j], gate[i + j]);
            continue;
        }
        // Same conservative range-reduction technique as vector softmax, here
        // extended to both signs. |r| <= ln(2)/2; the degree-13 Taylor remainder
        // is below 6e-18. This is not a universal correctly-rounded exp proof.
        const auto x = _mm256_cvtps_pd(x32);
        const auto n = _mm256_round_pd(_mm256_mul_pd(x, _mm256_set1_pd(1.4426950408889634074)),
                                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        auto r = _mm256_sub_pd(x, _mm256_mul_pd(n, _mm256_set1_pd(0.69314718036912381649)));
        r = _mm256_sub_pd(r, _mm256_mul_pd(n, _mm256_set1_pd(1.9082149292705877000e-10)));
        auto p = _mm256_set1_pd(1.0 / 6227020800.0);
        constexpr double coefficients[] = {1.0 / 479001600.0, 1.0 / 39916800.0, 1.0 / 3628800.0,
            1.0 / 362880.0, 1.0 / 40320.0, 1.0 / 5040.0, 1.0 / 720.0,
            1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0};
        for (double coefficient : coefficients)
            p = _mm256_add_pd(_mm256_mul_pd(p, r), _mm256_set1_pd(coefficient));
        const auto exponent = _mm256_slli_epi64(_mm256_add_epi64(
            _mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(n)), _mm256_set1_epi64x(1023)), 52);
        const auto exp32 = _mm256_cvtpd_ps(_mm256_mul_pd(p, _mm256_castsi256_pd(exponent)));
        const auto silu = _mm_div_ps(g, _mm_add_ps(_mm_set1_ps(1.0f), exp32));
        _mm_storeu_ps(up + i, _mm_mul_ps(_mm_loadu_ps(up + i), silu));
    }
    for (; i < size; ++i) up[i] = silu_gate_scalar(up[i], gate[i]);
}
} // namespace silu_gate_detail
#endif

// up and gate must be disjoint or identical ranges; partial overlap is unsupported.
// Unaligned storage and arbitrary element counts are supported. No allocation.
inline void silu_gate_inplace(float* up, const float* gate, std::size_t size, bool use_avx2 = true) {
    if (size && (!up || !gate)) throw std::invalid_argument("SiLU gating requires valid storage");
#if defined(__x86_64__) && defined(__GNUC__)
    if (use_avx2 && __builtin_cpu_supports("avx2")) {
        silu_gate_detail::vector(up, gate, size);
        return;
    }
#else
    (void)use_avx2;
#endif
    for (std::size_t i = 0; i < size; ++i) up[i] = silu_gate_scalar(up[i], gate[i]);
}
} // namespace leaf::kernels
