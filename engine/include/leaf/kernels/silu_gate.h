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
template <unsigned Streams>
__attribute__((target("avx2,no-fma"), always_inline)) inline void batch(float* up, const float* gate) {
    static_assert(Streams == 1 || Streams == 4, "SiLU batches use one or four independent vectors");
    bool ordinary_batch = true;
#pragma GCC unroll 4
    for (unsigned stream = 0; stream < Streams; ++stream) {
        const auto g = _mm_loadu_ps(gate + stream * 4);
        const auto x32 = _mm_xor_ps(g, _mm_set1_ps(-0.0f));
        const auto ordinary = _mm_and_ps(_mm_cmpge_ps(x32, _mm_set1_ps(-80.0f)),
                                         _mm_cmple_ps(x32, _mm_set1_ps(80.0f)));
        ordinary_batch = ordinary_batch && _mm_movemask_ps(ordinary) == 15;
    }
    // Preserve the old four-lane fallback boundaries, even when one of the
    // other streams has an extreme input. Partial overlap remains unsupported.
    if (!ordinary_batch) {
        if constexpr (Streams == 1) {
            for (unsigned j = 0; j < 4; ++j)
                up[j] = silu_gate_scalar(up[j], gate[j]);
        } else {
            for (unsigned stream = 0; stream < Streams; ++stream)
                batch<1>(up + stream * 4, gate + stream * 4);
        }
        return;
    }
    __m256d n[Streams], r[Streams], p[Streams];
#pragma GCC unroll 4
    for (unsigned stream = 0; stream < Streams; ++stream) {
        const auto x32 = _mm_xor_ps(_mm_loadu_ps(gate + stream * 4), _mm_set1_ps(-0.0f));
        const auto x = _mm256_cvtps_pd(x32);
        n[stream] = _mm256_round_pd(_mm256_mul_pd(x, _mm256_set1_pd(1.4426950408889634074)),
                                   _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        r[stream] = _mm256_sub_pd(x, _mm256_mul_pd(n[stream], _mm256_set1_pd(0.69314718036912381649)));
        r[stream] = _mm256_sub_pd(r[stream], _mm256_mul_pd(n[stream], _mm256_set1_pd(1.9082149292705877000e-10)));
        p[stream] = _mm256_set1_pd(1.0 / 6227020800.0);
    }
    // Interleave independent Horner chains. Each lane retains the same FP64
    // operations, coefficients and FP32 rounding as the established kernel.
    // |r| <= ln(2)/2; the degree-13 remainder is below 6e-18. This still is
    // not a universal correctly-rounded exp proof, and FMA remains disabled.
    constexpr double coefficients[] = {1.0 / 479001600.0, 1.0 / 39916800.0, 1.0 / 3628800.0,
        1.0 / 362880.0, 1.0 / 40320.0, 1.0 / 5040.0, 1.0 / 720.0,
        1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0};
    for (double coefficient : coefficients) {
#pragma GCC unroll 4
        for (unsigned stream = 0; stream < Streams; ++stream)
            p[stream] = _mm256_add_pd(_mm256_mul_pd(p[stream], r[stream]), _mm256_set1_pd(coefficient));
    }
    // Gate values are reloaded after the polynomial rather than occupying
    // four more vector registers throughout all the Horner chains.
    asm volatile("" ::: "memory");
#pragma GCC unroll 4
    for (unsigned stream = 0; stream < Streams; ++stream) {
        const auto exponent = _mm256_slli_epi64(_mm256_add_epi64(
            _mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(n[stream])), _mm256_set1_epi64x(1023)), 52);
        const auto exp32 = _mm256_cvtpd_ps(_mm256_mul_pd(p[stream], _mm256_castsi256_pd(exponent)));
        const auto g = _mm_loadu_ps(gate + stream * 4);
        const auto silu = _mm_div_ps(g, _mm_add_ps(_mm_set1_ps(1.0f), exp32));
        _mm_storeu_ps(up + stream * 4, _mm_mul_ps(_mm_loadu_ps(up + stream * 4), silu));
    }
}

__attribute__((target("avx2,no-fma"))) inline void vector(
        float* up, const float* gate, std::size_t size) {
    std::size_t i = 0;
    for (; size - i >= 16; i += 16) batch<4>(up + i, gate + i);
    for (; size - i >= 4; i += 4) batch<1>(up + i, gate + i);
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
