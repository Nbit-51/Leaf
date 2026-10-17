#pragma once
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#if defined(__MINGW32__) && defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define LEAF_GELU_ERF_AVX2 1
#else
#define LEAF_GELU_ERF_AVX2 0
#endif

namespace leaf::kernels {
inline float gelu_erf_scalar(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475f));
}

// Experimental MinGW path: reproduce its existing erff operations, including
// separate multiply/add rounding. Other libm implementations retain scalar erf.
inline bool gelu_erf_vector_available() noexcept {
#if LEAF_GELU_ERF_AVX2
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

#if LEAF_GELU_ERF_AVX2
namespace gelu_erf_detail {
// Small/middle-domain coefficients and arithmetic adapted from sf_erf.c:
// https://sources.debian.org/src/mingw-w64/6.0.0-3/mingw-w64-crt/math/sf_erf.c/
// Float conversion by Ian Lance Taylor, Cygnus Support.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
//
// Only 2^-28 <= |erf argument| < 1.25 is vectorized. Tiny values, exponential
// tails and exceptional lanes execute the original scalar expression. No FMA,
// reciprocal approximation, activation substitution or precision reduction.
__attribute__((target("avx2,no-fma"))) inline void vector(float* values, std::size_t size) {
    const auto one = _mm256_set1_ps(1.0f), half = _mm256_set1_ps(0.5f);
    const auto sign = _mm256_set1_ps(-0.0f), zero = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; size - i >= 8; i += 8) {
        const auto original = _mm256_loadu_ps(values + i);
        const auto argument = _mm256_mul_ps(original, _mm256_set1_ps(0.7071067811865475f));
        const auto absolute = _mm256_andnot_ps(sign, argument);
        const auto fast = _mm256_and_ps(
            _mm256_cmp_ps(absolute, _mm256_set1_ps(0x1p-28f), _CMP_GE_OQ),
            _mm256_cmp_ps(absolute, _mm256_set1_ps(1.25f), _CMP_LT_OQ));
        const int lanes = _mm256_movemask_ps(fast);
        if (!lanes) {
            for (unsigned lane = 0; lane < 8; ++lane) values[i + lane] = gelu_erf_scalar(values[i + lane]);
            continue;
        }
        // Sanitize inactive lanes so discarded polynomial intermediates cannot
        // overflow on large finite inputs or introduce NaNs before fallback.
        const auto x = _mm256_blendv_ps(half, argument, fast);
        const auto z = _mm256_mul_ps(x, x);
        auto p = _mm256_set1_ps(-2.3763017452e-05f);
        p = _mm256_add_ps(_mm256_set1_ps(-5.7702702470e-03f), _mm256_mul_ps(z, p));
        p = _mm256_add_ps(_mm256_set1_ps(-2.8481749818e-02f), _mm256_mul_ps(z, p));
        p = _mm256_add_ps(_mm256_set1_ps(-3.2504209876e-01f), _mm256_mul_ps(z, p));
        p = _mm256_add_ps(_mm256_set1_ps(1.2837916613e-01f), _mm256_mul_ps(z, p));
        auto q = _mm256_set1_ps(-3.9602282413e-06f);
        q = _mm256_add_ps(_mm256_set1_ps(1.3249473704e-04f), _mm256_mul_ps(z, q));
        q = _mm256_add_ps(_mm256_set1_ps(5.0813062117e-03f), _mm256_mul_ps(z, q));
        q = _mm256_add_ps(_mm256_set1_ps(6.5022252500e-02f), _mm256_mul_ps(z, q));
        q = _mm256_add_ps(_mm256_set1_ps(3.9791721106e-01f), _mm256_mul_ps(z, q));
        q = _mm256_add_ps(one, _mm256_mul_ps(z, q));
        auto erf = _mm256_add_ps(x, _mm256_mul_ps(x, _mm256_div_ps(p, q)));
        const auto middle = _mm256_and_ps(fast,
            _mm256_cmp_ps(absolute, _mm256_set1_ps(0.84375f), _CMP_GE_OQ));
        if (_mm256_movemask_ps(middle)) {
            const auto t = _mm256_sub_ps(_mm256_andnot_ps(sign, x), one);
            p = _mm256_set1_ps(-2.1663755178e-03f);
            p = _mm256_add_ps(_mm256_set1_ps(3.5478305072e-02f), _mm256_mul_ps(t, p));
            p = _mm256_add_ps(_mm256_set1_ps(-1.1089469492e-01f), _mm256_mul_ps(t, p));
            p = _mm256_add_ps(_mm256_set1_ps(3.1834661961e-01f), _mm256_mul_ps(t, p));
            p = _mm256_add_ps(_mm256_set1_ps(-3.7220788002e-01f), _mm256_mul_ps(t, p));
            p = _mm256_add_ps(_mm256_set1_ps(4.1485610604e-01f), _mm256_mul_ps(t, p));
            p = _mm256_add_ps(_mm256_set1_ps(-2.3621185683e-03f), _mm256_mul_ps(t, p));
            q = _mm256_set1_ps(1.1984500103e-02f);
            q = _mm256_add_ps(_mm256_set1_ps(1.3637083583e-02f), _mm256_mul_ps(t, q));
            q = _mm256_add_ps(_mm256_set1_ps(1.2617121637e-01f), _mm256_mul_ps(t, q));
            q = _mm256_add_ps(_mm256_set1_ps(7.1828655899e-02f), _mm256_mul_ps(t, q));
            q = _mm256_add_ps(_mm256_set1_ps(5.4039794207e-01f), _mm256_mul_ps(t, q));
            q = _mm256_add_ps(_mm256_set1_ps(1.0642088205e-01f), _mm256_mul_ps(t, q));
            q = _mm256_add_ps(one, _mm256_mul_ps(t, q));
            const auto ratio = _mm256_div_ps(p, q);
            const auto positive = _mm256_add_ps(_mm256_set1_ps(8.4506291151e-01f), ratio);
            const auto negative = _mm256_sub_ps(_mm256_set1_ps(-8.4506291151e-01f), ratio);
            erf = _mm256_blendv_ps(erf, _mm256_blendv_ps(positive, negative,
                _mm256_cmp_ps(x, zero, _CMP_LT_OQ)), middle);
        }
        const auto safe_original = _mm256_blendv_ps(zero, original, fast);
        const auto result = _mm256_mul_ps(_mm256_mul_ps(half, safe_original), _mm256_add_ps(one, erf));
        alignas(32) float computed[8];
        _mm256_store_ps(computed, result);
        for (unsigned lane = 0; lane < 8; ++lane)
            values[i + lane] = lanes & (1 << lane) ? computed[lane] : gelu_erf_scalar(values[i + lane]);
    }
    for (; i < size; ++i) values[i] = gelu_erf_scalar(values[i]);
}
} // namespace gelu_erf_detail
#endif

inline void gelu_erf_inplace(float* values, std::size_t size, bool use_avx2 = true) {
    if (!values && size) throw std::invalid_argument("GELU-erf requires valid storage");
#if LEAF_GELU_ERF_AVX2
    if (use_avx2 && gelu_erf_vector_available() && std::fegetround() == FE_TONEAREST) {
        gelu_erf_detail::vector(values, size);
        return;
    }
#else
    (void)use_avx2;
#endif
    for (std::size_t i = 0; i < size; ++i) values[i] = gelu_erf_scalar(values[i]);
}
} // namespace leaf::kernels
