#pragma once
#include <cmath>
#include <cstddef>
#include <stdexcept>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#endif

namespace leaf::kernels {
inline float gelu_new_scalar(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f *
                                      (x + 0.044715f * x * x * x)));
}

#if defined(__x86_64__) && defined(__GNUC__)
namespace gelu_detail {
__attribute__((target("avx2,fma"))) inline void vector(float* values, std::size_t size) {
    const auto one = _mm256_set1_ps(1.0f), half = _mm256_set1_ps(0.5f);
    const auto bound = _mm256_set1_ps(8.0f), lower = _mm256_set1_ps(-8.0f);
    std::size_t i = 0;
    for (; size - i >= 8; i += 8) {
        const auto original = _mm256_loadu_ps(values + i);
        // Preserve the scalar expression's NaN/Inf behavior, including -Inf.
        const auto absolute = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), original);
        if (_mm256_movemask_ps(_mm256_cmp_ps(absolute, _mm256_set1_ps(INFINITY), _CMP_LT_OQ)) != 255) {
            for (unsigned lane = 0; lane < 8; ++lane) values[i + lane] = gelu_new_scalar(values[i + lane]);
            continue;
        }
        // Saturated tails are evaluated without overflowing the cubic.
        const auto x = _mm256_max_ps(_mm256_set1_ps(-10.0f), _mm256_min_ps(_mm256_set1_ps(10.0f), original));
        const auto cube_term = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.044715f), x), x), x);
        const auto argument = _mm256_mul_ps(_mm256_set1_ps(0.7978845608028654f), _mm256_add_ps(x, cube_term));
        const auto t = _mm256_max_ps(lower, _mm256_min_ps(bound, argument));
        const auto z = _mm256_mul_ps(t, t);
        // [13/12] Pade approximant, derived by matching tanh's Taylor series
        // through x^25. Horner evaluation; clamp at |argument| >= 8.
        auto p = _mm256_set1_ps(1.0f / 7905853580625.0f);
        p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(1.0f / 1930611375.0f));
        p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(2.0f / 6194475.0f));
        p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(4.0f / 60375.0f));
        p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(3.0f / 575.0f));
        p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(11.0f / 75.0f));
        p = _mm256_fmadd_ps(p, z, one);
        auto q = _mm256_set1_ps(1.0f / 86877511875.0f);
        q = _mm256_fmadd_ps(q, z, _mm256_set1_ps(8.0f / 526530375.0f));
        q = _mm256_fmadd_ps(q, z, _mm256_set1_ps(2.0f / 382375.0f));
        q = _mm256_fmadd_ps(q, z, _mm256_set1_ps(8.0f / 12075.0f));
        q = _mm256_fmadd_ps(q, z, _mm256_set1_ps(11.0f / 345.0f));
        q = _mm256_fmadd_ps(q, z, _mm256_set1_ps(12.0f / 25.0f));
        q = _mm256_fmadd_ps(q, z, one);
        auto tangent = _mm256_div_ps(_mm256_mul_ps(t, p), q);
        tangent = _mm256_max_ps(_mm256_set1_ps(-1.0f), _mm256_min_ps(one, tangent));
        tangent = _mm256_blendv_ps(tangent, one, _mm256_cmp_ps(argument, bound, _CMP_GE_OQ));
        tangent = _mm256_blendv_ps(tangent, _mm256_set1_ps(-1.0f), _mm256_cmp_ps(argument, lower, _CMP_LE_OQ));
        _mm256_storeu_ps(values + i, _mm256_mul_ps(_mm256_mul_ps(half, original), _mm256_add_ps(one, tangent)));
    }
    for (; i < size; ++i) values[i] = gelu_new_scalar(values[i]);
}
} // namespace gelu_detail
#endif

inline void gelu_new_inplace(float* values, std::size_t size, bool use_avx2 = true) {
    if (!values && size) throw std::invalid_argument("GELU requires valid storage");
#if defined(__x86_64__) && defined(__GNUC__)
    if (use_avx2 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        gelu_detail::vector(values, size);
        return;
    }
#else
    (void)use_avx2;
#endif
    for (std::size_t i = 0; i < size; ++i) values[i] = gelu_new_scalar(values[i]);
}
} // namespace leaf::kernels
