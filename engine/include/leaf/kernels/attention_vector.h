#pragma once
#include "leaf/kernels/transformer.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#endif

namespace leaf::kernels {
#if defined(__x86_64__) && defined(__GNUC__)
namespace attention_vector_detail {
#define LEAF_ATTENTION_TARGET __attribute__((target("avx2,no-fma")))
// Softmax arguments are nonpositive. Evaluate four ordinary arguments in
// double precision, then round once to float. Range reduction gives |r| <=
// ln(2)/2; the degree-13 Taylor remainder there is below 6e-18. This avoids
// scalar libm overhead without a float reciprocal or reassociated reduction.
// Rare underflow tails and exceptional lanes retain scalar libm semantics.
LEAF_ATTENTION_TARGET inline void exp_scores(float* scores, std::size_t size, float maximum) {
    std::size_t i = 0;
    for (; size - i >= 4; i += 4) {
        const auto shifted = _mm_sub_ps(_mm_loadu_ps(scores + i), _mm_set1_ps(maximum));
        const auto ordinary = _mm_and_ps(_mm_cmpge_ps(shifted, _mm_set1_ps(-80.0f)),
                                         _mm_cmple_ps(shifted, _mm_setzero_ps()));
        const int lanes = _mm_movemask_ps(ordinary);
        const auto x = _mm256_cvtps_pd(_mm_and_ps(shifted, ordinary));
        const auto n = _mm256_round_pd(_mm256_mul_pd(x, _mm256_set1_pd(1.4426950408889634074)),
                                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // Split ln(2) keeps cancellation error small throughout [-80, 0].
        auto r = _mm256_sub_pd(x, _mm256_mul_pd(n, _mm256_set1_pd(0.69314718036912381649)));
        r = _mm256_sub_pd(r, _mm256_mul_pd(n, _mm256_set1_pd(1.9082149292705877000e-10)));
        auto p = _mm256_set1_pd(1.0 / 6227020800.0);
        for (double coefficient : {1.0 / 479001600.0, 1.0 / 39916800.0, 1.0 / 3628800.0,
                1.0 / 362880.0, 1.0 / 40320.0, 1.0 / 5040.0, 1.0 / 720.0,
                1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0})
            p = _mm256_add_pd(_mm256_mul_pd(p, r), _mm256_set1_pd(coefficient));
        const auto exponent = _mm256_slli_epi64(_mm256_add_epi64(
            _mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(n)), _mm256_set1_epi64x(1023)), 52);
        _mm_storeu_ps(scores + i, _mm256_cvtpd_ps(_mm256_mul_pd(p, _mm256_castsi256_pd(exponent))));
        if (lanes != 15) {
            alignas(16) float values[4];
            _mm_store_ps(values, shifted);
            for (unsigned j = 0; j < 4; ++j)
                if (!(lanes & (1 << j))) scores[i + j] = values[j] == -INFINITY ? 0.0f : std::exp(values[j]);
        }
    }
    for (; i < size; ++i)
        scores[i] = scores[i] == -INFINITY ? 0.0f : std::exp(scores[i] - maximum);
}

// Sum eight lanes left to right, then the scalar tail: the established order.
LEAF_ATTENTION_TARGET inline float finish_dot(__m256 sum, const float* x, const float* w,
                                              std::size_t from, std::size_t dim) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float dot = 0;
    for (float lane : lanes) dot += lane;
    for (std::size_t d = from; d < dim; ++d) dot += x[d] * w[d];
    return dot;
}

// Four keys share each query load. Every key keeps its own accumulator, chunk
// order and horizontal sum, so each score is bit-identical to one-key code.
LEAF_ATTENTION_TARGET inline void dot4(const float* x, const float* w0, const float* w1,
        const float* w2, const float* w3, std::size_t dim, float* out) {
    auto s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    std::size_t d = 0;
    for (; dim - d >= 8; d += 8) {
        const auto q = _mm256_loadu_ps(x + d);
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(q, _mm256_loadu_ps(w0 + d)));
        s1 = _mm256_add_ps(s1, _mm256_mul_ps(q, _mm256_loadu_ps(w1 + d)));
        s2 = _mm256_add_ps(s2, _mm256_mul_ps(q, _mm256_loadu_ps(w2 + d)));
        s3 = _mm256_add_ps(s3, _mm256_mul_ps(q, _mm256_loadu_ps(w3 + d)));
    }
    out[0] = finish_dot(s0, x, w0, d, dim); out[1] = finish_dot(s1, x, w1, d, dim);
    out[2] = finish_dot(s2, x, w2, d, dim); out[3] = finish_dot(s3, x, w3, d, dim);
}

LEAF_ATTENTION_TARGET inline float dot1(const float* x, const float* w, std::size_t dim) {
    auto sum = _mm256_setzero_ps();
    std::size_t d = 0;
    for (; dim - d >= 8; d += 8)
        sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(x + d), _mm256_loadu_ps(w + d)));
    return finish_dot(sum, x, w, d, dim);
}

// y[0:8N] += p[k] * v[k, 0:8N] for every key in increasing order, with the
// output held in registers. Starting from +0 matches the zero-filled output.
template <unsigned N>
LEAF_ATTENTION_TARGET inline void weighted_values(const float* probabilities, const float* value,
        std::size_t keys, std::size_t dim, float* y) {
    __m256 acc[N];
    for (unsigned j = 0; j < N; ++j) acc[j] = _mm256_setzero_ps();
    for (std::size_t k = 0; k < keys; ++k) {
        const auto p = _mm256_set1_ps(probabilities[k]);
        const auto* v = value + k * dim;
        for (unsigned j = 0; j < N; ++j)
            acc[j] = _mm256_add_ps(acc[j], _mm256_mul_ps(p, _mm256_loadu_ps(v + 8 * j)));
    }
    for (unsigned j = 0; j < N; ++j) _mm256_storeu_ps(y + 8 * j, acc[j]);
}

LEAF_ATTENTION_TARGET inline void attend(
        const float* query, const float* key, const float* value, const float* mask, float* output,
        std::size_t batch, std::size_t heads, std::size_t kv_heads, std::size_t queries,
        std::size_t keys, std::size_t stride, std::size_t dim, const std::size_t* shape,
        float scale, bool valid_nonzero) {
    std::vector<float> scores(keys);
    std::vector<std::size_t> valid(keys);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < heads; ++h) {
            const auto kv = h / (heads / kv_heads);
            const auto* key_head = key + (b * kv_heads + kv) * stride * dim;
            const auto* value_head = value + (b * kv_heads + kv) * stride * dim;
            for (std::size_t q = 0; q < queries; ++q) {
                const auto* x = query + ((b * heads + h) * queries + q) * dim;
                auto* y = output + ((b * queries + q) * heads + h) * dim;
                const auto mb = shape[0] == 1 ? 0 : b, mh = shape[1] == 1 ? 0 : h;
                const auto mq = shape[2] == 1 ? 0 : q;
                const auto* mask_row = mask + ((mb * shape[1] + mh) * shape[2] + mq) * shape[3];
                std::size_t count = 0;
                for (std::size_t k = 0; k < keys; ++k) {
                    if ((mask_row[shape[3] == 1 ? 0 : k] != 0.0f) != valid_nonzero)
                        scores[k] = -std::numeric_limits<float>::infinity();
                    else
                        valid[count++] = k;
                }
                float dots[4];
                std::size_t i = 0;
                for (; count - i >= 4; i += 4) {
                    dot4(x, key_head + valid[i] * dim, key_head + valid[i + 1] * dim,
                         key_head + valid[i + 2] * dim, key_head + valid[i + 3] * dim, dim, dots);
                    for (unsigned j = 0; j < 4; ++j) scores[valid[i + j]] = dots[j] * scale * scale;
                }
                for (; i < count; ++i)
                    scores[valid[i]] = dot1(x, key_head + valid[i] * dim, dim) * scale * scale;
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t c = 0; c < count; ++c) maximum = std::max(maximum, scores[valid[c]]);
                if (!std::isfinite(maximum)) { std::fill_n(y, dim, 0.0f); continue; }
                exp_scores(scores.data(), keys, maximum);
                float denominator = 0;
                for (std::size_t k = 0; k < keys; ++k) denominator += scores[k];
                for (std::size_t k = 0; k < keys; ++k) scores[k] = scores[k] / denominator;
                // Preserve key order and separate multiply/add for each output
                // dimension. Do not skip masked values: 0 * NaN/Inf is observable.
                std::size_t d = 0;
                for (; dim - d >= 64; d += 64) weighted_values<8>(scores.data(), value_head + d, keys, dim, y + d);
                switch ((dim - d) / 8) {
                case 1: weighted_values<1>(scores.data(), value_head + d, keys, dim, y + d); break;
                case 2: weighted_values<2>(scores.data(), value_head + d, keys, dim, y + d); break;
                case 3: weighted_values<3>(scores.data(), value_head + d, keys, dim, y + d); break;
                case 4: weighted_values<4>(scores.data(), value_head + d, keys, dim, y + d); break;
                case 5: weighted_values<5>(scores.data(), value_head + d, keys, dim, y + d); break;
                case 6: weighted_values<6>(scores.data(), value_head + d, keys, dim, y + d); break;
                case 7: weighted_values<7>(scores.data(), value_head + d, keys, dim, y + d); break;
                }
                d += (dim - d) / 8 * 8;
                for (; d < dim; ++d) {
                    float sum = 0.0f;
                    for (std::size_t k = 0; k < keys; ++k) sum += scores[k] * value_head[k * dim + d];
                    y[d] = sum;
                }
            }
        }
    }
}
#undef LEAF_ATTENTION_TARGET
} // namespace attention_vector_detail
#endif

// Same shape/storage contract as attention_f32_gqa_strided; caller validates.
// Vector Q.K uses lane-wise reduction; softmax uses the bounded vector exp
// above. Scalar and unsupported CPUs retain the established path. No global
// ISA build flags or model-name checks.
inline void attention_f32_gqa_strided_vector(
        const float* query, const float* key, const float* value, const float* mask, float* output,
        std::size_t batch, std::size_t heads, std::size_t kv_heads, std::size_t queries,
        std::size_t keys, std::size_t stride, std::size_t dim, const std::size_t* shape,
        float scale, bool valid_nonzero = true, bool use_avx2 = true) {
#if defined(__x86_64__) && defined(__GNUC__)
    if (use_avx2 && __builtin_cpu_supports("avx2")) {
        attention_vector_detail::attend(query, key, value, mask, output, batch, heads,
            kv_heads, queries, keys, stride, dim, shape, scale, valid_nonzero);
        return;
    }
#else
    (void)use_avx2;
#endif
    attention_f32_gqa_strided(query, key, value, mask, output, batch, heads, kv_heads,
        queries, keys, stride, dim, shape, scale, valid_nonzero);
}
} // namespace leaf::kernels
