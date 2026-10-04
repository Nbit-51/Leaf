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
__attribute__((target("avx2,no-fma"))) inline void attend(
        const float* query, const float* key, const float* value, const float* mask, float* output,
        std::size_t batch, std::size_t heads, std::size_t kv_heads, std::size_t queries,
        std::size_t keys, std::size_t stride, std::size_t dim, const std::size_t* shape,
        float scale, bool valid_nonzero) {
    std::vector<float> scores(keys);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < heads; ++h) {
            const auto kv = h / (heads / kv_heads);
            const auto* key_head = key + (b * kv_heads + kv) * stride * dim;
            const auto* value_head = value + (b * kv_heads + kv) * stride * dim;
            for (std::size_t q = 0; q < queries; ++q) {
                const auto* x = query + ((b * heads + h) * queries + q) * dim;
                auto* y = output + ((b * queries + q) * heads + h) * dim;
                std::fill_n(y, dim, 0.0f);
                const auto mb = shape[0] == 1 ? 0 : b, mh = shape[1] == 1 ? 0 : h;
                const auto mq = shape[2] == 1 ? 0 : q;
                const auto* mask_row = mask + ((mb * shape[1] + mh) * shape[2] + mq) * shape[3];
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t k = 0; k < keys; ++k) {
                    if ((mask_row[shape[3] == 1 ? 0 : k] != 0.0f) != valid_nonzero) {
                        scores[k] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    const auto* w = key_head + k * dim;
                    auto sum = _mm256_setzero_ps();
                    std::size_t d = 0;
                    for (; dim - d >= 8; d += 8)
                        sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(x + d), _mm256_loadu_ps(w + d)));
                    alignas(32) float lanes[8];
                    _mm256_store_ps(lanes, sum);
                    float dot = 0;
                    for (float lane : lanes) dot += lane;
                    for (; d < dim; ++d) dot += x[d] * w[d];
                    scores[k] = dot * scale * scale;
                    maximum = std::max(maximum, scores[k]);
                }
                if (!std::isfinite(maximum)) continue;
                float denominator = 0;
                for (std::size_t k = 0; k < keys; ++k) {
                    scores[k] = scores[k] == -std::numeric_limits<float>::infinity()
                        ? 0.0f : std::exp(scores[k] - maximum);
                    denominator += scores[k];
                }
                // Preserve key order and separate multiply/add for each output
                // dimension. Do not skip masked values: 0 * NaN/Inf is observable.
                for (std::size_t k = 0; k < keys; ++k) {
                    const float probability = scores[k] / denominator;
                    const auto p = _mm256_set1_ps(probability);
                    const auto* v = value_head + k * dim;
                    std::size_t d = 0;
                    for (; dim - d >= 8; d += 8)
                        _mm256_storeu_ps(y + d, _mm256_add_ps(_mm256_loadu_ps(y + d),
                            _mm256_mul_ps(p, _mm256_loadu_ps(v + d))));
                    for (; d < dim; ++d) y[d] += probability * v[d];
                }
            }
        }
    }
}
} // namespace attention_vector_detail
#endif

// Same shape/storage contract as attention_f32_gqa_strided; caller validates.
// Only the Q.K reduction order changes. Scalar and unsupported CPUs retain
// the established path. No global ISA build flags or model-name checks.
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
