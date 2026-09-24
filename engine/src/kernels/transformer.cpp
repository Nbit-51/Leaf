#include "leaf/kernels/transformer.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace leaf::kernels {

void rmsnorm_f32_reference(const float* input, const float* weight, float* output,
                           std::size_t rows, std::size_t hidden, float epsilon) {
    for (std::size_t row = 0; row < rows; ++row) {
        const float* source = input + row * hidden;
        float* destination = output + row * hidden;
        float sum_squares = 0.0f;
        for (std::size_t column = 0; column < hidden; ++column) {
            sum_squares += source[column] * source[column];
        }
        const float factor = 1.0f / std::sqrt(sum_squares / static_cast<float>(hidden) + epsilon);
        for (std::size_t column = 0; column < hidden; ++column) {
            destination[column] = source[column] * factor * weight[column];
        }
    }
}

void rmsnorm_f32(const float* input, const float* weight, float* output,
                 std::size_t rows, std::size_t hidden, float epsilon) {
#if defined(__AVX2__)
    for (std::size_t row = 0; row < rows; ++row) {
        const float* source = input + row * hidden;
        float* destination = output + row * hidden;
        __m256 squares = _mm256_setzero_ps();
        std::size_t column = 0;
        for (; column + 8 <= hidden; column += 8) {
            const __m256 value = _mm256_loadu_ps(source + column);
            squares = _mm256_add_ps(squares, _mm256_mul_ps(value, value));
        }
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, squares);
        float sum_squares = 0.0f;
        for (const float value : lanes) sum_squares += value;
        for (; column < hidden; ++column) {
            sum_squares += source[column] * source[column];
        }
        const float factor = 1.0f / std::sqrt(sum_squares / static_cast<float>(hidden) + epsilon);
        const __m256 factor_vector = _mm256_set1_ps(factor);
        column = 0;
        for (; column + 8 <= hidden; column += 8) {
            const __m256 value = _mm256_loadu_ps(source + column);
            const __m256 scale = _mm256_loadu_ps(weight + column);
            _mm256_storeu_ps(destination + column,
                             _mm256_mul_ps(_mm256_mul_ps(value, factor_vector), scale));
        }
        for (; column < hidden; ++column) {
            destination[column] = source[column] * factor * weight[column];
        }
    }
#else
    rmsnorm_f32_reference(input, weight, output, rows, hidden, epsilon);
#endif
}

namespace {

float scalar_dot(const float* left, const float* right, std::size_t count) {
    float sum = 0.0f;
    for (std::size_t index = 0; index < count; ++index) sum += left[index] * right[index];
    return sum;
}

float fast_dot(const float* left, const float* right, std::size_t count) {
#if defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    std::size_t index = 0;
    for (; index + 8 <= count; index += 8) {
        sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(left + index),
                                                _mm256_loadu_ps(right + index)));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result = 0.0f;
    for (float lane : lanes) result += lane;
    for (; index < count; ++index) result += left[index] * right[index];
    return result;
#else
    return scalar_dot(left, right, count);
#endif
}

void attention_impl(const float* query, const float* key, const float* value,
                    const float* mask, float* output, std::size_t batch,
                    std::size_t heads, std::size_t kv_heads,
                    std::size_t query_tokens, std::size_t key_tokens,
                    std::size_t key_stride_tokens, std::size_t head_dim,
                    const std::size_t* mask_shape, float scale, bool mask_nonzero_is_valid,
                    float (*dot)(const float*, const float*, std::size_t)) {
    std::vector<float> scores(key_tokens);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < heads; ++h) {
            const std::size_t kv_head = h / (heads / kv_heads);
            for (std::size_t q = 0; q < query_tokens; ++q) {
                const float* query_row = query + ((b * heads + h) * query_tokens + q) * head_dim;
                float* output_row = output + ((b * query_tokens + q) * heads + h) * head_dim;
                std::fill(output_row, output_row + head_dim, 0.0f);
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t k = 0; k < key_tokens; ++k) {
                    const std::size_t mb = mask_shape[0] == 1 ? 0 : b;
                    const std::size_t mh = mask_shape[1] == 1 ? 0 : h;
                    const std::size_t mq = mask_shape[2] == 1 ? 0 : q;
                    const std::size_t mk = mask_shape[3] == 1 ? 0 : k;
                    const std::size_t mask_index = ((mb * mask_shape[1] + mh) *
                                                    mask_shape[2] + mq) * mask_shape[3] + mk;
                    if ((mask[mask_index] != 0.0f) != mask_nonzero_is_valid) {
                        scores[k] = -std::numeric_limits<float>::infinity();
                    } else {
                        const float* key_row = key +
                            ((b * kv_heads + kv_head) * key_stride_tokens + k) * head_dim;
                        scores[k] = dot(query_row, key_row, head_dim) * scale * scale;
                        maximum = std::max(maximum, scores[k]);
                    }
                }
                if (!std::isfinite(maximum)) continue;  // PyTorch's NaN -> 0 branch.
                float denominator = 0.0f;
                for (std::size_t k = 0; k < key_tokens; ++k) {
                    scores[k] = std::exp(scores[k] - maximum);
                    denominator += scores[k];
                }
                for (std::size_t k = 0; k < key_tokens; ++k) {
                    const float probability = scores[k] / denominator;
                    const float* value_row = value +
                        ((b * kv_heads + kv_head) * key_stride_tokens + k) * head_dim;
                    for (std::size_t d = 0; d < head_dim; ++d) {
                        output_row[d] += probability * value_row[d];
                    }
                }
            }
        }
    }
}

}  // namespace

void attention_f32_reference(const float* query, const float* key, const float* value,
                             const float* mask, float* output, std::size_t batch,
                             std::size_t heads, std::size_t query_tokens,
                             std::size_t key_tokens, std::size_t head_dim,
                             const std::size_t* mask_shape, float scale,
                             bool mask_nonzero_is_valid) {
    attention_impl(query, key, value, mask, output, batch, heads, heads,
                   query_tokens, key_tokens, key_tokens, head_dim, mask_shape,
                   scale, mask_nonzero_is_valid, scalar_dot);
}

void attention_f32(const float* query, const float* key, const float* value,
                   const float* mask, float* output, std::size_t batch,
                   std::size_t heads, std::size_t query_tokens,
                   std::size_t key_tokens, std::size_t head_dim,
                   const std::size_t* mask_shape, float scale,
                   bool mask_nonzero_is_valid) {
    attention_impl(query, key, value, mask, output, batch, heads, heads,
                   query_tokens, key_tokens, key_tokens, head_dim, mask_shape,
                   scale, mask_nonzero_is_valid, fast_dot);
}

void attention_f32_gqa_strided(const float* query, const float* key,
                               const float* value, const float* mask, float* output,
                               std::size_t batch, std::size_t query_heads,
                               std::size_t kv_heads, std::size_t query_tokens,
                               std::size_t key_tokens, std::size_t key_stride_tokens,
                               std::size_t head_dim, const std::size_t* mask_shape,
                               float scale, bool mask_nonzero_is_valid) {
    attention_impl(query, key, value, mask, output, batch, query_heads, kv_heads,
                   query_tokens, key_tokens, key_stride_tokens, head_dim, mask_shape,
                   scale, mask_nonzero_is_valid, fast_dot);
}

void rope_table_f32(const float* frequencies, const float* positions,
                    float* cosine, float* sine, std::size_t batch,
                    std::size_t half_dim, std::size_t tokens,
                    float cosine_scale, float sine_scale) {
    const std::size_t width = half_dim * 2;
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t token = 0; token < tokens; ++token) {
            for (std::size_t dimension = 0; dimension < half_dim; ++dimension) {
                const float angle = frequencies[b * half_dim + dimension] *
                                    positions[b * tokens + token];
                const float cos_value = std::cos(angle) * cosine_scale;
                const float sin_value = std::sin(angle) * sine_scale;
                const std::size_t index = (b * tokens + token) * width + dimension;
                cosine[index] = cosine[index + half_dim] = cos_value;
                sine[index] = sine[index + half_dim] = sin_value;
            }
        }
    }
}

void repeat_kv_f32(const float* input, float* output, std::size_t batch,
                   std::size_t kv_heads, std::size_t tokens,
                   std::size_t head_dim, std::size_t repeats) {
    const std::size_t head_elements = tokens * head_dim;
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t head = 0; head < kv_heads; ++head) {
            const float* source = input + (b * kv_heads + head) * head_elements;
            for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
                float* destination = output +
                    (b * kv_heads * repeats + head * repeats + repeat) * head_elements;
                std::copy(source, source + head_elements, destination);
            }
        }
    }
}

}  // namespace leaf::kernels
