#pragma once

#include <cstddef>

namespace leaf::kernels {

// Applies RMSNorm independently to each row of a contiguous [..., hidden]
// tensor. The caller validates shapes and epsilon.
void rmsnorm_f32_reference(const float* input, const float* weight, float* output,
                           std::size_t rows, std::size_t hidden, float epsilon);
void rmsnorm_f32(const float* input, const float* weight, float* output,
                 std::size_t rows, std::size_t hidden, float epsilon);

// Q/K/V are contiguous [batch, heads, tokens, head_dim]. The mask is
// contiguous [mask_batch, mask_heads, mask_queries, mask_keys], with each
// dimension either 1 or its corresponding attention dimension. Mask polarity
// is explicit. Output is [batch, query_tokens, heads, head_dim].
void attention_f32_reference(const float* query, const float* key, const float* value,
                             const float* mask, float* output, std::size_t batch,
                             std::size_t heads, std::size_t query_tokens,
                             std::size_t key_tokens, std::size_t head_dim,
                             const std::size_t* mask_shape, float scale,
                             bool mask_nonzero_is_valid = true);
void attention_f32(const float* query, const float* key, const float* value,
                   const float* mask, float* output, std::size_t batch,
                   std::size_t heads, std::size_t query_tokens,
                   std::size_t key_tokens, std::size_t head_dim,
                   const std::size_t* mask_shape, float scale,
                   bool mask_nonzero_is_valid = true);

// Cached grouped-query attention. K/V use a token stride larger than or equal
// to key_tokens (the cache capacity); query heads map consecutively onto KV
// heads, avoiding a materialized RepeatKV tensor during decode.
void attention_f32_gqa_strided(const float* query, const float* key,
                               const float* value, const float* mask, float* output,
                               std::size_t batch, std::size_t query_heads,
                               std::size_t kv_heads, std::size_t query_tokens,
                               std::size_t key_tokens, std::size_t key_stride_tokens,
                               std::size_t head_dim, const std::size_t* mask_shape,
                               float scale, bool mask_nonzero_is_valid = true);

// Inputs are [batch, half_dim, 1] and [batch, 1, tokens]. The duplicated
// frequency table outputs are [batch, tokens, 2 * half_dim].
void rope_table_f32(const float* frequencies, const float* positions,
                    float* cosine, float* sine, std::size_t batch,
                    std::size_t half_dim, std::size_t tokens,
                    float cosine_scale, float sine_scale);

// Repeat each KV head consecutively: [batch, kv_heads, tokens, head_dim]
// becomes [batch, kv_heads * repeats, tokens, head_dim].
void repeat_kv_f32(const float* input, float* output, std::size_t batch,
                   std::size_t kv_heads, std::size_t tokens,
                   std::size_t head_dim, std::size_t repeats);

}  // namespace leaf::kernels
