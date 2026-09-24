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

}  // namespace leaf::kernels
