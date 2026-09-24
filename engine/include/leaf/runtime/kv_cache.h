#pragma once

#include <cstddef>
#include <vector>

namespace leaf::runtime {

// Per-session FP32 key/value storage. Appended tensors are contiguous
// [batch, kv_heads, new_tokens, head_dim]; storage grows geometrically and
// preserves prior tokens. The cache is owned by one decoder session and has
// no process-global state.
class KVCache {
public:
    KVCache(std::size_t batch, std::size_t kv_heads, std::size_t head_dim);

    void append(const float* keys, const float* values, std::size_t new_tokens);
    void reset() noexcept { length_ = 0; }

    std::size_t length() const noexcept { return length_; }
    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t batch() const noexcept { return batch_; }
    std::size_t kv_heads() const noexcept { return kv_heads_; }
    std::size_t head_dim() const noexcept { return head_dim_; }
    const float* keys_data() const noexcept { return keys_.data(); }
    const float* values_data() const noexcept { return values_.data(); }

    // Query is [batch, query_heads, query_tokens, head_dim]. Output is
    // [batch, query_tokens, query_heads, head_dim]. Query heads may be a
    // multiple of KV heads; GQA repetition is virtual, not materialized.
    void attend(const float* query, const float* mask, float* output,
                std::size_t query_heads, std::size_t query_tokens,
                const std::size_t* mask_shape, float scale,
                bool mask_nonzero_is_valid = true) const;

private:
    std::size_t batch_;
    std::size_t kv_heads_;
    std::size_t head_dim_;
    std::size_t length_ = 0;
    std::size_t capacity_ = 0;
    std::vector<float> keys_;
    std::vector<float> values_;
};

}  // namespace leaf::runtime
