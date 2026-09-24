#include "leaf/runtime/kv_cache.h"

#include "leaf/kernels/transformer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace leaf::runtime {

KVCache::KVCache(std::size_t batch, std::size_t kv_heads, std::size_t head_dim)
    : batch_(batch), kv_heads_(kv_heads), head_dim_(head_dim) {
    if (batch == 0 || kv_heads == 0 || head_dim == 0 ||
        batch > std::numeric_limits<std::size_t>::max() / kv_heads ||
        batch * kv_heads > std::numeric_limits<std::size_t>::max() / head_dim) {
        throw std::invalid_argument("KVCache dimensions are zero or overflow");
    }
}

void KVCache::append(const float* keys, const float* values, std::size_t new_tokens) {
    if (keys == nullptr || values == nullptr || new_tokens == 0 ||
        new_tokens > std::numeric_limits<std::size_t>::max() - length_) {
        throw std::invalid_argument("KVCache append requires nonempty tensors");
    }
    const std::size_t required = length_ + new_tokens;
    const std::size_t planes = batch_ * kv_heads_;
    if (required > std::numeric_limits<std::size_t>::max() / (planes * head_dim_)) {
        throw std::overflow_error("KVCache allocation size overflow");
    }
    if (required > capacity_) {
        std::size_t next = std::max<std::size_t>(8, capacity_);
        while (next < required) {
            next = next > std::numeric_limits<std::size_t>::max() / 2
                ? required : next * 2;
        }
        const std::size_t maximum_capacity =
            std::numeric_limits<std::size_t>::max() / (planes * head_dim_);
        next = std::min(next, maximum_capacity);
        std::vector<float> grown_keys(planes * next * head_dim_);
        std::vector<float> grown_values(grown_keys.size());
        if (length_ > 0) {
            for (std::size_t plane = 0; plane < planes; ++plane) {
                const std::size_t old_offset = plane * capacity_ * head_dim_;
                const std::size_t new_offset = plane * next * head_dim_;
                std::copy_n(keys_.data() + old_offset, length_ * head_dim_,
                            grown_keys.data() + new_offset);
                std::copy_n(values_.data() + old_offset, length_ * head_dim_,
                            grown_values.data() + new_offset);
            }
        }
        keys_.swap(grown_keys);
        values_.swap(grown_values);
        capacity_ = next;
    }
    for (std::size_t plane = 0; plane < planes; ++plane) {
        const std::size_t source = plane * new_tokens * head_dim_;
        const std::size_t destination = (plane * capacity_ + length_) * head_dim_;
        std::copy_n(keys + source, new_tokens * head_dim_, keys_.data() + destination);
        std::copy_n(values + source, new_tokens * head_dim_, values_.data() + destination);
    }
    length_ = required;
}

void KVCache::attend(const float* query, const float* mask, float* output,
                     std::size_t query_heads, std::size_t query_tokens,
                     const std::size_t* mask_shape, float scale,
                     bool mask_nonzero_is_valid) const {
    if (length_ == 0 || query == nullptr || mask == nullptr || output == nullptr ||
        mask_shape == nullptr || query_heads == 0 || query_tokens == 0 ||
        query_heads % kv_heads_ != 0 || !std::isfinite(scale) || scale <= 0.0f) {
        throw std::invalid_argument("KVCache attention arguments are invalid");
    }
    const std::size_t target_shape[4] = {batch_, query_heads, query_tokens, length_};
    for (std::size_t axis = 0; axis < 4; ++axis) {
        if (mask_shape[axis] != 1 && mask_shape[axis] != target_shape[axis]) {
            throw std::invalid_argument("KVCache attention mask is not broadcastable");
        }
    }
    kernels::attention_f32_gqa_strided(query, keys_.data(), values_.data(), mask,
                                        output, batch_, query_heads, kv_heads_,
                                        query_tokens, length_, capacity_, head_dim_,
                                        mask_shape, scale, mask_nonzero_is_valid);
}

}  // namespace leaf::runtime
