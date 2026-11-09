#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace leaf::runtime {

enum class FP32Math { Fast, Accurate };

// Executes a serialized decoder plan: normalization, positional encoding,
// attention geometry, residual topology and feed-forward semantics are data.
// Mapped immutable weights can be shared; each instance owns its KV state.
class Decoder {
public:
    Decoder(const std::string& artifact, unsigned threads = 1, bool force_scalar = false,
            unsigned activation_bits = 32, const std::vector<unsigned>& worker_cpus = {},
            FP32Math fp32_math = FP32Math::Fast);
    ~Decoder();
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    void reset();
    // Append tokens to this session; logits are [tokens, vocab] when all_logits
    // is true, or only [vocab] for generation/latency measurements.
    std::vector<float> forward(const std::vector<std::uint32_t>& tokens, bool all_logits = false);
    // Append tokens and write the same logits into caller-owned storage. The
    // buffer is resized/overwritten on every call; keeping its capacity avoids
    // repeated output allocation. Its contents after an error are unspecified.
    // Like forward(), this mutates the session and is not reentrant.
    void forward_into(const std::vector<std::uint32_t>& tokens, std::vector<float>& logits,
                      bool all_logits = false);
    std::uint32_t vocabulary_size() const;
    std::uint32_t eos_token() const;
    bool is_eos_token(std::uint32_t token) const;
    std::size_t cache_tokens() const;
    std::uint64_t weight_bytes() const;
    bool uses_avx2() const;
    bool uses_optimized_fp32() const;
    bool uses_accurate_fp32() const;
    bool uses_vector_silu_gate() const;
    std::size_t silu_gate_vector_calls() const;
    std::size_t source_bf16_tensors() const;
    std::size_t source_bf16_linear_calls() const;
    bool uses_vnni() const;
    unsigned activation_bits() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace leaf::runtime
