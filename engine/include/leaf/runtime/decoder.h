#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace leaf::runtime {

// Executes a serialized decoder plan: normalization, positional encoding,
// attention geometry, residual topology and feed-forward semantics are data.
// Mapped immutable weights can be shared; each instance owns its KV state.
class Decoder {
public:
    Decoder(const std::string& artifact, unsigned threads = 1, bool force_scalar = false,
            unsigned activation_bits = 32);
    ~Decoder();
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    void reset();
    // Append tokens to this session; logits are [tokens, vocab] when all_logits
    // is true, or only [vocab] for generation/latency measurements.
    std::vector<float> forward(const std::vector<std::uint32_t>& tokens, bool all_logits = false);
    std::uint32_t vocabulary_size() const;
    std::uint32_t eos_token() const;
    std::size_t cache_tokens() const;
    std::uint64_t weight_bytes() const;
    bool uses_avx2() const;
    bool uses_vnni() const;
    unsigned activation_bits() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace leaf::runtime
