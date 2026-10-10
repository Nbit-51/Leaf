#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace leaf::runtime {
// Stateless, bidirectional FP32 encoder. Each call is an independent document;
// causal KV reuse is intentionally absent. Weights are memory mapped once.
class EmbeddingEncoder {
  public:
    EmbeddingEncoder(const std::string &artifact, unsigned threads = 1, bool scalar = false);
    ~EmbeddingEncoder();
    std::vector<float> encode(const std::vector<std::uint32_t> &tokens);
    std::size_t dimensions() const;
    std::size_t maximum_tokens() const;
    bool uses_avx2() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace leaf::runtime
