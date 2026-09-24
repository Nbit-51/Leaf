#pragma once

#include "leaf/graph_parser.h"
#include "leaf/runtime/kv_cache.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace leaf {

// A runtime tensor owns only activation data.  Model initializers remain in
// Graph's file buffer and are referenced without an extra model-sized copy.
struct Tensor {
    std::vector<size_t> shape;
    std::vector<float> values;
    float* arena_data = nullptr;

    size_t element_count() const;
    float* data() { return arena_data != nullptr ? arena_data : values.data(); }
    const float* data() const { return arena_data != nullptr ? arena_data : values.data(); }
};

// Executes the FP32 subset emitted by Leaf's current ResNet export path.
// The runtime is deliberately framework-free: an exported .leaf artifact,
// this executor, and AVX2-capable C++ are all the target machine needs.
class Executor {
public:
    // Convenience API for existing one-input/one-output artifacts.
    Tensor run(const Graph& graph, const Tensor& input) const;

    // General graph boundary: named concrete inputs and owned output copies.
    // Stateful decoding is handled by an explicit KVCache owned by the caller.
    std::unordered_map<std::string, Tensor> run_outputs(
        const Graph& graph, const std::unordered_map<std::string, Tensor>& inputs) const;

    // Attention nodes with a nonempty cache_id append their K/V inputs to the
    // caller-owned cache map. Reuse the map across decode calls; clear it at a
    // sequence boundary. Ordinary run()/run_outputs() remain stateless.
    std::unordered_map<std::string, Tensor> run_outputs_cached(
        const Graph& graph, const std::unordered_map<std::string, Tensor>& inputs,
        std::unordered_map<std::string, runtime::KVCache>& caches) const;
};

}  // namespace leaf
