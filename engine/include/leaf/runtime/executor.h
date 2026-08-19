#pragma once

#include "leaf/graph_parser.h"

#include <cstddef>
#include <vector>

namespace leaf {

// A runtime tensor owns only activation data.  Model initializers remain in
// Graph's file buffer and are referenced without an extra model-sized copy.
struct Tensor {
    std::vector<size_t> shape;
    std::vector<float> values;

    size_t element_count() const;
};

// Executes the FP32 subset emitted by Leaf's current ResNet export path.
// The runtime is deliberately framework-free: an exported .leaf artifact,
// this executor, and AVX2-capable C++ are all the target machine needs.
class Executor {
public:
    // The current artifact format records graph-input names but not their
    // shapes, so the caller supplies a single concrete input tensor.
    // Models with multiple dynamic inputs are rejected explicitly.
    Tensor run(const Graph& graph, const Tensor& input) const;
};

}  // namespace leaf
