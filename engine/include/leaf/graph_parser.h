#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace leaf {

// Mirrors tools/graph_opt/export_binary.py's format exactly.
// See that file's module docstring for the authoritative byte layout.

struct Node {
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::string attributes_json;  // parsed lazily by whichever kernel needs it
};

struct InitializerMeta {
    std::string name;
    std::vector<uint32_t> shape;
    uint8_t dtype_tag;   // 0 = float32 (only supported type for now)
    uint64_t byte_offset;
    uint64_t byte_length;
};

class Graph {
public:
    // Loads and fully parses a .leaf file. Throws std::runtime_error on
    // any format mismatch (bad magic, truncated file, unsupported
    // version) -- fail loudly, never silently load a corrupt graph.
    static Graph load(const std::string& path);

    // Returns a raw pointer to this initializer's float32 data inside
    // the (still-owned) data block. No copy -- the Graph must outlive
    // any pointer obtained this way.
    const float* initializer_data(const std::string& name) const;
    const InitializerMeta& initializer_meta(const std::string& name) const;

    uint32_t version() const { return version_; }
    const std::vector<std::string>& inputs() const { return inputs_; }
    const std::vector<std::string>& outputs() const { return outputs_; }
    const std::vector<Node>& nodes() const { return nodes_; }
    size_t initializer_count() const { return init_meta_.size(); }

private:
    uint32_t version_ = 0;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::vector<Node> nodes_;
    std::vector<InitializerMeta> init_meta_;
    std::unordered_map<std::string, size_t> init_index_;  // name -> index into init_meta_

    // Owns the raw file bytes for the data block's lifetime. Using the
    // whole-file buffer (rather than copying just the data block out)
    // keeps loading simple for now; a later optimization can mmap this
    // instead of reading the whole file into memory.
    std::vector<uint8_t> file_data_;
    size_t data_block_start_ = 0;
};

}  // namespace leaf
