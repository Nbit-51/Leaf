#include "leaf/graph_parser.h"

#include <fstream>
#include <stdexcept>
#include <cstring>

namespace leaf {

namespace {

// Little-endian reads matching Python's struct.pack("<...", ...) exactly.
// x86/x86-64 is little-endian natively, so this is a straight memcpy --
// no byte-swapping needed on the machines this engine targets.

uint32_t read_u32(const uint8_t* data, size_t& offset) {
    uint32_t v;
    std::memcpy(&v, data + offset, 4);
    offset += 4;
    return v;
}

uint64_t read_u64(const uint8_t* data, size_t& offset) {
    uint64_t v;
    std::memcpy(&v, data + offset, 8);
    offset += 8;
    return v;
}

uint8_t read_u8(const uint8_t* data, size_t& offset) {
    return data[offset++];
}

std::string read_string(const uint8_t* data, size_t& offset) {
    uint32_t length = read_u32(data, offset);
    std::string s(reinterpret_cast<const char*>(data + offset), length);
    offset += length;
    return s;
}

std::vector<std::string> read_string_array(const uint8_t* data, size_t& offset) {
    uint32_t count = read_u32(data, offset);
    std::vector<std::string> items;
    items.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        items.push_back(read_string(data, offset));
    }
    return items;
}

}  // namespace

Graph Graph::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("leaf::Graph::load: cannot open file: " + path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    Graph g;
    g.file_data_.resize(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(g.file_data_.data()), size)) {
        throw std::runtime_error("leaf::Graph::load: failed to read file: " + path);
    }

    const uint8_t* data = g.file_data_.data();
    size_t offset = 0;

    // -- Header --
    if (size < 4 || std::memcmp(data, "LEAF", 4) != 0) {
        throw std::runtime_error("leaf::Graph::load: bad magic bytes (not a .leaf file): " + path);
    }
    offset += 4;

    g.version_ = read_u32(data, offset);
    if (g.version_ != 1) {
        throw std::runtime_error(
            "leaf::Graph::load: unsupported format version " + std::to_string(g.version_) +
            " (this parser supports version 1)");
    }

    uint32_t node_count = read_u32(data, offset);
    uint32_t initializer_count = read_u32(data, offset);

    g.inputs_ = read_string_array(data, offset);
    g.outputs_ = read_string_array(data, offset);

    // -- Node table --
    g.nodes_.reserve(node_count);
    for (uint32_t i = 0; i < node_count; ++i) {
        Node node;
        node.op_type = read_string(data, offset);
        node.inputs = read_string_array(data, offset);
        node.outputs = read_string_array(data, offset);
        node.attributes_json = read_string(data, offset);
        g.nodes_.push_back(std::move(node));
    }

    // -- Initializer table --
    g.init_meta_.reserve(initializer_count);
    for (uint32_t i = 0; i < initializer_count; ++i) {
        InitializerMeta meta;
        meta.name = read_string(data, offset);

        uint32_t ndims = read_u32(data, offset);
        meta.shape.reserve(ndims);
        for (uint32_t d = 0; d < ndims; ++d) {
            meta.shape.push_back(read_u32(data, offset));
        }

        meta.dtype_tag = read_u8(data, offset);
        meta.byte_offset = read_u64(data, offset);
        meta.byte_length = read_u64(data, offset);

        g.init_index_[meta.name] = g.init_meta_.size();
        g.init_meta_.push_back(std::move(meta));
    }

    // Everything from here to EOF is the raw data block. Initializer
    // byte_offset values are relative to this point, matching how
    // export_binary.py computed them.
    g.data_block_start_ = offset;

    if (g.file_data_.size() < g.data_block_start_) {
        throw std::runtime_error("leaf::Graph::load: file truncated before data block: " + path);
    }

    return g;
}

const InitializerMeta& Graph::initializer_meta(const std::string& name) const {
    auto it = init_index_.find(name);
    if (it == init_index_.end()) {
        throw std::runtime_error("leaf::Graph::initializer_meta: no such initializer: " + name);
    }
    return init_meta_[it->second];
}

const float* Graph::initializer_data(const std::string& name) const {
    const InitializerMeta& meta = initializer_meta(name);
    size_t absolute_offset = data_block_start_ + meta.byte_offset;

    if (absolute_offset + meta.byte_length > file_data_.size()) {
        throw std::runtime_error("leaf::Graph::initializer_data: data out of bounds for: " + name);
    }
    if (meta.dtype_tag != 0) {
        throw std::runtime_error("leaf::Graph::initializer_data: unsupported dtype for: " + name);
    }

    return reinterpret_cast<const float*>(file_data_.data() + absolute_offset);
}

}  // namespace leaf
