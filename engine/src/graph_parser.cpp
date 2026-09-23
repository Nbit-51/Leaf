#include "leaf/graph_parser.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_set>

namespace leaf {
namespace {

struct Reader {
    const std::vector<uint8_t>& bytes;
    size_t offset = 0;

    void require(size_t count) const {
        if (offset > bytes.size() || count > bytes.size() - offset) {
            throw std::runtime_error("leaf::Graph::load: truncated .leaf file");
        }
    }

    uint8_t u8() {
        require(1);
        return bytes[offset++];
    }

    uint32_t u32() {
        require(4);
        uint32_t value = 0;
        for (unsigned i = 0; i < 4; ++i) {
            value |= static_cast<uint32_t>(bytes[offset++]) << (8 * i);
        }
        return value;
    }

    uint64_t u64() {
        require(8);
        uint64_t value = 0;
        for (unsigned i = 0; i < 8; ++i) {
            value |= static_cast<uint64_t>(bytes[offset++]) << (8 * i);
        }
        return value;
    }

    float f32() {
        const uint32_t bits = u32();
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    std::string string() {
        const uint32_t length = u32();
        require(length);
        std::string value(reinterpret_cast<const char*>(bytes.data() + offset), length);
        offset += length;
        return value;
    }

    std::vector<std::string> strings() {
        const uint32_t count = u32();
        if (count > (bytes.size() - offset) / 4) {
            throw std::runtime_error("leaf::Graph::load: invalid string count");
        }
        std::vector<std::string> values;
        values.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            values.push_back(string());
        }
        return values;
    }
};

size_t aligned_32(size_t value) {
    if (value > std::numeric_limits<size_t>::max() - 31) {
        throw std::runtime_error("leaf::Graph::load: invalid data-block offset");
    }
    return (value + 31) & ~size_t(31);
}

size_t element_count(const std::vector<uint32_t>& shape) {
    size_t count = 1;
    for (uint32_t dimension : shape) {
        if (dimension != 0 && count > std::numeric_limits<size_t>::max() / dimension) {
            throw std::runtime_error("leaf::Graph::load: initializer shape overflow");
        }
        count *= dimension;
    }
    return count;
}

}  // namespace

Graph Graph::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("leaf::Graph::load: cannot open file: " + path);
    }
    const std::streamsize size = file.tellg();
    if (size < 0) {
        throw std::runtime_error("leaf::Graph::load: cannot determine file size: " + path);
    }
    file.seekg(0, std::ios::beg);

    Graph graph;
    graph.file_data_.resize(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(graph.file_data_.data()), size)) {
        throw std::runtime_error("leaf::Graph::load: failed to read file: " + path);
    }
    Reader reader{graph.file_data_};
    reader.require(4);
    if (std::memcmp(graph.file_data_.data(), "LEAF", 4) != 0) {
        throw std::runtime_error("leaf::Graph::load: bad magic bytes: " + path);
    }
    reader.offset = 4;
    graph.version_ = reader.u32();
    if (graph.version_ != 1 && graph.version_ != 2 && graph.version_ != 3) {
        throw std::runtime_error("leaf::Graph::load: unsupported format version " +
                                 std::to_string(graph.version_));
    }
    const uint32_t node_count = reader.u32();
    const uint32_t initializer_count = reader.u32();
    graph.inputs_ = reader.strings();
    graph.outputs_ = reader.strings();
    const size_t remaining = graph.file_data_.size() - reader.offset;
    const size_t minimum_node_bytes = graph.version_ >= 2 ? 17 : 16;
    if (node_count > remaining / minimum_node_bytes ||
        initializer_count > remaining / 25) {
        throw std::runtime_error("leaf::Graph::load: impossible table count");
    }

    graph.nodes_.reserve(node_count);
    for (uint32_t i = 0; i < node_count; ++i) {
        Node node;
        node.op_type = reader.string();
        node.inputs = reader.strings();
        node.outputs = reader.strings();
        node.attributes_json = reader.string();
        if (graph.version_ >= 2) {
            const uint8_t flag = reader.u8();
            if (flag > 1) {
                throw std::runtime_error("leaf::Graph::load: invalid quantization flag");
            }
            node.quantized = flag == 1;
            if (node.quantized) {
                node.input_scale = reader.f32();
                node.weight_axis = reader.u8();
                const uint32_t scale_count = reader.u32();
                if (scale_count == 0 || scale_count >
                        (graph.file_data_.size() - reader.offset) / sizeof(float)) {
                    throw std::runtime_error("leaf::Graph::load: invalid channel-scale count");
                }
                node.weight_scales.reserve(scale_count);
                for (uint32_t channel = 0; channel < scale_count; ++channel) {
                    const float scale = reader.f32();
                    if (!std::isfinite(scale) || scale <= 0.0f) {
                        throw std::runtime_error("leaf::Graph::load: invalid channel scale");
                    }
                    node.weight_scales.push_back(scale);
                }
                if (!std::isfinite(node.input_scale) || node.input_scale <= 0.0f) {
                    throw std::runtime_error("leaf::Graph::load: invalid input scale");
                }
            }
        }
        graph.nodes_.push_back(std::move(node));
    }

    graph.init_meta_.reserve(initializer_count);
    for (uint32_t i = 0; i < initializer_count; ++i) {
        InitializerMeta meta;
        meta.name = reader.string();
        const uint32_t rank = reader.u32();
        if (rank > (graph.file_data_.size() - reader.offset) / sizeof(uint32_t)) {
            throw std::runtime_error("leaf::Graph::load: invalid initializer rank");
        }
        meta.shape.reserve(rank);
        for (uint32_t axis = 0; axis < rank; ++axis) {
            meta.shape.push_back(reader.u32());
        }
        meta.dtype_tag = reader.u8();
        meta.byte_offset = reader.u64();
        meta.byte_length = reader.u64();
        if (meta.dtype_tag > 1 || (graph.version_ == 1 && meta.dtype_tag != 0)) {
            throw std::runtime_error("leaf::Graph::load: unsupported initializer dtype");
        }
        const size_t count = element_count(meta.shape);
        const size_t item_size = meta.dtype_tag == 0 ? sizeof(float) : sizeof(int8_t);
        if (count > std::numeric_limits<size_t>::max() / item_size ||
            meta.byte_length != count * item_size) {
            throw std::runtime_error("leaf::Graph::load: initializer byte length mismatch");
        }
        if (graph.version_ >= 2 && meta.byte_offset % 32 != 0) {
            throw std::runtime_error("leaf::Graph::load: unaligned initializer offset");
        }
        if (!graph.init_index_.emplace(meta.name, graph.init_meta_.size()).second) {
            throw std::runtime_error("leaf::Graph::load: duplicate initializer name");
        }
        graph.init_meta_.push_back(std::move(meta));
    }

    if (graph.version_ >= 3) {
        graph.has_memory_plan_ = true;
        MemoryPlan& plan = graph.memory_plan_;
        plan.alignment = reader.u32();
        plan.arena_size = reader.u64();
        plan.graph_fingerprint = reader.string();
        if (plan.alignment < alignof(float) ||
            (plan.alignment & (plan.alignment - 1)) != 0 ||
            plan.arena_size > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("leaf::Graph::load: invalid memory-plan arena");
        }
        const uint32_t allocation_count = reader.u32();
        if (allocation_count > (graph.file_data_.size() - reader.offset) / 28) {
            throw std::runtime_error("leaf::Graph::load: invalid memory-plan count");
        }
        plan.allocations.reserve(allocation_count);
        for (uint32_t i = 0; i < allocation_count; ++i) {
            MemoryAllocation allocation;
            allocation.tensor = reader.string();
            allocation.offset = reader.u64();
            allocation.size = reader.u64();
            allocation.first_node = reader.u32();
            allocation.last_node = reader.u32();
            if (allocation.offset % plan.alignment != 0 ||
                allocation.size % plan.alignment != 0 ||
                allocation.offset > plan.arena_size ||
                allocation.size > plan.arena_size - allocation.offset ||
                allocation.first_node >= graph.nodes_.size() ||
                allocation.last_node < allocation.first_node ||
                allocation.last_node > graph.nodes_.size() ||
                graph.nodes_[allocation.first_node].outputs.size() != 1 ||
                graph.nodes_[allocation.first_node].outputs[0] != allocation.tensor) {
                throw std::runtime_error("leaf::Graph::load: invalid memory-plan allocation");
            }
            if (!graph.allocation_index_.emplace(allocation.tensor,
                                                 plan.allocations.size()).second) {
                throw std::runtime_error("leaf::Graph::load: duplicate memory-plan tensor");
            }
            plan.allocations.push_back(std::move(allocation));
        }
        plan.unplanned_tensors = reader.strings();
        std::unordered_set<std::string> unplanned(
            plan.unplanned_tensors.begin(), plan.unplanned_tensors.end());
        for (const Node& node : graph.nodes_) {
            for (const std::string& name : node.outputs) {
                if (graph.allocation_index_.find(name) == graph.allocation_index_.end() &&
                    unplanned.find(name) == unplanned.end()) {
                    throw std::runtime_error("leaf::Graph::load: memory plan omits tensor: " + name);
                }
            }
        }
        std::vector<std::vector<size_t>> starts(graph.nodes_.size() + 2);
        std::vector<std::vector<size_t>> ends(graph.nodes_.size() + 2);
        for (size_t index = 0; index < plan.allocations.size(); ++index) {
            const MemoryAllocation& allocation = plan.allocations[index];
            starts[allocation.first_node].push_back(index);
            ends[allocation.last_node + 1].push_back(index);
        }
        std::map<uint64_t, size_t> active;
        for (size_t node_index = 0; node_index <= graph.nodes_.size(); ++node_index) {
            for (const size_t index : ends[node_index]) {
                active.erase(plan.allocations[index].offset);
            }
            for (const size_t index : starts[node_index]) {
                const MemoryAllocation& allocation = plan.allocations[index];
                const auto next = active.lower_bound(allocation.offset);
                if ((next != active.end() &&
                     allocation.offset + allocation.size > next->first) ||
                    (next != active.begin() &&
                     std::prev(next)->first + plan.allocations[std::prev(next)->second].size >
                         allocation.offset)) {
                    throw std::runtime_error("leaf::Graph::load: overlapping live arena allocations");
                }
                active.emplace(allocation.offset, index);
            }
        }
    }

    graph.data_block_start_ = graph.version_ >= 2 ? aligned_32(reader.offset) : reader.offset;
    if (graph.data_block_start_ > graph.file_data_.size()) {
        throw std::runtime_error("leaf::Graph::load: truncated data block");
    }
    const size_t data_bytes = graph.file_data_.size() - graph.data_block_start_;
    for (const InitializerMeta& meta : graph.init_meta_) {
        if (meta.byte_offset > data_bytes ||
            meta.byte_length > data_bytes - static_cast<size_t>(meta.byte_offset)) {
            throw std::runtime_error("leaf::Graph::load: initializer data out of bounds: " + meta.name);
        }
    }
    return graph;
}

const InitializerMeta& Graph::initializer_meta(const std::string& name) const {
    const auto found = init_index_.find(name);
    if (found == init_index_.end()) {
        throw std::runtime_error("leaf::Graph::initializer_meta: no such initializer: " + name);
    }
    return init_meta_[found->second];
}

bool Graph::has_initializer(const std::string& name) const {
    return init_index_.find(name) != init_index_.end();
}

const MemoryAllocation* Graph::allocation_for(const std::string& name) const {
    const auto found = allocation_index_.find(name);
    return found == allocation_index_.end() ? nullptr :
        &memory_plan_.allocations[found->second];
}

const float* Graph::initializer_data(const std::string& name) const {
    const InitializerMeta& meta = initializer_meta(name);
    if (meta.dtype_tag != 0) {
        throw std::runtime_error("leaf::Graph::initializer_data: not float32: " + name);
    }
    return reinterpret_cast<const float*>(file_data_.data() + data_block_start_ +
                                          static_cast<size_t>(meta.byte_offset));
}

const int8_t* Graph::initializer_i8_data(const std::string& name) const {
    const InitializerMeta& meta = initializer_meta(name);
    if (meta.dtype_tag != 1) {
        throw std::runtime_error("leaf::Graph::initializer_i8_data: not int8: " + name);
    }
    return reinterpret_cast<const int8_t*>(file_data_.data() + data_block_start_ +
                                           static_cast<size_t>(meta.byte_offset));
}

}  // namespace leaf
