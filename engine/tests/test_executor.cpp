// End-to-end C++ runtime test.  It writes a tiny valid .leaf artifact,
// reloads it through the production parser, and executes
// Conv -> Relu -> GlobalAveragePool -> Flatten -> Gemm.

#include "leaf/graph_parser.h"
#include "leaf/runtime/executor.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Initializer {
    std::string name;
    std::vector<uint32_t> shape;
    std::vector<float> values;
    uint64_t offset = 0;
};

void write_u32(std::ofstream& file, uint32_t value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_u64(std::ofstream& file, uint64_t value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_string(std::ofstream& file, const std::string& value) {
    write_u32(file, static_cast<uint32_t>(value.size()));
    file.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void write_string_array(std::ofstream& file, const std::vector<std::string>& values) {
    write_u32(file, static_cast<uint32_t>(values.size()));
    for (const std::string& value : values) {
        write_string(file, value);
    }
}

void write_node(std::ofstream& file, const std::string& operation,
                const std::vector<std::string>& inputs,
                const std::vector<std::string>& outputs,
                const std::string& attributes) {
    write_string(file, operation);
    write_string_array(file, inputs);
    write_string_array(file, outputs);
    write_string(file, attributes);
}

uint64_t align32(uint64_t offset) {
    return (offset + 31u) & ~uint64_t{31u};
}

void write_test_graph(const std::string& path) {
    std::vector<Initializer> initializers = {
        {"w", {1, 1, 1, 1}, {2.0f}},
        {"b", {1}, {1.0f}},
        {"fc", {2, 1}, {4.0f, -2.0f}},
        {"fc_b", {2}, {1.0f, 10.0f}},
    };
    uint64_t data_offset = 0;
    for (Initializer& initializer : initializers) {
        data_offset = align32(data_offset);
        initializer.offset = data_offset;
        data_offset += static_cast<uint64_t>(initializer.values.size() * sizeof(float));
    }

    std::ofstream file(path, std::ios::binary);
    assert(file);
    file.write("LEAF", 4);
    write_u32(file, 1);  // format version
    write_u32(file, 5);  // node count
    write_u32(file, static_cast<uint32_t>(initializers.size()));
    write_string_array(file, {"input"});
    write_string_array(file, {"output"});

    write_node(file, "Conv", {"input", "w", "b"}, {"conv_out"},
               R"({"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "group": 1})");
    write_node(file, "Relu", {"conv_out"}, {"relu_out"}, "{}");
    write_node(file, "GlobalAveragePool", {"relu_out"}, {"pooled"}, "{}");
    write_node(file, "Flatten", {"pooled"}, {"flat"}, R"({"axis": 1})");
    write_node(file, "Gemm", {"flat", "fc", "fc_b"}, {"output"}, R"({"transB": 1})");

    for (const Initializer& initializer : initializers) {
        write_string(file, initializer.name);
        write_u32(file, static_cast<uint32_t>(initializer.shape.size()));
        for (const uint32_t dimension : initializer.shape) {
            write_u32(file, dimension);
        }
        const uint8_t dtype = 0;
        file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
        write_u64(file, initializer.offset);
        write_u64(file, static_cast<uint64_t>(initializer.values.size() * sizeof(float)));
    }

    uint64_t written = 0;
    const char zero = 0;
    for (const Initializer& initializer : initializers) {
        while (written < initializer.offset) {
            file.write(&zero, 1);
            ++written;
        }
        file.write(reinterpret_cast<const char*>(initializer.values.data()),
                   static_cast<std::streamsize>(initializer.values.size() * sizeof(float)));
        written += static_cast<uint64_t>(initializer.values.size() * sizeof(float));
    }
    assert(file);
}

bool close(float left, float right) {
    return std::fabs(left - right) < 1e-5f;
}

}  // namespace

int main() {
    const std::string artifact_path = "leaf_executor_test.leaf";
    write_test_graph(artifact_path);

    const leaf::Graph graph = leaf::Graph::load(artifact_path);
    const leaf::Tensor input{{1, 1, 2, 2}, {-1.0f, 2.0f, 3.0f, -4.0f}};
    const leaf::Tensor output = leaf::Executor().run(graph, input);

    assert((output.shape == std::vector<size_t>{1, 2}));
    // Conv+bias -> [-1, 5, 7, -7], ReLU -> [0, 5, 7, 0], GAP -> 3.
    // Gemm: [3] @ [[4], [-2]]^T + [1, 10] = [13, 4].
    assert(close(output.values[0], 13.0f));
    assert(close(output.values[1], 4.0f));
    std::remove(artifact_path.c_str());
    std::cout << "PASS: C++ graph executor end-to-end\n";
    return 0;
}
