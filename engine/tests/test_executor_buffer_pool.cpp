// Covers the ops test_executor.cpp does not: Add, MaxPool, and
// BatchNormalization. These paths were changed by the thread_local
// activation-buffer pool added to executor.cpp (see ADR-003), so they
// need their own numerical check rather than relying on test_executor.cpp
// to exercise them incidentally.
//
// The graph also runs twice on the same Executor to confirm that reusing
// pooled buffers across repeated run() calls -- not just across nodes
// within one call -- produces identical output both times, since
// g_buffer_pool is thread_local and persists between calls exactly like
// leaf_bench's repeated-run loop would exercise it.

#include "leaf/graph_parser.h"
#include "leaf/runtime/executor.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

// input (1,2,4,4) --Conv(w_a,b_a)--> a_out (1,2,4,4)
//       \-Conv(w_b,b_b)--> b_out (1,2,4,4)
// Add(a_out, b_out) -> sum_out (1,2,4,4)
// MaxPool k2s2 -> pooled (1,2,2,2)
// BatchNormalization -> bn_out (1,2,2,2)
// GlobalAveragePool -> gap (1,2,1,1)
// Flatten axis1 -> flat (1,2)
// Gemm(flat, fc^T, fc_b) -> output (1,2)
void write_test_graph(const std::string& path) {
    std::vector<Initializer> initializers = {
        {"w_a", {2, 2, 1, 1}, {1.0f, 0.0f, 0.0f, 1.0f}},   // identity-per-channel 1x1 conv
        {"b_a", {2}, {0.0f, 0.0f}},
        {"w_b", {2, 2, 1, 1}, {0.5f, 0.0f, 0.0f, 0.5f}},   // half-identity 1x1 conv
        {"b_b", {2}, {1.0f, -1.0f}},
        {"bn_scale", {2}, {2.0f, 0.5f}},
        {"bn_bias", {2}, {0.0f, 1.0f}},
        {"bn_mean", {2}, {0.0f, 0.0f}},
        {"bn_var", {2}, {1.0f, 1.0f}},
        {"fc", {2, 2}, {1.0f, 2.0f, -1.0f, 1.0f}},
        {"fc_b", {2}, {0.0f, 0.0f}},
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
    write_u32(file, 8);  // node count
    write_u32(file, static_cast<uint32_t>(initializers.size()));
    write_string_array(file, {"input"});
    write_string_array(file, {"output"});

    write_node(file, "Conv", {"input", "w_a", "b_a"}, {"a_out"},
               R"({"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "group": 1})");
    write_node(file, "Conv", {"input", "w_b", "b_b"}, {"b_out"},
               R"({"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "group": 1})");
    write_node(file, "Add", {"a_out", "b_out"}, {"sum_out"}, "{}");
    write_node(file, "MaxPool", {"sum_out"}, {"pooled"},
               R"({"kernel_shape": [2, 2], "pads": [0, 0, 0, 0], "strides": [2, 2]})");
    write_node(file, "BatchNormalization", {"pooled", "bn_scale", "bn_bias", "bn_mean", "bn_var"},
               {"bn_out"}, R"({"epsilon": 0.0})");
    write_node(file, "GlobalAveragePool", {"bn_out"}, {"gap"}, "{}");
    write_node(file, "Flatten", {"gap"}, {"flat"}, R"({"axis": 1})");
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
    const std::string artifact_path = "leaf_executor_buffer_pool_test.leaf";
    write_test_graph(artifact_path);

    const leaf::Graph graph = leaf::Graph::load(artifact_path);
    const leaf::Tensor input{{1, 2, 4, 4},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
         -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16}};

    // Hand-computed reference:
    // a_out = input (identity 1x1 conv, zero bias) -> channel0 = input ch0, channel1 = input ch1
    // b_out = 0.5*input + bias -> ch0 = 0.5*input_ch0 + 1, ch1 = 0.5*input_ch1 - 1
    // sum_out ch0 = 1.5*input_ch0 + 1, ch1 = 1.5*input_ch1 - 1
    // MaxPool 2x2 s2 over each 4x4 -> 2x2, taking max of each 2x2 block.
    // ch0 input block maxes: [6,8;14,16] -> sum_out ch0 blocks: 1.5*that+1 = [10,13;22,25]
    // ch1 input block maxes (least negative = max): input ch1 blocks are negative,
    //   max of {-1,-2,-5,-6}=-1, {-3,-4,-7,-8}=-3, {-9,-10,-13,-14}=-9, {-11,-12,-15,-16}=-11
    //   sum_out ch1 = 1.5*max_val - 1: [-2.5,-5.5;-14.5,-17.5]
    // BatchNormalization epsilon=0, mean=0, var=1: ch0 -> 2*x + 0, ch1 -> 0.5*x + 1
    // bn ch0 = [20,26;44,50], bn ch1 = [-0.25,-1.75;-6.25,-7.75]
    // GlobalAveragePool: ch0 mean = (20+26+44+50)/4 = 35, ch1 mean = (-0.25-1.75-6.25-7.75)/4 = -4.0
    // Flatten -> [35, -4]
    // Gemm transB: fc is (2,2) row-major [[1,2],[-1,1]], B^T applied:
    //   out[0] = 35*1 + (-4)*2 = 27
    //   out[1] = 35*-1 + (-4)*1 = -39
    const float expected0 = 27.0f;
    const float expected1 = -39.0f;

    const leaf::Executor executor;
    for (int run_index = 0; run_index < 2; ++run_index) {
        const leaf::Tensor output = executor.run(graph, input);
        assert((output.shape == std::vector<size_t>{1, 2}));
        if (!close(output.values[0], expected0) || !close(output.values[1], expected1)) {
            std::cerr << "run " << run_index << " mismatch: got ["
                      << output.values[0] << ", " << output.values[1] << "], expected ["
                      << expected0 << ", " << expected1 << "]\n";
            return 1;
        }
    }

    std::remove(artifact_path.c_str());
    std::cout << "PASS: Add/MaxPool/BatchNormalization + repeated-run buffer pool reuse\n";
    return 0;
}
