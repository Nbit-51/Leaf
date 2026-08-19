// Loads real ResNet-18 stem-conv weights from a .leaf file, runs them
// through leaf::conv2d, and diffs against a Python-computed reference
// output (itself already verified against PyTorch). This is the test
// that actually proves the C++ Conv2D kernel is numerically correct on
// real data, not just hand-checked toy cases.

#include "leaf/graph_parser.h"
#include "../kernels/conv2d.h"
#include "../kernels/im2col.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>

std::vector<float> read_binary_floats(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open: " + path);
    }
    std::vector<float> data(count);
    f.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));
    if (!f) {
        throw std::runtime_error("failed to read expected float count from: " + path);
    }
    return data;
}

const leaf::Node& conv_node_for_weight(const leaf::Graph& graph,
                                       const std::string& weight_name) {
    for (const leaf::Node& node : graph.nodes()) {
        if (node.op_type == "Conv" && node.inputs.size() >= 2 &&
            node.inputs[1] == weight_name) {
            return node;
        }
    }
    throw std::runtime_error("cannot find Conv node for weight: " + weight_name);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <graph.leaf> <input.bin> <expected_output.bin>\n";
        return 1;
    }

    try {
        leaf::Graph graph = leaf::Graph::load(argv[1]);

        const std::string weight_name = "onnx::Conv_193";
        const auto& meta = graph.initializer_meta(weight_name);
        const float* weights = graph.initializer_data(weight_name);
        const leaf::Node& conv_node = conv_node_for_weight(graph, weight_name);
        const float* bias = nullptr;
        if (conv_node.inputs.size() >= 3 && !conv_node.inputs[2].empty()) {
            bias = graph.initializer_data(conv_node.inputs[2]);
            const auto& bias_meta = graph.initializer_meta(conv_node.inputs[2]);
            if (bias_meta.shape.size() != 1 || bias_meta.shape[0] != meta.shape[0]) {
                throw std::runtime_error("Conv bias shape does not match output channels");
            }
            std::cout << "Using fused Conv bias initializer: " << conv_node.inputs[2] << "\n";
        }

        // shape = [out_c, in_c, kh, kw]
        size_t out_c = meta.shape[0];
        size_t in_c = meta.shape[1];
        size_t kh = meta.shape[2];
        size_t kw = meta.shape[3];

        size_t in_h = 224, in_w = 224;
        std::vector<float> input = read_binary_floats(argv[2], in_c * in_h * in_w);

        size_t out_h, out_w;
        leaf::conv2d_output_shape(in_h, in_w, kh, kw, 3, 3, 3, 3, 2, 2, 1, 1, out_h, out_w);

        std::vector<float> output(out_c * out_h * out_w);

        // ONNX inference export may fold BatchNorm into Conv, in which case
        // the Conv has a third input containing its fused bias.  Read the
        // graph connection rather than assuming that bias is absent.
        leaf::conv2d(
            input.data(), in_c, in_h, in_w,
            weights, out_c, kh, kw,
            bias,
            3, 3, 3, 3,   // pads
            2, 2,         // strides
            1, 1,         // dilations
            output.data()
        );

        std::vector<float> expected = read_binary_floats(argv[3], out_c * out_h * out_w);

        float max_abs_diff = 0.0f;
        size_t max_idx = 0;
        for (size_t i = 0; i < output.size(); ++i) {
            float diff = std::fabs(output[i] - expected[i]);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_idx = i;
            }
        }

        std::cout << "Output shape: (" << out_c << ", " << out_h << ", " << out_w << ")\n";
        std::cout << "Max absolute difference vs Python reference: " << max_abs_diff << "\n";
        std::cout << "  at index " << max_idx
                   << " -- cpp=" << output[max_idx] << ", python=" << expected[max_idx] << "\n";

        const float tolerance = 1e-3f;
        if (max_abs_diff < tolerance) {
            std::cout << "\nPASS: C++ conv2d matches Python reference within tolerance.\n";
            return 0;
        } else {
            std::cout << "\nFAIL: divergence exceeds tolerance (" << tolerance << ").\n";
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
