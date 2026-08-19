// Correctness diagnostic for the ResNet-18 stem convolution.
//
// This deliberately avoids im2col and GEMM.  It also runs the normal
// conv2d path, so one command identifies whether a mismatch originates in
// the reference-data setup or in the optimized decomposition.

#include "leaf/graph_parser.h"
#include "../kernels/conv2d.h"
#include "../kernels/im2col.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_binary_floats(const std::string& path, size_t count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open: " + path);
    }

    std::vector<float> data(count);
    file.read(reinterpret_cast<char*>(data.data()),
              static_cast<std::streamsize>(count * sizeof(float)));
    if (file.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
        throw std::runtime_error("wrong byte count in: " + path);
    }
    return data;
}

void direct_conv2d(
    const float* input, size_t in_c, size_t in_h, size_t in_w,
    const float* weights, size_t out_c, size_t kh, size_t kw,
    const float* bias,
    size_t pad_h, size_t pad_w, size_t stride_h, size_t stride_w,
    float* output, size_t out_h, size_t out_w
) {
    for (size_t oc = 0; oc < out_c; ++oc) {
        for (size_t oh = 0; oh < out_h; ++oh) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                float sum = bias == nullptr ? 0.0f : bias[oc];
                for (size_t ic = 0; ic < in_c; ++ic) {
                    for (size_t ki = 0; ki < kh; ++ki) {
                        const long input_h = static_cast<long>(oh * stride_h + ki)
                                           - static_cast<long>(pad_h);
                        if (input_h < 0 || input_h >= static_cast<long>(in_h)) {
                            continue;
                        }
                        for (size_t kj = 0; kj < kw; ++kj) {
                            const long input_w = static_cast<long>(ow * stride_w + kj)
                                               - static_cast<long>(pad_w);
                            if (input_w < 0 || input_w >= static_cast<long>(in_w)) {
                                continue;
                            }

                            const size_t weight_index = (((oc * in_c + ic) * kh + ki) * kw + kj);
                            const size_t input_index = ((ic * in_h + static_cast<size_t>(input_h)) * in_w
                                                      + static_cast<size_t>(input_w));
                            sum += weights[weight_index] * input[input_index];
                        }
                    }
                }
                output[(oc * out_h + oh) * out_w + ow] = sum;
            }
        }
    }
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

struct Difference {
    float max_abs = 0.0f;
    size_t index = 0;
};

Difference max_difference(const std::vector<float>& actual,
                          const std::vector<float>& expected) {
    Difference difference;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float value = std::fabs(actual[i] - expected[i]);
        if (value > difference.max_abs) {
            difference.max_abs = value;
            difference.index = i;
        }
    }
    return difference;
}

void print_difference(const char* label, const std::vector<float>& actual,
                      const std::vector<float>& expected) {
    const Difference difference = max_difference(actual, expected);
    std::cout << label << ": " << difference.max_abs
              << " at index " << difference.index
              << " (actual=" << actual[difference.index]
              << ", expected=" << expected[difference.index] << ")\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <graph.leaf> <input.bin> <expected_output.bin>\n";
        return 1;
    }

    try {
        constexpr size_t in_h = 224;
        constexpr size_t in_w = 224;
        constexpr size_t pad_h = 3;
        constexpr size_t pad_w = 3;
        constexpr size_t stride_h = 2;
        constexpr size_t stride_w = 2;

        const leaf::Graph graph = leaf::Graph::load(argv[1]);
        const std::string weight_name = "onnx::Conv_193";
        const auto& meta = graph.initializer_meta(weight_name);
        if (meta.shape.size() != 4) {
            throw std::runtime_error("expected a 4-D OIHW weight tensor");
        }

        const size_t out_c = meta.shape[0];
        const size_t in_c = meta.shape[1];
        const size_t kh = meta.shape[2];
        const size_t kw = meta.shape[3];
        const float* weights = graph.initializer_data(weight_name);
        const leaf::Node& conv_node = conv_node_for_weight(graph, weight_name);
        const float* bias = nullptr;
        if (conv_node.inputs.size() >= 3 && !conv_node.inputs[2].empty()) {
            bias = graph.initializer_data(conv_node.inputs[2]);
            const auto& bias_meta = graph.initializer_meta(conv_node.inputs[2]);
            if (bias_meta.shape.size() != 1 || bias_meta.shape[0] != out_c) {
                throw std::runtime_error("Conv bias shape does not match output channels");
            }
            std::cout << "Using fused Conv bias initializer: " << conv_node.inputs[2] << "\n";
        } else {
            std::cout << "Conv has no bias initializer.\n";
        }

        size_t out_h = 0;
        size_t out_w = 0;
        leaf::conv2d_output_shape(in_h, in_w, kh, kw,
                                  pad_h, pad_w, pad_h, pad_w,
                                  stride_h, stride_w, 1, 1, out_h, out_w);

        const std::vector<float> input = read_binary_floats(argv[2], in_c * in_h * in_w);
        const std::vector<float> expected = read_binary_floats(argv[3], out_c * out_h * out_w);
        std::vector<float> direct(expected.size());
        std::vector<float> fast(expected.size());

        direct_conv2d(input.data(), in_c, in_h, in_w, weights, out_c, kh, kw, bias,
                      pad_h, pad_w, stride_h, stride_w, direct.data(), out_h, out_w);
        leaf::conv2d(input.data(), in_c, in_h, in_w, weights, out_c, kh, kw, bias,
                     pad_h, pad_w, pad_h, pad_w, stride_h, stride_w, 1, 1, fast.data());

        std::cout << "Output shape: (" << out_c << ", " << out_h << ", " << out_w << ")\n";
        print_difference("Direct convolution vs Python", direct, expected);
        print_difference("Fast convolution vs direct", fast, direct);
        print_difference("Fast convolution vs Python", fast, expected);

        constexpr float tolerance = 1e-3f;
        const Difference direct_difference = max_difference(direct, expected);
        const Difference fast_difference = max_difference(fast, direct);
        if (direct_difference.max_abs < tolerance && fast_difference.max_abs < tolerance) {
            std::cout << "PASS: direct and fast convolution both match.\n";
            return 0;
        }
        if (direct_difference.max_abs >= tolerance) {
            std::cout << "FAIL: direct convolution differs from the Python reference.\n";
        } else {
            std::cout << "FAIL: fast path differs from direct convolution; inspect im2col/GEMM.\n";
        }
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
