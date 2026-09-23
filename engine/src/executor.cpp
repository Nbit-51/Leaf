#include "leaf/runtime/executor.h"

#include "conv2d.h"
#include "gemm.h"
#include "im2col.h"
#include "leaf/kernels/conv.h"
#include "leaf/kernels/gemm.h"
#include "leaf/runtime/arena.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace leaf {
namespace {

struct TensorView {
    const float* data = nullptr;
    std::vector<size_t> shape;
};

size_t element_count(const std::vector<size_t>& shape) {
    size_t count = 1;
    for (const size_t dimension : shape) {
        if (dimension == 0 || count > std::numeric_limits<size_t>::max() / dimension) {
            throw std::runtime_error("invalid or overflowing tensor shape");
        }
        count *= dimension;
    }
    return count;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("leaf::Executor: " + message);
    }
}

std::vector<size_t> integer_list_attribute(const Node& node, const std::string& key,
                                           const std::vector<size_t>& fallback) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = node.attributes_json.find(needle);
    if (key_pos == std::string::npos) {
        return fallback;
    }

    const size_t colon = node.attributes_json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        throw std::runtime_error("malformed attribute '" + key + "' on " + node.op_type);
    }
    size_t pos = node.attributes_json.find_first_not_of(" \t\n\r", colon + 1);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing attribute value for '" + key + "'");
    }

    std::vector<size_t> values;
    if (node.attributes_json[pos] == '[') {
        ++pos;
        while (pos < node.attributes_json.size()) {
            pos = node.attributes_json.find_first_not_of(" \t\n\r,", pos);
            if (pos == std::string::npos || node.attributes_json[pos] == ']') {
                break;
            }
            char* end = nullptr;
            const long value = std::strtol(node.attributes_json.c_str() + pos, &end, 10);
            if (end == node.attributes_json.c_str() + pos || value < 0) {
                throw std::runtime_error("invalid integer list attribute '" + key + "'");
            }
            values.push_back(static_cast<size_t>(value));
            pos = static_cast<size_t>(end - node.attributes_json.c_str());
        }
    } else {
        char* end = nullptr;
        const long value = std::strtol(node.attributes_json.c_str() + pos, &end, 10);
        if (end == node.attributes_json.c_str() + pos || value < 0) {
            throw std::runtime_error("invalid integer attribute '" + key + "'");
        }
        values.push_back(static_cast<size_t>(value));
    }
    return values.empty() ? fallback : values;
}

float float_attribute(const Node& node, const std::string& key, float fallback) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = node.attributes_json.find(needle);
    if (key_pos == std::string::npos) {
        return fallback;
    }
    const size_t colon = node.attributes_json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        throw std::runtime_error("malformed attribute '" + key + "' on " + node.op_type);
    }
    const size_t pos = node.attributes_json.find_first_not_of(" \t\n\r", colon + 1);
    char* end = nullptr;
    const float value = std::strtof(node.attributes_json.c_str() + pos, &end);
    if (end == node.attributes_json.c_str() + pos) {
        throw std::runtime_error("invalid float attribute '" + key + "'");
    }
    return value;
}

std::string string_attribute(const Node& node, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = node.attributes_json.find(needle);
    if (key_pos == std::string::npos) {
        return {};
    }
    const size_t colon = node.attributes_json.find(':', key_pos + needle.size());
    const size_t quote_begin = node.attributes_json.find('"', colon + 1);
    const size_t quote_end = quote_begin == std::string::npos
        ? std::string::npos : node.attributes_json.find('"', quote_begin + 1);
    if (quote_begin == std::string::npos || quote_end == std::string::npos) {
        throw std::runtime_error("invalid string attribute '" + key + "'");
    }
    return node.attributes_json.substr(quote_begin + 1, quote_end - quote_begin - 1);
}

// Grow-only pool of retired activation buffers, reused across nodes within
// a run() call (and across repeated run() calls on the same thread)
// instead of a fresh heap allocation per node per pass. Mirrors the
// thread_local scratch buffer already used by conv2d() (see ADR-002) but
// sized per-tensor rather than per-conv-layer, since intermediate
// activation shapes vary node to node through the graph.
//
// Every op that acquires a buffer here either fully overwrites every
// element before any value is read back out (Conv -- conv2d() memsets
// internally; MaxPool, GlobalAveragePool, Add -- every index assigned
// exactly once by the loop), copies the full input range in before
// modifying in place (Relu, Flatten, BatchNormalization), or explicitly
// zero-fills before an accumulating write (Gemm's beta*C term). So reused
// content never leaks between nodes or runs.
thread_local std::vector<std::vector<float>> g_buffer_pool;

std::vector<float> acquire_buffer(size_t needed) {
    // Best-fit: reuse the freed buffer whose capacity is closest to (but
    // not below) what's needed, so a right-sized buffer already in the
    // pool isn't passed over in favor of growing an oversized one.
    size_t best = g_buffer_pool.size();
    for (size_t i = 0; i < g_buffer_pool.size(); ++i) {
        if (g_buffer_pool[i].capacity() >= needed &&
            (best == g_buffer_pool.size() ||
             g_buffer_pool[i].capacity() < g_buffer_pool[best].capacity())) {
            best = i;
        }
    }
    if (best == g_buffer_pool.size()) {
        // Nothing large enough: grow the largest available buffer instead
        // of discarding pool space, or allocate fresh if the pool is empty.
        if (g_buffer_pool.empty()) {
            return std::vector<float>(needed);
        }
        best = 0;
        for (size_t i = 1; i < g_buffer_pool.size(); ++i) {
            if (g_buffer_pool[i].capacity() > g_buffer_pool[best].capacity()) {
                best = i;
            }
        }
    }
    std::vector<float> buffer = std::move(g_buffer_pool[best]);
    g_buffer_pool.erase(g_buffer_pool.begin() + static_cast<std::ptrdiff_t>(best));
    buffer.resize(needed);
    return buffer;
}

void release_buffer(std::vector<float>&& buffer) {
    g_buffer_pool.push_back(std::move(buffer));
}

struct OutputAllocator {
    const Graph& graph;
    runtime::Arena* arena;
    const std::string& name;

    Tensor make(std::vector<size_t> shape) const {
        Tensor result;
        result.shape = std::move(shape);
        const size_t count = element_count(result.shape);
        const MemoryAllocation* allocation = graph.allocation_for(name);
        if (allocation != nullptr) {
            require(arena != nullptr && count <= allocation->size / sizeof(float),
                    "memory-plan allocation is smaller than runtime tensor: " + name);
            result.arena_data = static_cast<float*>(arena->at(
                static_cast<size_t>(allocation->offset),
                count * sizeof(float)));
        } else {
            result.values = acquire_buffer(count);
        }
        return result;
    }
};

TensorView tensor_for(const Graph& graph,
                      const std::unordered_map<std::string, Tensor>& intermediates,
                      const std::unordered_map<std::string, TensorView>& graph_inputs,
                      const std::string& name) {
    const auto intermediate = intermediates.find(name);
    if (intermediate != intermediates.end()) {
        return {intermediate->second.data(), intermediate->second.shape};
    }
    const auto input = graph_inputs.find(name);
    if (input != graph_inputs.end()) {
        return input->second;
    }
    if (graph.has_initializer(name)) {
        const InitializerMeta& meta = graph.initializer_meta(name);
        return {graph.initializer_data(name),
                std::vector<size_t>(meta.shape.begin(), meta.shape.end())};
    }
    throw std::runtime_error("leaf::Executor: tensor not found: " + name);
}

std::vector<int8_t> quantize_input(const float* values, size_t count, float scale) {
    require(std::isfinite(scale) && scale > 0.0f, "invalid activation scale");
    std::vector<int8_t> result(count);
    for (size_t index = 0; index < count; ++index) {
        require(std::isfinite(values[index]), "INT8 input contains a non-finite value");
        const float scaled = values[index] / scale;
        const float clamped = std::max(-127.0f, std::min(127.0f, scaled));
        result[index] = static_cast<int8_t>(std::nearbyint(clamped));
    }
    return result;
}

float apply_activation(float value, const std::string& name) {
    if (name.empty()) return value;
    if (name == "Relu") return std::max(value, 0.0f);
    if (name == "Silu") return value / (1.0f + std::exp(-value));
    if (name == "Gelu") {
        constexpr float root_two_over_pi = 0.7978845608028654f;
        return 0.5f * value * (1.0f +
            std::tanh(root_two_over_pi * (value + 0.044715f * value * value * value)));
    }
    throw std::runtime_error("leaf::Executor: unsupported fused activation: " + name);
}

void require_channel_scales(const Node& node, size_t channels, uint8_t axis) {
    require(node.quantized, "missing INT8 node parameters");
    require(node.weight_axis == axis && node.weight_scales.size() == channels,
            "INT8 weight scales do not match output channels");
}

Tensor execute_quant_conv(const Graph& graph, const Node& node, const TensorView& input,
                          const std::unordered_map<std::string, Tensor>& intermediates,
                          const std::unordered_map<std::string, TensorView>& graph_inputs,
                          const OutputAllocator& allocator) {
    require(input.shape.size() == 4 && input.shape[0] == 1,
            "INT8 Conv requires one NCHW image");
    const std::string& weight_name = node.inputs.at(1);
    const InitializerMeta& meta = graph.initializer_meta(weight_name);
    require(meta.dtype_tag == 1 && meta.shape.size() == 4,
            "INT8 Conv weight must be an OIHW initializer");
    const size_t out_c = meta.shape[0], in_c = meta.shape[1];
    const size_t kh = meta.shape[2], kw = meta.shape[3];
    require(in_c == input.shape[1], "INT8 Conv input-channel mismatch");
    require_channel_scales(node, out_c, 0);
    const std::vector<size_t> pads = integer_list_attribute(node, "pads", {0, 0, 0, 0});
    const std::vector<size_t> strides = integer_list_attribute(node, "strides", {1, 1});
    const std::vector<size_t> dilations = integer_list_attribute(node, "dilations", {1, 1});
    const std::vector<size_t> groups = integer_list_attribute(node, "group", {1});
    require(pads.size() == 4 && strides.size() == 2 && dilations.size() == 2 &&
            groups.size() == 1 && groups[0] == 1, "unsupported INT8 Conv attributes");
    size_t out_h = 0, out_w = 0;
    conv2d_output_shape(input.shape[2], input.shape[3], kh, kw,
                        pads[0], pads[1], pads[2], pads[3], strides[0], strides[1],
                        dilations[0], dilations[1], out_h, out_w);
    const float* bias = nullptr;
    if (node.inputs.size() >= 3 && !node.inputs[2].empty()) {
        const TensorView bias_tensor = tensor_for(graph, intermediates, graph_inputs, node.inputs[2]);
        require(bias_tensor.shape == std::vector<size_t>{out_c},
                "INT8 Conv bias shape must be [out_channels]");
        bias = bias_tensor.data;
    }

    Tensor output = allocator.make({1, out_c, out_h, out_w});
    const std::vector<int8_t> qinput = quantize_input(input.data, element_count(input.shape),
                                                      node.input_scale);
    const int8_t* weight = graph.initializer_i8_data(weight_name);
    const std::string activation = string_attribute(node, "activation");
    require(activation.empty() || activation == "Relu", "unsupported INT8 Conv activation");
    const kernels::Activation fused = activation == "Relu"
        ? kernels::Activation::Relu : kernels::Activation::None;
    const size_t patch = in_c * kh * kw;
    const bool fast_layout = pads[0] == pads[2] && pads[1] == pads[3] &&
                             dilations[0] == 1 && dilations[1] == 1 &&
                             patch <= static_cast<size_t>(INT32_MAX / (127 * 127));
    if (fast_layout) {
        const kernels::Conv2DShape shape{1, in_c, input.shape[2], input.shape[3],
                                         out_c, kh, kw, pads[0], pads[1],
                                         strides[0], strides[1]};
        std::vector<int8_t> packed(patch * out_c);
        std::vector<int8_t> workspace(out_h * out_w * patch);
        std::vector<float> gemm_output(out_h * out_w * out_c);
        kernels::pack_conv_weights_i8(weight, packed.data(), shape);
        kernels::conv2d_i8_im2col(qinput.data(), packed.data(), bias, node.input_scale,
                                  node.weight_scales.data(), output.data(),
                                  workspace.data(), gemm_output.data(), shape, fused);
    } else {
        // General padding/dilation path. Accumulate in int64 to avoid overflow
        // for unusually wide kernels that exceed the int32 fast-path limit.
        for (size_t oc = 0; oc < out_c; ++oc) {
            for (size_t y = 0; y < out_h; ++y) {
                for (size_t x = 0; x < out_w; ++x) {
                    int64_t sum = 0;
                    for (size_t ic = 0; ic < in_c; ++ic) {
                        for (size_t ky = 0; ky < kh; ++ky) {
                            for (size_t kx = 0; kx < kw; ++kx) {
                                const int64_t iy = static_cast<int64_t>(y * strides[0] + ky * dilations[0]) -
                                                   static_cast<int64_t>(pads[0]);
                                const int64_t ix = static_cast<int64_t>(x * strides[1] + kx * dilations[1]) -
                                                   static_cast<int64_t>(pads[1]);
                                if (iy < 0 || ix < 0 || iy >= static_cast<int64_t>(input.shape[2]) ||
                                    ix >= static_cast<int64_t>(input.shape[3])) continue;
                                const size_t input_index = (ic * input.shape[2] + static_cast<size_t>(iy)) *
                                                           input.shape[3] + static_cast<size_t>(ix);
                                const size_t weight_index = ((oc * in_c + ic) * kh + ky) * kw + kx;
                                sum += static_cast<int32_t>(qinput[input_index]) *
                                       static_cast<int32_t>(weight[weight_index]);
                            }
                        }
                    }
                    float value = static_cast<float>(sum) * node.input_scale * node.weight_scales[oc];
                    if (bias != nullptr) value += bias[oc];
                    output.data()[(oc * out_h + y) * out_w + x] = apply_activation(value, activation);
                }
            }
        }
    }
    return output;
}

Tensor execute_conv(const Graph& graph, const Node& node, const TensorView& input,
                    const std::unordered_map<std::string, Tensor>& intermediates,
                    const std::unordered_map<std::string, TensorView>& graph_inputs,
                    const OutputAllocator& allocator) {
    require(input.shape.size() == 4 && input.shape[0] == 1,
            "Conv currently supports one NCHW image at a time");
    const TensorView weights = tensor_for(graph, intermediates, graph_inputs, node.inputs.at(1));
    require(weights.shape.size() == 4, "Conv weights must use OIHW layout");
    require(weights.shape[1] == input.shape[1], "Conv input-channel mismatch");

    const std::vector<size_t> pads = integer_list_attribute(node, "pads", {0, 0, 0, 0});
    const std::vector<size_t> strides = integer_list_attribute(node, "strides", {1, 1});
    const std::vector<size_t> dilations = integer_list_attribute(node, "dilations", {1, 1});
    const std::vector<size_t> groups = integer_list_attribute(node, "group", {1});
    require(pads.size() == 4 && strides.size() == 2 && dilations.size() == 2 && groups.size() == 1,
            "invalid Conv attributes");
    require(groups[0] == 1, "grouped Conv is not implemented yet");

    const float* bias = nullptr;
    if (node.inputs.size() >= 3 && !node.inputs[2].empty()) {
        const TensorView bias_tensor = tensor_for(graph, intermediates, graph_inputs, node.inputs[2]);
        require(bias_tensor.shape.size() == 1 && bias_tensor.shape[0] == weights.shape[0],
                "Conv bias shape must be [out_channels]");
        bias = bias_tensor.data;
    }

    size_t out_h = 0;
    size_t out_w = 0;
    conv2d_output_shape(input.shape[2], input.shape[3], weights.shape[2], weights.shape[3],
                        pads[0], pads[1], pads[2], pads[3], strides[0], strides[1],
                        dilations[0], dilations[1], out_h, out_w);
    Tensor output = allocator.make({1, weights.shape[0], out_h, out_w});
    conv2d(input.data, input.shape[1], input.shape[2], input.shape[3],
           weights.data, weights.shape[0], weights.shape[2], weights.shape[3], bias,
           pads[0], pads[1], pads[2], pads[3], strides[0], strides[1],
           dilations[0], dilations[1], output.data());

    const std::string activation = string_attribute(node, "activation");
    if (activation == "Relu") {
        for (size_t i = 0; i < output.element_count(); ++i) {
            output.data()[i] = std::max(output.data()[i], 0.0f);
        }
    } else {
        require(activation.empty(), "unsupported fused Conv activation: " + activation);
    }
    return output;
}

Tensor execute_maxpool(const Node& node, const TensorView& input,
                       const OutputAllocator& allocator) {
    require(input.shape.size() == 4 && input.shape[0] == 1,
            "MaxPool currently supports one NCHW image at a time");
    const std::vector<size_t> kernel = integer_list_attribute(node, "kernel_shape", {});
    const std::vector<size_t> pads = integer_list_attribute(node, "pads", {0, 0, 0, 0});
    const std::vector<size_t> strides = integer_list_attribute(node, "strides", {1, 1});
    const std::vector<size_t> dilations = integer_list_attribute(node, "dilations", {1, 1});
    const std::vector<size_t> ceil_mode = integer_list_attribute(node, "ceil_mode", {0});
    require(kernel.size() == 2 && pads.size() == 4 && strides.size() == 2 && dilations.size() == 2,
            "invalid MaxPool attributes");
    require(dilations[0] == 1 && dilations[1] == 1 && ceil_mode.size() == 1 && ceil_mode[0] == 0,
            "dilated or ceil-mode MaxPool is not implemented yet");

    size_t out_h = 0;
    size_t out_w = 0;
    conv2d_output_shape(input.shape[2], input.shape[3], kernel[0], kernel[1],
                        pads[0], pads[1], pads[2], pads[3], strides[0], strides[1],
                        1, 1, out_h, out_w);
    Tensor output = allocator.make({1, input.shape[1], out_h, out_w});
    for (size_t channel = 0; channel < input.shape[1]; ++channel) {
        for (size_t oh = 0; oh < out_h; ++oh) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                float maximum = -std::numeric_limits<float>::infinity();
                for (size_t kh = 0; kh < kernel[0]; ++kh) {
                    const long ih = static_cast<long>(oh * strides[0] + kh) - static_cast<long>(pads[0]);
                    if (ih < 0 || ih >= static_cast<long>(input.shape[2])) {
                        continue;
                    }
                    for (size_t kw = 0; kw < kernel[1]; ++kw) {
                        const long iw = static_cast<long>(ow * strides[1] + kw) - static_cast<long>(pads[1]);
                        if (iw >= 0 && iw < static_cast<long>(input.shape[3])) {
                            maximum = std::max(maximum, input.data[(channel * input.shape[2]
                                + static_cast<size_t>(ih)) * input.shape[3] + static_cast<size_t>(iw)]);
                        }
                    }
                }
                output.data()[(channel * out_h + oh) * out_w + ow] = maximum;
            }
        }
    }
    return output;
}

Tensor execute_global_average_pool(const TensorView& input,
                                   const OutputAllocator& allocator) {
    require(input.shape.size() == 4 && input.shape[0] == 1,
            "GlobalAveragePool currently supports one NCHW image at a time");
    const size_t spatial = input.shape[2] * input.shape[3];
    Tensor output = allocator.make({1, input.shape[1], 1, 1});
    for (size_t channel = 0; channel < input.shape[1]; ++channel) {
        float sum = 0.0f;
        const float* channel_data = input.data + channel * spatial;
        for (size_t index = 0; index < spatial; ++index) {
            sum += channel_data[index];
        }
        output.data()[channel] = sum / static_cast<float>(spatial);
    }
    return output;
}

Tensor execute_flatten(const Node& node, const TensorView& input,
                       const OutputAllocator& allocator) {
    const std::vector<size_t> axis_attribute = integer_list_attribute(node, "axis", {1});
    require(axis_attribute.size() == 1, "invalid Flatten axis");
    const size_t rank = input.shape.size();
    require(axis_attribute[0] <= rank, "Flatten axis is out of range");
    const size_t axis = axis_attribute[0];
    size_t outer = 1;
    size_t inner = 1;
    for (size_t i = 0; i < axis; ++i) {
        outer *= input.shape[i];
    }
    for (size_t i = axis; i < rank; ++i) {
        inner *= input.shape[i];
    }
    Tensor output = allocator.make({outer, inner});
    std::copy(input.data, input.data + output.element_count(), output.data());
    return output;
}

Tensor execute_gemm(const Graph& graph, const Node& node, const TensorView& a,
                    const std::unordered_map<std::string, Tensor>& intermediates,
                    const std::unordered_map<std::string, TensorView>& graph_inputs,
                    const OutputAllocator& allocator) {
    require(a.shape.size() >= 2, "Gemm A must have rank at least 2");
    const TensorView b = tensor_for(graph, intermediates, graph_inputs, node.inputs.at(1));
    require(b.shape.size() == 2, "Gemm B must be rank 2");
    const bool trans_a = integer_list_attribute(node, "transA", {0}).at(0) != 0;
    const bool trans_b = integer_list_attribute(node, "transB", {0}).at(0) != 0;
    require(!trans_a || a.shape.size() == 2, "batched Gemm transA is unsupported");
    const size_t k = trans_a ? a.shape[0] : a.shape.back();
    const size_t m = trans_a ? a.shape[1] : element_count(a.shape) / k;
    const size_t n = trans_b ? b.shape[0] : b.shape[1];
    const size_t b_k = trans_b ? b.shape[1] : b.shape[0];
    require(k == b_k, "Gemm K dimensions do not match");

    std::vector<float> a_matrix(m * k);
    for (size_t row = 0; row < m; ++row) {
        for (size_t column = 0; column < k; ++column) {
            a_matrix[row * k + column] = trans_a
                ? a.data[column * a.shape[1] + row]
                : a.data[row * k + column];
        }
    }
    std::vector<float> b_matrix(k * n);
    for (size_t row = 0; row < k; ++row) {
        for (size_t column = 0; column < n; ++column) {
            b_matrix[row * n + column] = trans_b
                ? b.data[column * b.shape[1] + row]
                : b.data[row * b.shape[1] + column];
        }
    }

    std::vector<size_t> output_shape = a.shape;
    if (trans_a) output_shape = {m, n};
    else output_shape.back() = n;
    Tensor output = allocator.make(output_shape);
    std::fill_n(output.data(), m * n, 0.0f);
    if (node.inputs.size() >= 3 && !node.inputs[2].empty()) {
        const TensorView c = tensor_for(graph, intermediates, graph_inputs, node.inputs[2]);
        if (element_count(c.shape) == 1) {
            std::fill_n(output.data(), m * n, c.data[0]);
        } else if (c.shape.size() == 1 && c.shape[0] == n) {
            for (size_t row = 0; row < m; ++row) {
                std::copy(c.data, c.data + n, output.data() + row * n);
            }
        } else {
            require(c.shape == output.shape, "unsupported Gemm C broadcast shape");
            std::copy(c.data, c.data + m * n, output.data());
        }
    }
    gemm_f32(a_matrix.data(), b_matrix.data(), output.data(), m, k, n,
             float_attribute(node, "alpha", 1.0f), float_attribute(node, "beta", 1.0f));
    return output;
}

Tensor execute_quant_linear(const Graph& graph, const Node& node, const TensorView& a,
                            const std::unordered_map<std::string, Tensor>& intermediates,
                            const std::unordered_map<std::string, TensorView>& graph_inputs,
                            const OutputAllocator& allocator) {
    const bool gemm = node.op_type == "Gemm";
    require(a.shape.size() >= 2,
            "INT8 linear input must be a matrix or batched matrix");
    const std::string& weight_name = node.inputs.at(1);
    const InitializerMeta& meta = graph.initializer_meta(weight_name);
    require(meta.dtype_tag == 1 && meta.shape.size() == 2,
            "INT8 linear weight must be a rank-2 initializer");
    const bool trans_a = gemm && integer_list_attribute(node, "transA", {0}).at(0) != 0;
    const bool trans_b = gemm && integer_list_attribute(node, "transB", {0}).at(0) != 0;
    require(!trans_a || a.shape.size() == 2, "batched INT8 Gemm transA is unsupported");
    const size_t m = trans_a ? a.shape[1] : element_count(a.shape) / a.shape.back();
    const size_t k = trans_a ? a.shape[0] : a.shape.back();
    const size_t n = trans_b ? meta.shape[0] : meta.shape[1];
    const size_t weight_k = trans_b ? meta.shape[1] : meta.shape[0];
    require(k == weight_k, "INT8 linear K dimensions do not match");
    require_channel_scales(node, n, trans_b ? 0 : 1);

    std::vector<int8_t> qa;
    if (trans_a) {
        std::vector<float> transposed(m * k);
        for (size_t row = 0; row < m; ++row) {
            for (size_t col = 0; col < k; ++col) {
                transposed[row * k + col] = a.data[col * m + row];
            }
        }
        qa = quantize_input(transposed.data(), transposed.size(), node.input_scale);
    } else {
        qa = quantize_input(a.data, m * k, node.input_scale);
    }
    const int8_t* weight = graph.initializer_i8_data(weight_name);
    std::vector<int8_t> transposed_weight;
    if (trans_b) {
        transposed_weight.resize(k * n);
        for (size_t row = 0; row < k; ++row) {
            for (size_t col = 0; col < n; ++col) {
                transposed_weight[row * n + col] = weight[col * k + row];
            }
        }
        weight = transposed_weight.data();
    }

    std::vector<size_t> output_shape = a.shape;
    if (trans_a) output_shape = {m, n};
    else output_shape.back() = n;
    Tensor output = allocator.make(output_shape);
    if (k <= static_cast<size_t>(INT32_MAX / (127 * 127))) {
        kernels::gemm_i8_per_channel(qa.data(), weight, nullptr, node.input_scale,
                                     node.weight_scales.data(), output.data(),
                                     m, k, n, kernels::Activation::None);
    } else {
        for (size_t row = 0; row < m; ++row) {
            for (size_t col = 0; col < n; ++col) {
                int64_t sum = 0;
                for (size_t inner = 0; inner < k; ++inner) {
                    sum += static_cast<int32_t>(qa[row * k + inner]) *
                           static_cast<int32_t>(weight[inner * n + col]);
                }
                output.data()[row * n + col] = static_cast<float>(sum) *
                    node.input_scale * node.weight_scales[col];
            }
        }
    }

    const float alpha = gemm ? float_attribute(node, "alpha", 1.0f) : 1.0f;
    const float beta = gemm ? float_attribute(node, "beta", 1.0f) : 0.0f;
    const float* c = nullptr;
    std::vector<size_t> c_shape;
    if (gemm && node.inputs.size() >= 3 && !node.inputs[2].empty()) {
        const TensorView bias = tensor_for(graph, intermediates, graph_inputs, node.inputs[2]);
        c = bias.data;
        c_shape = bias.shape;
        require(element_count(c_shape) == 1 ||
                (c_shape.size() == 1 && c_shape[0] == n) ||
                c_shape == output_shape, "unsupported INT8 Gemm C broadcast shape");
    }
    const std::string activation = string_attribute(node, "activation");
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            float value = alpha * output.data()[row * n + col];
            if (c != nullptr) {
                const size_t bias_index = element_count(c_shape) == 1 ? 0 :
                    c_shape.size() == 1 ? col : row * n + col;
                value += beta * c[bias_index];
            }
            output.data()[row * n + col] = apply_activation(value, activation);
        }
    }
    return output;
}

Tensor execute_matmul(const Graph& graph, const Node& node, const TensorView& a,
                      const std::unordered_map<std::string, Tensor>& intermediates,
                      const std::unordered_map<std::string, TensorView>& graph_inputs,
                      const OutputAllocator& allocator) {
    require(a.shape.size() >= 2, "MatMul input must have rank at least 2");
    const TensorView b = tensor_for(graph, intermediates, graph_inputs, node.inputs.at(1));
    require(b.shape.size() == 2 && a.shape.back() == b.shape[0],
            "MatMul weight shape mismatch");
    const size_t k = a.shape.back(), n = b.shape[1];
    const size_t m = element_count(a.shape) / k;
    std::vector<size_t> shape = a.shape;
    shape.back() = n;
    Tensor output = allocator.make(shape);
    kernels::gemm_f32_optimized(a.data, b.data, nullptr, output.data(), m, k, n);
    return output;
}

}  // namespace

size_t Tensor::element_count() const {
    return leaf::element_count(shape);
}

Tensor Executor::run(const Graph& graph, const Tensor& input) const {
    require(graph.inputs().size() == 1, "only single-input models are supported");
    require(input.arena_data == nullptr && input.element_count() == input.values.size(),
            "input data does not match its shape");
    runtime::Arena* arena = nullptr;
    if (const MemoryPlan* plan = graph.memory_plan()) {
        static thread_local std::unique_ptr<runtime::Arena> cached_arena;
        static thread_local size_t cached_alignment = 0;
        if (cached_arena == nullptr || cached_arena->size() < plan->arena_size ||
            cached_alignment != plan->alignment) {
            cached_arena = std::make_unique<runtime::Arena>(
                static_cast<size_t>(plan->arena_size), plan->alignment);
            cached_alignment = plan->alignment;
        }
        arena = cached_arena.get();
    }
    std::unordered_map<std::string, TensorView> graph_inputs;
    graph_inputs.emplace(graph.inputs()[0], TensorView{input.values.data(), input.shape});

    std::unordered_map<std::string, size_t> remaining_uses;
    for (const Node& node : graph.nodes()) {
        for (const std::string& name : node.inputs) {
            ++remaining_uses[name];
        }
    }
    std::unordered_set<std::string> graph_outputs(graph.outputs().begin(), graph.outputs().end());
    std::unordered_map<std::string, Tensor> intermediates;

    for (const Node& node : graph.nodes()) {
        require(node.outputs.size() == 1, node.op_type + " must have one output");
        const OutputAllocator allocator{graph, arena, node.outputs[0]};
        const TensorView first_input = tensor_for(graph, intermediates, graph_inputs, node.inputs.at(0));
        Tensor output;
        if (node.op_type == "Conv") {
            output = node.quantized
                ? execute_quant_conv(graph, node, first_input, intermediates, graph_inputs, allocator)
                : execute_conv(graph, node, first_input, intermediates, graph_inputs, allocator);
        } else if (node.op_type == "Relu") {
            output = allocator.make(first_input.shape);
            std::copy(first_input.data, first_input.data + output.element_count(), output.data());
            for (size_t i = 0; i < output.element_count(); ++i) {
                output.data()[i] = std::max(output.data()[i], 0.0f);
            }
        } else if (node.op_type == "Identity") {
            output = allocator.make(first_input.shape);
            std::copy(first_input.data, first_input.data + output.element_count(), output.data());
        } else if (node.op_type == "Add") {
            const TensorView second_input = tensor_for(graph, intermediates, graph_inputs, node.inputs.at(1));
            require(first_input.shape == second_input.shape, "Add requires equal input shapes");
            output = allocator.make(first_input.shape);
            for (size_t i = 0; i < output.element_count(); ++i) {
                output.data()[i] = first_input.data[i] + second_input.data[i];
            }
        } else if (node.op_type == "MaxPool") {
            output = execute_maxpool(node, first_input, allocator);
        } else if (node.op_type == "GlobalAveragePool") {
            output = execute_global_average_pool(first_input, allocator);
        } else if (node.op_type == "Flatten") {
            output = execute_flatten(node, first_input, allocator);
        } else if (node.op_type == "Gemm") {
            output = node.quantized
                ? execute_quant_linear(graph, node, first_input, intermediates, graph_inputs, allocator)
                : execute_gemm(graph, node, first_input, intermediates, graph_inputs, allocator);
        } else if (node.op_type == "MatMul") {
            output = node.quantized
                ? execute_quant_linear(graph, node, first_input, intermediates, graph_inputs, allocator)
                : execute_matmul(graph, node, first_input, intermediates, graph_inputs, allocator);
        } else if (node.op_type == "BatchNormalization") {
            require(first_input.shape.size() == 4 && first_input.shape[0] == 1,
                    "BatchNormalization requires one NCHW image");
            const TensorView scale = tensor_for(graph, intermediates, graph_inputs, node.inputs.at(1));
            const TensorView bias = tensor_for(graph, intermediates, graph_inputs, node.inputs.at(2));
            const TensorView mean = tensor_for(graph, intermediates, graph_inputs, node.inputs.at(3));
            const TensorView variance = tensor_for(graph, intermediates, graph_inputs, node.inputs.at(4));
            const size_t channels = first_input.shape[1];
            require(scale.shape == std::vector<size_t>{channels} && bias.shape == scale.shape &&
                    mean.shape == scale.shape && variance.shape == scale.shape,
                    "BatchNormalization parameter shape mismatch");
            output = allocator.make(first_input.shape);
            std::copy(first_input.data, first_input.data + output.element_count(), output.data());
            const size_t spatial = first_input.shape[2] * first_input.shape[3];
            const float epsilon = float_attribute(node, "epsilon", 1e-5f);
            for (size_t channel = 0; channel < channels; ++channel) {
                const float factor = scale.data[channel] / std::sqrt(variance.data[channel] + epsilon);
                for (size_t i = 0; i < spatial; ++i) {
                    float& value = output.data()[channel * spatial + i];
                    value = factor * (value - mean.data[channel]) + bias.data[channel];
                }
            }
        } else {
            throw std::runtime_error("leaf::Executor: unsupported operation: " + node.op_type);
        }
        intermediates[node.outputs[0]] = std::move(output);

        for (const std::string& name : node.inputs) {
            auto use = remaining_uses.find(name);
            if (use != remaining_uses.end() && --use->second == 0 &&
                graph_outputs.find(name) == graph_outputs.end()) {
                auto retiring = intermediates.find(name);
                if (retiring != intermediates.end()) {
                    if (retiring->second.arena_data == nullptr) {
                        release_buffer(std::move(retiring->second.values));
                    }
                    intermediates.erase(retiring);
                }
            }
        }
    }

    require(graph.outputs().size() == 1, "only single-output models are supported");
    const TensorView result = tensor_for(graph, intermediates, graph_inputs, graph.outputs()[0]);
    return {result.shape, std::vector<float>(result.data, result.data + element_count(result.shape))};
}

}  // namespace leaf
