#include "leaf/runtime/executor.h"

#include "conv2d.h"
#include "gemm.h"
#include "im2col.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
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

TensorView tensor_for(const Graph& graph,
                      const std::unordered_map<std::string, Tensor>& intermediates,
                      const std::unordered_map<std::string, TensorView>& graph_inputs,
                      const std::string& name) {
    const auto intermediate = intermediates.find(name);
    if (intermediate != intermediates.end()) {
        return {intermediate->second.values.data(), intermediate->second.shape};
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

Tensor execute_conv(const Graph& graph, const Node& node, const TensorView& input,
                    const std::unordered_map<std::string, Tensor>& intermediates,
                    const std::unordered_map<std::string, TensorView>& graph_inputs) {
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
    Tensor output{{1, weights.shape[0], out_h, out_w},
                  acquire_buffer(weights.shape[0] * out_h * out_w)};
    conv2d(input.data, input.shape[1], input.shape[2], input.shape[3],
           weights.data, weights.shape[0], weights.shape[2], weights.shape[3], bias,
           pads[0], pads[1], pads[2], pads[3], strides[0], strides[1],
           dilations[0], dilations[1], output.values.data());

    const std::string activation = string_attribute(node, "activation");
    if (activation == "Relu") {
        for (float& value : output.values) {
            value = std::max(value, 0.0f);
        }
    } else {
        require(activation.empty(), "unsupported fused Conv activation: " + activation);
    }
    return output;
}

Tensor execute_maxpool(const Node& node, const TensorView& input) {
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
    Tensor output{{1, input.shape[1], out_h, out_w},
                  acquire_buffer(input.shape[1] * out_h * out_w)};
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
                output.values[(channel * out_h + oh) * out_w + ow] = maximum;
            }
        }
    }
    return output;
}

Tensor execute_global_average_pool(const TensorView& input) {
    require(input.shape.size() == 4 && input.shape[0] == 1,
            "GlobalAveragePool currently supports one NCHW image at a time");
    const size_t spatial = input.shape[2] * input.shape[3];
    Tensor output{{1, input.shape[1], 1, 1}, acquire_buffer(input.shape[1])};
    for (size_t channel = 0; channel < input.shape[1]; ++channel) {
        float sum = 0.0f;
        const float* channel_data = input.data + channel * spatial;
        for (size_t index = 0; index < spatial; ++index) {
            sum += channel_data[index];
        }
        output.values[channel] = sum / static_cast<float>(spatial);
    }
    return output;
}

Tensor execute_flatten(const Node& node, const TensorView& input) {
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
    std::vector<float> buffer = acquire_buffer(element_count(input.shape));
    std::copy(input.data, input.data + buffer.size(), buffer.begin());
    return {{outer, inner}, std::move(buffer)};
}

Tensor execute_gemm(const Graph& graph, const Node& node, const TensorView& a,
                    const std::unordered_map<std::string, Tensor>& intermediates,
                    const std::unordered_map<std::string, TensorView>& graph_inputs) {
    require(a.shape.size() == 2, "Gemm A must be rank 2");
    const TensorView b = tensor_for(graph, intermediates, graph_inputs, node.inputs.at(1));
    require(b.shape.size() == 2, "Gemm B must be rank 2");
    const bool trans_a = integer_list_attribute(node, "transA", {0}).at(0) != 0;
    const bool trans_b = integer_list_attribute(node, "transB", {0}).at(0) != 0;
    const size_t m = trans_a ? a.shape[1] : a.shape[0];
    const size_t k = trans_a ? a.shape[0] : a.shape[1];
    const size_t n = trans_b ? b.shape[0] : b.shape[1];
    const size_t b_k = trans_b ? b.shape[1] : b.shape[0];
    require(k == b_k, "Gemm K dimensions do not match");

    std::vector<float> a_matrix(m * k);
    for (size_t row = 0; row < m; ++row) {
        for (size_t column = 0; column < k; ++column) {
            a_matrix[row * k + column] = trans_a
                ? a.data[column * a.shape[1] + row]
                : a.data[row * a.shape[1] + column];
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

    std::vector<float> out_buffer = acquire_buffer(m * n);
    std::fill(out_buffer.begin(), out_buffer.end(), 0.0f);
    Tensor output{{m, n}, std::move(out_buffer)};
    if (node.inputs.size() >= 3 && !node.inputs[2].empty()) {
        const TensorView c = tensor_for(graph, intermediates, graph_inputs, node.inputs[2]);
        if (element_count(c.shape) == 1) {
            std::fill(output.values.begin(), output.values.end(), c.data[0]);
        } else if (c.shape.size() == 1 && c.shape[0] == n) {
            for (size_t row = 0; row < m; ++row) {
                std::copy(c.data, c.data + n, output.values.begin() + row * n);
            }
        } else {
            require(c.shape == output.shape, "unsupported Gemm C broadcast shape");
            std::copy(c.data, c.data + output.values.size(), output.values.begin());
        }
    }
    gemm_f32(a_matrix.data(), b_matrix.data(), output.values.data(), m, k, n,
             float_attribute(node, "alpha", 1.0f), float_attribute(node, "beta", 1.0f));
    return output;
}

}  // namespace

size_t Tensor::element_count() const {
    return leaf::element_count(shape);
}

Tensor Executor::run(const Graph& graph, const Tensor& input) const {
    require(graph.inputs().size() == 1, "only single-input models are supported");
    require(input.element_count() == input.values.size(), "input data does not match its shape");
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
        const TensorView first_input = tensor_for(graph, intermediates, graph_inputs, node.inputs.at(0));
        Tensor output;
        if (node.op_type == "Conv") {
            output = execute_conv(graph, node, first_input, intermediates, graph_inputs);
        } else if (node.op_type == "Relu") {
            std::vector<float> buffer = acquire_buffer(element_count(first_input.shape));
            std::copy(first_input.data, first_input.data + buffer.size(), buffer.begin());
            for (float& value : buffer) {
                value = std::max(value, 0.0f);
            }
            output = {first_input.shape, std::move(buffer)};
        } else if (node.op_type == "Identity") {
            std::vector<float> buffer = acquire_buffer(element_count(first_input.shape));
            std::copy(first_input.data, first_input.data + buffer.size(), buffer.begin());
            output = {first_input.shape, std::move(buffer)};
        } else if (node.op_type == "Add") {
            const TensorView second_input = tensor_for(graph, intermediates, graph_inputs, node.inputs.at(1));
            require(first_input.shape == second_input.shape, "Add requires equal input shapes");
            std::vector<float> buffer = acquire_buffer(element_count(first_input.shape));
            for (size_t i = 0; i < buffer.size(); ++i) {
                buffer[i] = first_input.data[i] + second_input.data[i];
            }
            output = {first_input.shape, std::move(buffer)};
        } else if (node.op_type == "MaxPool") {
            output = execute_maxpool(node, first_input);
        } else if (node.op_type == "GlobalAveragePool") {
            output = execute_global_average_pool(first_input);
        } else if (node.op_type == "Flatten") {
            output = execute_flatten(node, first_input);
        } else if (node.op_type == "Gemm") {
            output = execute_gemm(graph, node, first_input, intermediates, graph_inputs);
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
            std::vector<float> buffer = acquire_buffer(element_count(first_input.shape));
            std::copy(first_input.data, first_input.data + buffer.size(), buffer.begin());
            const size_t spatial = first_input.shape[2] * first_input.shape[3];
            const float epsilon = float_attribute(node, "epsilon", 1e-5f);
            for (size_t channel = 0; channel < channels; ++channel) {
                const float factor = scale.data[channel] / std::sqrt(variance.data[channel] + epsilon);
                for (size_t i = 0; i < spatial; ++i) {
                    float& value = buffer[channel * spatial + i];
                    value = factor * (value - mean.data[channel]) + bias.data[channel];
                }
            }
            output = {first_input.shape, std::move(buffer)};
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
                    release_buffer(std::move(retiring->second.values));
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
