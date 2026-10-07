// Standalone correctness checks, not a latency benchmark or model-quality gate.
#include "leaf/kernels/token_panel.h"
#include "leaf/kernels/weight_panel.h"
#include "leaf/kernels/wide_token.h"
#include "leaf/kernels/unrolled_token.h"
#include "leaf/kernels/row_reuse.h"
#include "leaf/kernels/mlp_panel.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
constexpr float sentinel = 9876.0f;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Exception, class Function>
void rejected(Function action) {
    try { action(); }
    catch (const Exception&) { return; }
    throw std::runtime_error("invalid token-panel arguments were accepted");
}

void check_matrix(std::size_t tokens, std::size_t rows, std::size_t columns, bool with_bias) {
    const auto input_stride = columns + 5, weight_stride = columns + 3, output_stride = rows + 4;
    std::vector<float> input(tokens * input_stride, sentinel), weights(rows * weight_stride, sentinel);
    std::vector<float> bias(rows), output(tokens * output_stride, sentinel), scalar(output.size(), sentinel);
    std::vector<float> blocked(output.size(), sentinel), blocked_scalar(output.size(), sentinel);
    for (std::size_t token = 0; token < tokens; ++token)
        for (std::size_t col = 0; col < columns; ++col)
            input[token * input_stride + col] = static_cast<float>(int((token * 11 + col * 7) % 41) - 20) / 37.0f;
    for (std::size_t row = 0; row < rows; ++row) {
        bias[row] = static_cast<float>(int(row % 7) - 3) / 19.0f;
        for (std::size_t col = 0; col < columns; ++col)
            weights[row * weight_stride + col] = static_cast<float>(int((row * 13 + col * 3) % 29) - 14) / 31.0f;
    }
    const auto saved_input = input, saved_weights = weights;
    leaf::kernels::TokenPanelsF32 panels;
    panels.pack(input.data(), tokens, columns, input_stride);
    leaf::kernels::TokenPanelsF32 vector_panels;
    vector_panels.pack(input.data(), tokens, columns, input_stride, true);
    for (std::size_t p = 0; p < panels.panels(); ++p)
        require(std::memcmp(panels.panel(p), vector_panels.panel(p), columns * 16 * sizeof(float)) == 0,
                "vector pack differs from scalar pack");
    require(panels.storage_size() == ((tokens - 1) / 16 + 1) * columns * 16, "unexpected packed workspace size");
    for (std::size_t panel = 0; panel < panels.panels(); ++panel)
        for (std::size_t col = 0; col < columns; ++col)
            for (std::size_t lane = 0; lane < 16; ++lane) {
                const auto token = panel * 16 + lane;
                const float expected = token < tokens ? input[token * input_stride + col] : 0.0f;
                require(panels.panel(panel)[col * 16 + lane] == expected, "packed token value or padding differs");
            }
    const auto* offset = with_bias ? bias.data() : nullptr;
    leaf::kernels::gemm_token_panels_f32(weights.data(), rows, columns, weight_stride, panels,
                                       output.data(), output_stride, offset, true);
    leaf::kernels::gemm_token_panels_f32(weights.data(), rows, columns, weight_stride, panels,
                                       scalar.data(), output_stride, offset, false);
    leaf::kernels::gemm_token_panels_f32_kblocked(weights.data(), rows, columns, weight_stride, panels,
                                                blocked.data(), output_stride, offset, true);
    leaf::kernels::gemm_token_panels_f32_kblocked(weights.data(), rows, columns, weight_stride, panels,
                                                blocked_scalar.data(), output_stride, offset, false);
    require(std::memcmp(output.data(), blocked.data(), output.size() * sizeof(float)) == 0,
            "K-blocked vector output is not bit-exact against full-K");
    require(std::memcmp(scalar.data(), blocked_scalar.data(), scalar.size() * sizeof(float)) == 0,
            "K-blocked scalar fallback changed the established scalar output");
    leaf::kernels::gemm_weight_panels_f32(input.data(), tokens, input_stride, weights.data(), rows,
        columns, weight_stride, blocked.data(), output_stride, offset, true);
    leaf::kernels::gemm_weight_panels_f32(input.data(), tokens, input_stride, weights.data(), rows,
        columns, weight_stride, blocked_scalar.data(), output_stride, offset, false);
    require(std::memcmp(output.data(), blocked.data(), output.size() * sizeof(float)) == 0,
            "weight-panel vector output is not bit-exact against full-K");
    require(std::memcmp(scalar.data(), blocked_scalar.data(), scalar.size() * sizeof(float)) == 0,
            "weight-panel scalar fallback changed the established scalar output");
    leaf::kernels::WideTokenPanelsF32 wide;
    wide.pack(input.data(), tokens, columns, input_stride);
    leaf::kernels::gemm_wide_tokens_f32(weights.data(), rows, columns, weight_stride, wide,
        blocked.data(), output_stride, offset, true);
    leaf::kernels::gemm_wide_tokens_f32(weights.data(), rows, columns, weight_stride, wide,
        blocked_scalar.data(), output_stride, offset, false);
    require(std::memcmp(output.data(), blocked.data(), output.size() * sizeof(float)) == 0,
            "wide-token output is not bit-exact against full-K");
    require(std::memcmp(scalar.data(), blocked_scalar.data(), scalar.size() * sizeof(float)) == 0,
            "wide-token scalar fallback changed the established scalar output");
    std::fill(blocked.begin(), blocked.end(), sentinel);
    std::fill(blocked_scalar.begin(), blocked_scalar.end(), sentinel);
    leaf::kernels::gemm_wide_tokens_f32(weights.data(), rows, columns, weight_stride, wide,
        blocked.data(), output_stride, offset, true, true);
    leaf::kernels::gemm_wide_tokens_f32(weights.data(), rows, columns, weight_stride, wide,
        blocked_scalar.data(), output_stride, offset, false, true);
    require(std::memcmp(output.data(), blocked.data(), output.size() * sizeof(float)) == 0,
            "wide row-reuse vector output differs");
    require(std::memcmp(scalar.data(), blocked_scalar.data(), scalar.size() * sizeof(float)) == 0,
            "wide row-reuse scalar fallback differs");
    leaf::kernels::gemm_token_panels_f32_unrolled(weights.data(), rows, columns, weight_stride, panels,
        blocked.data(), output_stride, offset, true);
    require(std::memcmp(output.data(), blocked.data(), output.size() * sizeof(float)) == 0,
            "unrolled output is not bit-exact against full-K");
    leaf::kernels::gemm_token_panels_f32_row_reuse(weights.data(), rows, columns, weight_stride, panels,
        blocked.data(), output_stride, offset, true);
    leaf::kernels::gemm_token_panels_f32_row_reuse(weights.data(), rows, columns, weight_stride, panels,
        blocked_scalar.data(), output_stride, offset, false);
    require(std::memcmp(output.data(), blocked.data(), output.size() * sizeof(float)) == 0,
            "row-reuse output is not bit-exact against full-K");
    require(std::memcmp(scalar.data(), blocked_scalar.data(), scalar.size() * sizeof(float)) == 0,
            "row-reuse scalar fallback changed");
    std::fill(blocked.begin(), blocked.end(), sentinel);
    std::fill(blocked_scalar.begin(), blocked_scalar.end(), sentinel);
    leaf::kernels::gemm_token_panels_f32_mlp_panel(weights.data(), rows, columns, weight_stride, panels,
        blocked.data(), output_stride, offset, true);
    leaf::kernels::gemm_token_panels_f32_mlp_panel(weights.data(), rows, columns, weight_stride, panels,
        blocked_scalar.data(), output_stride, offset, false);
    require(std::memcmp(output.data(), blocked.data(), output.size() * sizeof(float)) == 0,
            "MLP-panel output is not bit-exact against full-K");
    require(std::memcmp(scalar.data(), blocked_scalar.data(), scalar.size() * sizeof(float)) == 0,
            "MLP-panel scalar fallback changed");
    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t row = 0; row < rows; ++row) {
            double expected = with_bias ? bias[row] : 0.0;
            for (std::size_t col = 0; col < columns; ++col)
                expected += double(input[token * input_stride + col]) * weights[row * weight_stride + col];
            const auto index = token * output_stride + row;
            const auto tolerance = 2e-5 * (1.0 + std::abs(expected));
            require(std::abs(output[index] - expected) <= tolerance, "vector/fallback FP32 output differs from reference");
            require(std::abs(scalar[index] - expected) <= tolerance, "scalar FP32 output differs from reference");
            require(std::abs(blocked[index] - expected) <= tolerance, "K-blocked FP32 output differs from reference");
            require(std::abs(blocked_scalar[index] - expected) <= tolerance, "K-blocked scalar output differs from reference");
        }
        for (std::size_t row = rows; row < output_stride; ++row)
            require(output[token * output_stride + row] == sentinel && scalar[token * output_stride + row] == sentinel &&
                    blocked[token * output_stride + row] == sentinel && blocked_scalar[token * output_stride + row] == sentinel,
                    "token-panel GEMM overwrote output stride padding");
    }
    require(input == saved_input && weights == saved_weights, "token-panel GEMM modified caller inputs/weights");
    // Repack the same workspace with a one-token chunk; retained values from a
    // previous panel must never leak into the zero-padded lanes.
    panels.pack(input.data(), 1, columns, input_stride);
    require(panels.storage_size() == columns * 16, "repacked workspace logical size follows the old token count");
    for (std::size_t col = 0; col < columns; ++col)
        for (std::size_t lane = 1; lane < 16; ++lane)
            require(panels.panel(0)[col * 16 + lane] == 0.0f, "repacked tail padding contains stale values");
}

void check_row_slices(std::size_t rows, std::size_t columns, std::size_t slice_rows) {
    constexpr std::size_t tokens = 29;
    std::vector<float> input(tokens * columns, 0.25f), weights(rows * columns, 0.5f);
    std::vector<float> bias(rows), output(tokens * rows, sentinel), blocked(output.size(), sentinel);
    for (std::size_t row = 0; row < rows; ++row) bias[row] = static_cast<float>(row) / 16.0f;
    leaf::kernels::TokenPanelsF32 panels;
    panels.pack(input.data(), tokens, columns, columns);
    for (std::size_t first = 0; first < rows; first += slice_rows) {
        const auto count = std::min(slice_rows, rows - first);
        leaf::kernels::gemm_token_panels_f32(weights.data() + first * columns, count, columns, columns,
                                           panels, output.data() + first, rows, bias.data() + first);
        leaf::kernels::gemm_token_panels_f32_kblocked(weights.data() + first * columns, count, columns, columns,
                                                    panels, blocked.data() + first, rows, bias.data() + first);
    }
    require(std::memcmp(output.data(), blocked.data(), output.size() * sizeof(float)) == 0,
            "independent K-blocked output row slice is not bit-exact");
    std::fill(blocked.begin(), blocked.end(), sentinel);
    for (std::size_t first = 0; first < rows; first += slice_rows) {
        const auto count = std::min(slice_rows, rows - first);
        leaf::kernels::gemm_token_panels_f32_mlp_panel(weights.data() + first * columns, count, columns, columns,
                                                    panels, blocked.data() + first, rows, bias.data() + first);
    }
    require(std::memcmp(output.data(), blocked.data(), output.size() * sizeof(float)) == 0,
            "independent MLP-panel output row slice is not bit-exact");
    for (std::size_t token = 0; token < tokens; ++token)
        for (std::size_t row = 0; row < rows; ++row)
            require(blocked[token * rows + row] == static_cast<float>(columns) * 0.125f + bias[row],
                    "independent output row slice differs or applies bias more than once");
}

void check_invalid() {
    leaf::kernels::TokenPanelsF32 panels;
    float value = 1.0f, output = 0.0f;
    rejected<std::invalid_argument>([&] { panels.pack(nullptr, 1, 1, 1); });
    rejected<std::invalid_argument>([&] { panels.pack(&value, 0, 1, 1); });
    rejected<std::invalid_argument>([&] { panels.pack(&value, 1, 0, 1); });
    rejected<std::invalid_argument>([&] { panels.pack(&value, 1, 2, 1); });
    const auto maximum = std::numeric_limits<std::size_t>::max();
    rejected<std::overflow_error>([&] { panels.pack(&value, maximum, 1, 1); });
    rejected<std::overflow_error>([&] { panels.pack(&value, 2, 1, maximum); });
    rejected<std::out_of_range>([&] { panels.panel(0); });
    panels.pack(&value, 1, 1, 1);
    rejected<std::out_of_range>([&] { panels.panel(1); });
    for (auto gemm : {&leaf::kernels::gemm_token_panels_f32, &leaf::kernels::gemm_token_panels_f32_kblocked,
                     &leaf::kernels::gemm_token_panels_f32_unrolled,
                     &leaf::kernels::gemm_token_panels_f32_row_reuse,
                     &leaf::kernels::gemm_token_panels_f32_mlp_panel}) {
        rejected<std::invalid_argument>([&] { gemm(nullptr, 1, 1, 1, panels, &output, 1, nullptr, true); });
        rejected<std::invalid_argument>([&] { gemm(&value, 1, 1, 1, panels, nullptr, 1, nullptr, true); });
        rejected<std::invalid_argument>([&] { gemm(&value, 0, 1, 1, panels, &output, 1, nullptr, true); });
        rejected<std::invalid_argument>([&] { gemm(&value, 1, 0, 1, panels, &output, 1, nullptr, true); });
        rejected<std::invalid_argument>([&] { gemm(&value, 1, 2, 2, panels, &output, 1, nullptr, true); });
        rejected<std::invalid_argument>([&] { gemm(&value, 2, 1, 1, panels, &output, 1, nullptr, true); });
        rejected<std::invalid_argument>([&] { gemm(&value, 1, 1, 0, panels, &output, 1, nullptr, true); });
        rejected<std::overflow_error>([&] { gemm(&value, 2, 1, maximum, panels, &output, 2, nullptr, true); });
        leaf::kernels::TokenPanelsF32 empty;
        rejected<std::invalid_argument>([&] { gemm(&value, 1, 1, 1, empty, &output, 1, nullptr, true); });
        float pair[2] = {1.0f, 2.0f};
        leaf::kernels::TokenPanelsF32 two_tokens;
        two_tokens.pack(pair, 2, 1, 1);
        rejected<std::overflow_error>([&] { gemm(&value, 1, 1, 1, two_tokens, &output, maximum, nullptr, true); });
    }
    leaf::kernels::WideTokenPanelsF32 wide;
    rejected<std::invalid_argument>([&] { wide.pack(nullptr, 1, 1, 1); });
    rejected<std::invalid_argument>([&] { wide.pack(&value, 0, 1, 1); });
    rejected<std::invalid_argument>([&] { wide.pack(&value, 1, 2, 1); });
    rejected<std::overflow_error>([&] { wide.pack(&value, maximum, 1, 1); });
    wide.pack(&value, 1, 1, 1);
    rejected<std::invalid_argument>([&] { leaf::kernels::gemm_wide_tokens_f32(nullptr, 1, 1, 1, wide, &output, 1); });
    rejected<std::invalid_argument>([&] { leaf::kernels::gemm_wide_tokens_f32(&value, 1, 2, 2, wide, &output, 1); });
    rejected<std::overflow_error>([&] { leaf::kernels::gemm_wide_tokens_f32(&value, 2, 1, maximum, wide, &output, 2); });
    rejected<std::invalid_argument>([&] { leaf::kernels::gemm_weight_panels_f32(nullptr, 1, 1, &value, 1, 1, 1, &output, 1); });
    rejected<std::invalid_argument>([&] { leaf::kernels::gemm_weight_panels_f32(&value, 1, 1, &value, 1, 1, 1, nullptr, 1); });
    rejected<std::invalid_argument>([&] { leaf::kernels::gemm_weight_panels_f32(&value, 1, 1, &value, 2, 1, 1, &output, 1); });
    rejected<std::overflow_error>([&] { leaf::kernels::gemm_weight_panels_f32(&value, 2, maximum, &value, 1, 1, 1, &output, 1); });
}
}  // namespace

int main() {
    try {
        for (std::size_t tokens : {1, 6, 7, 8, 12, 15, 16, 17, 29, 31, 32, 33, 127})
            for (std::size_t rows : {1, 2, 5, 6, 7, 11, 12, 13, 16, 17, 31, 32})
                for (std::size_t columns : {1, 7, 8, 15, 30, 64, 257})
                    for (bool bias : {false, true}) check_matrix(tokens, rows, columns, bias);
        check_matrix(29, 13, 2048, true);
        // Exercise row blocks and every microtile tail, including multiple
        // row/K blocks and token tails. All four execution paths are checked.
        for (std::size_t tokens : {8, 17})
            for (std::size_t rows : {95, 96, 97, 98, 99, 100, 101, 102, 103, 191, 192, 193})
                for (std::size_t columns : {257, 513})
                    for (bool bias : {false, true}) check_matrix(tokens, rows, columns, bias);
        for (std::size_t rows : {97, 193})
            for (bool bias : {false, true}) check_matrix(29, rows, 2048, bias);
        check_matrix(33, 101, 256, true);  // short-K full-K fallback
        check_matrix(33, 101, 255, false);
        check_row_slices(13, 30, 5);
        check_row_slices(211, 257, 108);
        check_invalid();
        std::cout << "FP32 token-panel correctness passed; AVX2/FMA available="
                  << leaf::kernels::token_panel_f32_avx2_available() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
