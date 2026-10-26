#pragma once
#include "leaf/kernels/token_panel.h"

namespace leaf::kernels {
// Keep FP32 products and output scale semantics; widen only the RMS reduction.
inline float accurate_rms_square_sum(const float* values, std::size_t width) {
    double total = 0.0;
    for (std::size_t col = 0; col < width; ++col) {
        const float square = values[col] * values[col];
        total += square;
    }
    return static_cast<float>(total);
}

#if defined(__x86_64__) && defined(__GNUC__)
namespace accurate_fp32_detail {
template <unsigned Rows>
__attribute__((target("avx2,fma"))) inline void tile(
        const float* weights, std::size_t columns, std::size_t weight_stride,
        const float* packed, std::size_t count, float* output,
        std::size_t output_stride, const float* bias, float* levels) {
    alignas(32) float part[Rows * 16];
    std::size_t blocks = 0;
    unsigned depth = 0;
    for (std::size_t first = 0; first < columns;) {
        const auto width = std::min<std::size_t>(256, columns - first);
        token_panel_detail::accumulate_kblock<Rows>(
            weights + first, width, weight_stride, packed + first * 16, part, true);
        std::size_t occupied = blocks;
        unsigned level = 0;
        // Binary carries combine short FP32 partials in a balanced tree.
        for (; occupied & 1; occupied >>= 1, ++level)
            for (unsigned lane = 0; lane < Rows * 16; lane += 8)
                _mm256_store_ps(part + lane, _mm256_add_ps(
                    _mm256_load_ps(levels + level * 96 + lane), _mm256_load_ps(part + lane)));
        for (unsigned lane = 0; lane < Rows * 16; lane += 8)
            _mm256_store_ps(levels + level * 96 + lane, _mm256_load_ps(part + lane));
        depth = std::max(depth, level + 1);
        ++blocks;
        first += width;
    }
    bool initialized = false;
    for (unsigned level = depth; level-- > 0;) {
        if (!(blocks & (std::size_t(1) << level))) continue;
        for (unsigned lane = 0; lane < Rows * 16; lane += 8) {
            auto value = _mm256_load_ps(levels + level * 96 + lane);
            if (initialized) value = _mm256_add_ps(_mm256_load_ps(part + lane), value);
            _mm256_store_ps(part + lane, value);
        }
        initialized = true;
    }
    for (std::size_t token = 0; token < count; ++token)
        for (unsigned row = 0; row < Rows; ++row)
            output[token * output_stride + row] = part[row * 16 + token] + (bias ? bias[row] : 0.0f);
}

__attribute__((target("avx2,fma"))) inline void gemm(
        const float* weights, std::size_t rows, std::size_t columns,
        std::size_t weight_stride, const TokenPanelsF32& input,
        float* output, std::size_t output_stride, const float* bias) {
    alignas(32) float levels[std::numeric_limits<std::size_t>::digits * 96];
    for (std::size_t row = 0; row < rows;) {
        const auto live = std::min<std::size_t>(6, rows - row);
        for (std::size_t panel = 0; panel < input.panels(); ++panel) {
            const auto first = panel * 16;
            const auto count = std::min<std::size_t>(16, input.tokens() - first);
            const auto* w = weights + row * weight_stride;
            auto* y = output + first * output_stride + row;
            const auto* b = bias ? bias + row : nullptr;
            switch (live) {
            case 1: tile<1>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b, levels); break;
            case 2: tile<2>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b, levels); break;
            case 3: tile<3>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b, levels); break;
            case 4: tile<4>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b, levels); break;
            case 5: tile<5>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b, levels); break;
            case 6: tile<6>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b, levels); break;
            }
        }
        row += live;
    }
}
} // namespace accurate_fp32_detail
#endif

inline void gemm_token_panels_f32_accurate(const float* weights, std::size_t rows,
        std::size_t columns, std::size_t weight_stride, const TokenPanelsF32& input,
        float* output, std::size_t output_stride, const float* bias = nullptr,
        bool use_avx2 = true) {
    if (!weights || !output || !rows || !columns || columns != input.columns() ||
        !input.tokens() || weight_stride < columns || output_stride < rows)
        throw std::invalid_argument("accurate GEMM requires compatible nonempty strided matrices");
    token_panel_detail::extent(rows, weight_stride, columns);
    token_panel_detail::extent(input.tokens(), output_stride, rows);
#if defined(__x86_64__) && defined(__GNUC__)
    if (use_avx2 && token_panel_f32_avx2_available()) {
        accurate_fp32_detail::gemm(weights, rows, columns, weight_stride, input, output, output_stride, bias);
        return;
    }
#else
    (void)use_avx2;
#endif
    for (std::size_t token = 0; token < input.tokens(); ++token)
        for (std::size_t row = 0; row < rows; ++row) {
            double total = 0.0;
            const auto* packed = input.panel(token / 16);
            for (std::size_t col = 0; col < columns; ++col)
                total += static_cast<double>(weights[row * weight_stride + col]) * packed[col * 16 + token % 16];
            output[token * output_stride + row] = static_cast<float>(total) + (bias ? bias[row] : 0.0f);
        }
}
} // namespace leaf::kernels
