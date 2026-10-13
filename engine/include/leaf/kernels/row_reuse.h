#pragma once
#include "leaf/kernels/token_panel.h"

namespace leaf::kernels {
#if defined(__x86_64__) && defined(__GNUC__)
namespace row_reuse_detail {
// Packed panels keep their existing 16-lane stride. For a tail of <=8 tokens,
// only load and accumulate the first vector; each live lane retains its exact
// increasing-K FMA sequence and the same final bias addition.
template <unsigned Rows>
__attribute__((target("avx2,fma"))) inline void tail8(
        const float* weights, std::size_t columns, std::size_t weight_stride,
        const float* packed, std::size_t count, float* output,
        std::size_t output_stride, const float* bias) {
    __m256 sums[Rows];
    for (unsigned row = 0; row < Rows; ++row) sums[row] = _mm256_setzero_ps();
    for (std::size_t col = 0; col < columns; ++col) {
        const auto x = _mm256_loadu_ps(packed + col * 16);
        for (unsigned row = 0; row < Rows; ++row)
            sums[row] = _mm256_fmadd_ps(_mm256_broadcast_ss(weights + row * weight_stride + col), x, sums[row]);
    }
    alignas(32) float result[Rows][8];
    for (unsigned row = 0; row < Rows; ++row) _mm256_store_ps(result[row], sums[row]);
    for (std::size_t token = 0; token < count; ++token)
        for (unsigned row = 0; row < Rows; ++row)
            output[token * output_stride + row] = result[row][token] + (bias ? bias[row] : 0.0f);
}

template <unsigned Rows, bool NarrowTail>
__attribute__((target("avx2,fma"))) inline void tile(
        const float* w, std::size_t columns, std::size_t weight_stride,
        const float* packed, std::size_t count, float* y,
        std::size_t output_stride, const float* b) {
    if constexpr (NarrowTail) {
        if (count <= 8) {
            tail8<Rows>(w, columns, weight_stride, packed, count, y, output_stride, b);
            return;
        }
    }
    token_panel_detail::tile<Rows>(w, columns, weight_stride, packed, count, y, output_stride, b);
}

// Reuse six weight rows across token panels, retaining the established tile
// and increasing-K FMA order. No extra packing or partial-output workspace.
template <bool NarrowTail = false>
__attribute__((target("avx2,fma"))) inline void gemm(
        const float* weights, std::size_t rows, std::size_t columns,
        std::size_t weight_stride, const TokenPanelsF32& input, float* output,
        std::size_t output_stride, const float* bias) {
    for (std::size_t row = 0; row < rows;) {
        const auto count_rows = std::min<std::size_t>(6, rows - row);
        for (std::size_t panel = 0; panel < input.panels(); ++panel) {
            const auto first = panel * 16;
            const auto count = std::min<std::size_t>(16, input.tokens() - first);
            const auto* packed = input.panel(panel);
            const auto* w = weights + row * weight_stride;
            auto* y = output + first * output_stride + row;
            const auto* b = bias ? bias + row : nullptr;
            switch (count_rows) {
            case 1: tile<1, NarrowTail>(w, columns, weight_stride, packed, count, y, output_stride, b); break;
            case 2: tile<2, NarrowTail>(w, columns, weight_stride, packed, count, y, output_stride, b); break;
            case 3: tile<3, NarrowTail>(w, columns, weight_stride, packed, count, y, output_stride, b); break;
            case 4: tile<4, NarrowTail>(w, columns, weight_stride, packed, count, y, output_stride, b); break;
            case 5: tile<5, NarrowTail>(w, columns, weight_stride, packed, count, y, output_stride, b); break;
            case 6: tile<6, NarrowTail>(w, columns, weight_stride, packed, count, y, output_stride, b); break;
            }
        }
        row += count_rows;
    }
}
} // namespace row_reuse_detail
#endif

inline void gemm_token_panels_f32_row_reuse(const float* weights, std::size_t rows,
        std::size_t columns, std::size_t weight_stride, const TokenPanelsF32& input,
        float* output, std::size_t output_stride, const float* bias = nullptr,
        bool use_avx2 = true) {
    if (!weights || !output || !rows || !columns || columns != input.columns() || !input.tokens() ||
        weight_stride < columns || output_stride < rows)
        throw std::invalid_argument("row-reuse GEMM requires compatible nonempty strided FP32 matrices");
    token_panel_detail::extent(rows, weight_stride, columns);
    token_panel_detail::extent(input.tokens(), output_stride, rows);
#if defined(__x86_64__) && defined(__GNUC__)
    if (use_avx2 && token_panel_f32_avx2_available()) {
        row_reuse_detail::gemm(weights, rows, columns, weight_stride, input, output, output_stride, bias);
        return;
    }
#endif
    gemm_token_panels_f32(weights, rows, columns, weight_stride, input, output, output_stride, bias, use_avx2);
}
} // namespace leaf::kernels
