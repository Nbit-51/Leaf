#pragma once
#include "leaf/kernels/row_reuse.h"

namespace leaf::kernels {
#if defined(__x86_64__) && defined(__GNUC__)
namespace mlp_panel_detail {
// Reuse a bounded weight segment across up to four token panels. Carry each
// original accumulator through increasing K; never sum partial dot products.
__attribute__((target("avx2,fma"))) inline void gemm(
        const float* weights, std::size_t rows, std::size_t columns,
        std::size_t weight_stride, const TokenPanelsF32& input, float* output,
        std::size_t output_stride, const float* bias) {
    constexpr std::size_t panel_block = 4, column_block = 128;
    alignas(32) float partial[panel_block][6][16];
    for (std::size_t row = 0; row < rows; row += 6) {
        const auto row_count = std::min<std::size_t>(6, rows - row);
        for (std::size_t first_panel = 0; first_panel < input.panels(); first_panel += panel_block) {
            const auto panels = std::min(panel_block, input.panels() - first_panel);
            for (std::size_t col = 0; col < columns; col += column_block) {
                const auto columns_here = std::min(column_block, columns - col);
                const auto* w = weights + row * weight_stride + col;
                for (std::size_t p = 0; p < panels; ++p) {
                    const auto* x = input.panel(first_panel + p) + col * 16;
                    auto* sums = partial[p][0];
                    switch (row_count) {
                    case 1: token_panel_detail::accumulate_kblock<1>(w, columns_here, weight_stride, x, sums, col == 0); break;
                    case 2: token_panel_detail::accumulate_kblock<2>(w, columns_here, weight_stride, x, sums, col == 0); break;
                    case 3: token_panel_detail::accumulate_kblock<3>(w, columns_here, weight_stride, x, sums, col == 0); break;
                    case 4: token_panel_detail::accumulate_kblock<4>(w, columns_here, weight_stride, x, sums, col == 0); break;
                    case 5: token_panel_detail::accumulate_kblock<5>(w, columns_here, weight_stride, x, sums, col == 0); break;
                    case 6: token_panel_detail::accumulate_kblock<6>(w, columns_here, weight_stride, x, sums, col == 0); break;
                    }
                }
            }
            for (std::size_t p = 0; p < panels; ++p) {
                const auto first_token = (first_panel + p) * 16;
                const auto count = std::min<std::size_t>(16, input.tokens() - first_token);
                for (std::size_t t = 0; t < count; ++t)
                    for (std::size_t r = 0; r < row_count; ++r)
                        output[(first_token + t) * output_stride + row + r] =
                            partial[p][r][t] + (bias ? bias[row + r] : 0.0f);
            }
        }
    }
}
} // namespace mlp_panel_detail
#endif

inline void gemm_token_panels_f32_mlp_panel(const float* weights, std::size_t rows,
        std::size_t columns, std::size_t weight_stride, const TokenPanelsF32& input,
        float* output, std::size_t output_stride, const float* bias = nullptr,
        bool use_avx2 = true) {
    if (!weights || !output || !rows || !columns || columns != input.columns() || !input.tokens() ||
        weight_stride < columns || output_stride < rows)
        throw std::invalid_argument("MLP-panel GEMM requires compatible nonempty strided FP32 matrices");
    token_panel_detail::extent(rows, weight_stride, columns);
    token_panel_detail::extent(input.tokens(), output_stride, rows);
#if defined(__x86_64__) && defined(__GNUC__)
    if (use_avx2 && token_panel_f32_avx2_available() && input.panels() > 1 && columns > 128) {
        mlp_panel_detail::gemm(weights, rows, columns, weight_stride, input, output, output_stride, bias);
        return;
    }
#endif
    gemm_token_panels_f32_row_reuse(weights, rows, columns, weight_stride, input,
                                  output, output_stride, bias, use_avx2);
}
} // namespace leaf::kernels
