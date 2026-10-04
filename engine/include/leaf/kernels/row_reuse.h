#pragma once
#include "leaf/kernels/token_panel.h"

namespace leaf::kernels {
#if defined(__x86_64__) && defined(__GNUC__)
namespace row_reuse_detail {
// Reuse six weight rows across token panels, retaining the established tile
// and increasing-K FMA order. No extra packing or partial-output workspace.
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
            case 1: token_panel_detail::tile<1>(w, columns, weight_stride, packed, count, y, output_stride, b); break;
            case 2: token_panel_detail::tile<2>(w, columns, weight_stride, packed, count, y, output_stride, b); break;
            case 3: token_panel_detail::tile<3>(w, columns, weight_stride, packed, count, y, output_stride, b); break;
            case 4: token_panel_detail::tile<4>(w, columns, weight_stride, packed, count, y, output_stride, b); break;
            case 5: token_panel_detail::tile<5>(w, columns, weight_stride, packed, count, y, output_stride, b); break;
            case 6: token_panel_detail::tile<6>(w, columns, weight_stride, packed, count, y, output_stride, b); break;
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
