#pragma once
#include "leaf/kernels/row_reuse.h"

// Isolated accuracy/performance candidate. Not part of the release dispatch.
namespace leaf::experiments {
inline float rms_square_sum(const float* values, std::size_t width) {
    double total = 0.0;
    for (std::size_t col = 0; col < width; ++col) {
        const float square = values[col] * values[col];
        total += square;
    }
    return static_cast<float>(total);
}

#if defined(__x86_64__) && defined(__GNUC__)
namespace blocked_sum_detail {
// Keep the existing twelve FMA accumulators, six-row reuse and increasing-K
// order within each 128-column chunk. A call boundary keeps outer FP64 sums
// out of the inner register allocation. Scratch is bounded to 1,152 bytes.
template <unsigned Rows>
__attribute__((target("avx2,fma"), noinline)) void partial(
        const float* weights, std::size_t columns, std::size_t weight_stride,
        const float* packed, float* result) {
    leaf::kernels::token_panel_detail::accumulate_kblock<Rows>(
        weights, columns, weight_stride, packed, result, true);
}

template <unsigned Rows>
__attribute__((target("avx2,fma"))) inline void tile(
        const float* weights, std::size_t columns, std::size_t weight_stride,
        const float* packed, std::size_t count, float* output,
        std::size_t output_stride, const float* bias) {
    alignas(32) float part[Rows * 16];
    alignas(32) double total[Rows * 16] = {};
    for (std::size_t first = 0; first < columns;) {
        const auto width = std::min<std::size_t>(128, columns - first);
        partial<Rows>(weights + first, width, weight_stride, packed + first * 16, part);
        for (unsigned lane = 0; lane < Rows * 16; lane += 4) {
            const auto value = _mm256_cvtps_pd(_mm_load_ps(part + lane));
            _mm256_store_pd(total + lane, _mm256_add_pd(_mm256_load_pd(total + lane), value));
        }
        first += width;
    }
    for (std::size_t token = 0; token < count; ++token)
        for (unsigned row = 0; row < Rows; ++row)
            output[token * output_stride + row] = static_cast<float>(total[row * 16 + token])
                + (bias ? bias[row] : 0.0f);
}

__attribute__((target("avx2,fma"))) inline void gemm(
        const float* weights, std::size_t rows, std::size_t columns,
        std::size_t weight_stride, const leaf::kernels::TokenPanelsF32& input,
        float* output, std::size_t output_stride, const float* bias) {
    for (std::size_t row = 0; row < rows;) {
        const auto live = std::min<std::size_t>(6, rows - row);
        for (std::size_t panel = 0; panel < input.panels(); ++panel) {
            const auto first = panel * 16;
            const auto count = std::min<std::size_t>(16, input.tokens() - first);
            const auto* w = weights + row * weight_stride;
            auto* y = output + first * output_stride + row;
            const auto* b = bias ? bias + row : nullptr;
            switch (live) {
            case 1: tile<1>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b); break;
            case 2: tile<2>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b); break;
            case 3: tile<3>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b); break;
            case 4: tile<4>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b); break;
            case 5: tile<5>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b); break;
            case 6: tile<6>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b); break;
            }
        }
        row += live;
    }
}
} // namespace blocked_sum_detail
#endif

inline void gemm_blocked_sum(const float* weights, std::size_t rows,
        std::size_t columns, std::size_t weight_stride,
        const leaf::kernels::TokenPanelsF32& input, float* output,
        std::size_t output_stride, const float* bias = nullptr, bool use_avx2 = true) {
    if (!weights || !output || !rows || !columns || columns != input.columns() ||
        !input.tokens() || weight_stride < columns || output_stride < rows)
        throw std::invalid_argument("blocked-sum GEMM requires compatible nonempty strided matrices");
    leaf::kernels::token_panel_detail::extent(rows, weight_stride, columns);
    leaf::kernels::token_panel_detail::extent(input.tokens(), output_stride, rows);
#if defined(__x86_64__) && defined(__GNUC__)
    if (use_avx2 && leaf::kernels::token_panel_f32_avx2_available()) {
        blocked_sum_detail::gemm(weights, rows, columns, weight_stride, input, output, output_stride, bias);
        return;
    }
#endif
    leaf::kernels::gemm_token_panels_f32_row_reuse(
        weights, rows, columns, weight_stride, input, output, output_stride, bias, use_avx2);
}
} // namespace leaf::experiments
