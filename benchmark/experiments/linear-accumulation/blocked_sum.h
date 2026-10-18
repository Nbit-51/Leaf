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
// order within each 128-column chunk. The original call boundary and an
// implementation-only inline variant are retained for comparison.
template <unsigned Rows>
#if defined(LEAF_ACCUMULATION_INLINE_PARTIAL)
__attribute__((target("avx2,fma"), always_inline)) inline void partial(
#else
__attribute__((target("avx2,fma"), noinline)) void partial(
#endif
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

template <unsigned Rows>
__attribute__((target("avx2,fma"), always_inline)) inline void accumulate_chunk(
        const float* weights, std::size_t columns, std::size_t weight_stride,
        const float* packed, double* total) {
    alignas(32) float part[Rows * 16];
    partial<Rows>(weights, columns, weight_stride, packed, part);
    for (unsigned lane = 0; lane < Rows * 16; lane += 4) {
        const auto value = _mm256_cvtps_pd(_mm_load_ps(part + lane));
        _mm256_store_pd(total + lane, _mm256_add_pd(_mm256_load_pd(total + lane), value));
    }
}

// Match the existing bounded row-block size, now with FP64 outer sums.
// A 128-column input panel (8 KiB) is reused across up to sixteen six-row
// tiles. Unlike a full-K six-row traversal, this working set fits in L1.
// Maximum explicit scratch: 12 KiB totals + 384-byte partial tile per worker.
__attribute__((target("avx2,fma"))) inline void gemm_cached(
        const float* weights, std::size_t rows, std::size_t columns,
        std::size_t weight_stride, const leaf::kernels::TokenPanelsF32& input,
        float* output, std::size_t output_stride, const float* bias) {
    constexpr std::size_t row_block = 96, column_block = 128;
    alignas(32) double total[row_block * 16];
    for (std::size_t panel = 0; panel < input.panels(); ++panel) {
        const auto first_token = panel * 16;
        const auto count = std::min<std::size_t>(16, input.tokens() - first_token);
        for (std::size_t first_row = 0; first_row < rows;) {
            const auto live_rows = std::min(row_block, rows - first_row);
            std::fill(total, total + live_rows * 16, 0.0);
            for (std::size_t first_col = 0; first_col < columns;) {
                const auto width = std::min(column_block, columns - first_col);
                const auto* x = input.panel(panel) + first_col * 16;
                const auto* w = weights + first_row * weight_stride + first_col;
                std::size_t row = 0;
                for (; live_rows - row >= 6; row += 6)
                    accumulate_chunk<6>(w + row * weight_stride, width, weight_stride, x, total + row * 16);
                switch (live_rows - row) {
                case 5: accumulate_chunk<5>(w + row * weight_stride, width, weight_stride, x, total + row * 16); break;
                case 4: accumulate_chunk<4>(w + row * weight_stride, width, weight_stride, x, total + row * 16); break;
                case 3: accumulate_chunk<3>(w + row * weight_stride, width, weight_stride, x, total + row * 16); break;
                case 2: accumulate_chunk<2>(w + row * weight_stride, width, weight_stride, x, total + row * 16); break;
                case 1: accumulate_chunk<1>(w + row * weight_stride, width, weight_stride, x, total + row * 16); break;
                }
                first_col += width;
            }
            for (std::size_t token = 0; token < count; ++token)
                for (std::size_t row = 0; row < live_rows; ++row)
                    output[(first_token + token) * output_stride + first_row + row] =
                        static_cast<float>(total[row * 16 + token]) + (bias ? bias[first_row + row] : 0.0f);
            first_row += live_rows;
        }
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
#if defined(LEAF_ACCUMULATION_CACHE_BLOCKED)
        blocked_sum_detail::gemm_cached(weights, rows, columns, weight_stride, input, output, output_stride, bias);
#else
        blocked_sum_detail::gemm(weights, rows, columns, weight_stride, input, output, output_stride, bias);
#endif
        return;
    }
#endif
    leaf::kernels::gemm_token_panels_f32_row_reuse(
        weights, rows, columns, weight_stride, input, output, output_stride, bias, use_avx2);
}
} // namespace leaf::experiments
