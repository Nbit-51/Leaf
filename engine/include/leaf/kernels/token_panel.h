#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

// The 6x16 tile assumes the sixteen vector registers of the x86-64 ABI.
// Other architectures/compilers retain the portable scalar implementation.
#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define LEAF_TOKEN_PANEL_AVX2 1
#define LEAF_TOKEN_PANEL_TARGET __attribute__((target("avx2,fma")))
#else
#define LEAF_TOKEN_PANEL_AVX2 0
#endif

namespace leaf::kernels {
namespace token_panel_detail {
inline void extent(std::size_t rows, std::size_t stride, std::size_t width) {
    const auto limit = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (width > limit || (rows > 1 && stride > (limit - width) / (rows - 1)))
        throw std::overflow_error("token-panel matrix extent overflows address space");
}
}  // namespace token_panel_detail

// Bounded input-only workspace for X[tokens, columns]. Panels store
// X[panel * 16 + lane, column] at [panel, column, lane]; only a final token
// panel is zero padded. No weights are packed, expanded, or retained here.
class TokenPanelsF32 {
public:
    void pack(const float* input, std::size_t tokens, std::size_t columns,
              std::size_t input_token_stride) {
        if (!input || !tokens || !columns || input_token_stride < columns)
            throw std::invalid_argument("token-panel packing requires a nonempty valid input matrix");
        token_panel_detail::extent(tokens, input_token_stride, columns);
        const auto panels = (tokens - 1) / 16 + 1;
        const auto limit = std::numeric_limits<std::size_t>::max();
        if (panels > limit / 16 || columns > limit / (panels * 16) ||
            panels * 16 * columns > values_.max_size())
            throw std::overflow_error("token-panel packing workspace overflows address space");
        values_.resize(panels * 16 * columns);
        tokens_ = tokens; columns_ = columns; panels_ = panels;
        for (std::size_t panel = 0; panel < panels; ++panel) {
            const auto first = panel * 16, count = std::min<std::size_t>(16, tokens - first);
            auto* destination = values_.data() + panel * columns * 16;
            // Small 8x8 transpose blocks bound the active source cache lines.
            // A column-at-a-time gather from 16 power-of-two-strided rows can
            // repeatedly collide in the same L1 cache set.
            for (std::size_t column = 0; column < columns; column += 8) {
                const auto end = std::min(columns, column + 8);
                for (std::size_t lane_block = 0; lane_block < count; lane_block += 8) {
                    const auto lane_end = std::min(count, lane_block + 8);
                    for (std::size_t lane = lane_block; lane < lane_end; ++lane)
                        for (std::size_t col = column; col < end; ++col)
                            destination[col * 16 + lane] = input[(first + lane) * input_token_stride + col];
                }
                for (std::size_t col = column; col < end; ++col)
                    std::fill(destination + col * 16 + count, destination + (col + 1) * 16, 0.0f);
            }
        }
    }

    std::size_t tokens() const noexcept { return tokens_; }
    std::size_t columns() const noexcept { return columns_; }
    std::size_t panels() const noexcept { return panels_; }
    std::size_t storage_size() const noexcept { return values_.size(); }
    const float* panel(std::size_t index) const {
        if (index >= panels_) throw std::out_of_range("token-panel index is outside the packed input");
        return values_.data() + index * columns_ * 16;
    }

private:
    std::size_t tokens_ = 0, columns_ = 0, panels_ = 0;
    std::vector<float> values_;
};

inline bool token_panel_f32_avx2_available() noexcept {
#if LEAF_TOKEN_PANEL_AVX2
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

namespace token_panel_detail {
#if LEAF_TOKEN_PANEL_AVX2
template <unsigned Rows>
LEAF_TOKEN_PANEL_TARGET inline void tile(const float* weights, std::size_t columns,
        std::size_t weight_stride, const float* packed, std::size_t count,
        float* output, std::size_t output_stride, const float* bias) {
    static_assert(Rows >= 1 && Rows <= 6, "FP32 token tile contains one to six rows");
    __m256 a0 = _mm256_setzero_ps(), b0 = a0;
    __m256 a1 = a0, b1 = a0, a2 = a0, b2 = a0;
    __m256 a3 = a0, b3 = a0, a4 = a0, b4 = a0, a5 = a0, b5 = a0;
    // Full-K prototype: twelve accumulators, two input vectors and one reused
    // weight broadcast. K-blocking across row tiles requires a separate
    // partial-output workspace and should be measured as another candidate.
    for (std::size_t col = 0; col < columns; ++col) {
        const auto x0 = _mm256_loadu_ps(packed + col * 16);
        const auto x1 = _mm256_loadu_ps(packed + col * 16 + 8);
        auto weight = _mm256_broadcast_ss(weights + col);
        a0 = _mm256_fmadd_ps(weight, x0, a0); b0 = _mm256_fmadd_ps(weight, x1, b0);
        if constexpr (Rows > 1) {
            weight = _mm256_broadcast_ss(weights + weight_stride + col);
            a1 = _mm256_fmadd_ps(weight, x0, a1); b1 = _mm256_fmadd_ps(weight, x1, b1);
        }
        if constexpr (Rows > 2) {
            weight = _mm256_broadcast_ss(weights + 2 * weight_stride + col);
            a2 = _mm256_fmadd_ps(weight, x0, a2); b2 = _mm256_fmadd_ps(weight, x1, b2);
        }
        if constexpr (Rows > 3) {
            weight = _mm256_broadcast_ss(weights + 3 * weight_stride + col);
            a3 = _mm256_fmadd_ps(weight, x0, a3); b3 = _mm256_fmadd_ps(weight, x1, b3);
        }
        if constexpr (Rows > 4) {
            weight = _mm256_broadcast_ss(weights + 4 * weight_stride + col);
            a4 = _mm256_fmadd_ps(weight, x0, a4); b4 = _mm256_fmadd_ps(weight, x1, b4);
        }
        if constexpr (Rows > 5) {
            weight = _mm256_broadcast_ss(weights + 5 * weight_stride + col);
            a5 = _mm256_fmadd_ps(weight, x0, a5); b5 = _mm256_fmadd_ps(weight, x1, b5);
        }
    }
    alignas(32) float result[Rows][16];
    _mm256_store_ps(result[0], a0); _mm256_store_ps(result[0] + 8, b0);
    if constexpr (Rows > 1) { _mm256_store_ps(result[1], a1); _mm256_store_ps(result[1] + 8, b1); }
    if constexpr (Rows > 2) { _mm256_store_ps(result[2], a2); _mm256_store_ps(result[2] + 8, b2); }
    if constexpr (Rows > 3) { _mm256_store_ps(result[3], a3); _mm256_store_ps(result[3] + 8, b3); }
    if constexpr (Rows > 4) { _mm256_store_ps(result[4], a4); _mm256_store_ps(result[4] + 8, b4); }
    if constexpr (Rows > 5) { _mm256_store_ps(result[5], a5); _mm256_store_ps(result[5] + 8, b5); }
    // Store adjacent output rows together into the existing token-major ABI.
    for (std::size_t token = 0; token < count; ++token)
        for (std::size_t row = 0; row < Rows; ++row)
            output[token * output_stride + row] = result[row][token] + (bias ? bias[row] : 0.0f);
}

// Carry each original FP32 accumulator across increasing K blocks. Do not
// compute separately rounded partial dot products and add them together: that
// would change the full-K FMA reduction order. Scratch rows contain 16 floats
// and are 32-byte aligned; neither packed inputs nor weights are modified.
template <unsigned Rows>
LEAF_TOKEN_PANEL_TARGET inline void accumulate_kblock(const float* weights, std::size_t columns,
        std::size_t weight_stride, const float* packed, float* partial, bool initialize) {
    static_assert(Rows >= 1 && Rows <= 6, "FP32 token tile contains one to six rows");
    __m256 a0 = _mm256_setzero_ps(), b0 = a0;
    __m256 a1 = a0, b1 = a0, a2 = a0, b2 = a0;
    __m256 a3 = a0, b3 = a0, a4 = a0, b4 = a0, a5 = a0, b5 = a0;
    if (!initialize) {
        a0 = _mm256_load_ps(partial); b0 = _mm256_load_ps(partial + 8);
        if constexpr (Rows > 1) { a1 = _mm256_load_ps(partial + 16); b1 = _mm256_load_ps(partial + 24); }
        if constexpr (Rows > 2) { a2 = _mm256_load_ps(partial + 32); b2 = _mm256_load_ps(partial + 40); }
        if constexpr (Rows > 3) { a3 = _mm256_load_ps(partial + 48); b3 = _mm256_load_ps(partial + 56); }
        if constexpr (Rows > 4) { a4 = _mm256_load_ps(partial + 64); b4 = _mm256_load_ps(partial + 72); }
        if constexpr (Rows > 5) { a5 = _mm256_load_ps(partial + 80); b5 = _mm256_load_ps(partial + 88); }
    }
    for (std::size_t col = 0; col < columns; ++col) {
        const auto x0 = _mm256_loadu_ps(packed + col * 16);
        const auto x1 = _mm256_loadu_ps(packed + col * 16 + 8);
        auto weight = _mm256_broadcast_ss(weights + col);
        a0 = _mm256_fmadd_ps(weight, x0, a0); b0 = _mm256_fmadd_ps(weight, x1, b0);
        if constexpr (Rows > 1) {
            weight = _mm256_broadcast_ss(weights + weight_stride + col);
            a1 = _mm256_fmadd_ps(weight, x0, a1); b1 = _mm256_fmadd_ps(weight, x1, b1);
        }
        if constexpr (Rows > 2) {
            weight = _mm256_broadcast_ss(weights + 2 * weight_stride + col);
            a2 = _mm256_fmadd_ps(weight, x0, a2); b2 = _mm256_fmadd_ps(weight, x1, b2);
        }
        if constexpr (Rows > 3) {
            weight = _mm256_broadcast_ss(weights + 3 * weight_stride + col);
            a3 = _mm256_fmadd_ps(weight, x0, a3); b3 = _mm256_fmadd_ps(weight, x1, b3);
        }
        if constexpr (Rows > 4) {
            weight = _mm256_broadcast_ss(weights + 4 * weight_stride + col);
            a4 = _mm256_fmadd_ps(weight, x0, a4); b4 = _mm256_fmadd_ps(weight, x1, b4);
        }
        if constexpr (Rows > 5) {
            weight = _mm256_broadcast_ss(weights + 5 * weight_stride + col);
            a5 = _mm256_fmadd_ps(weight, x0, a5); b5 = _mm256_fmadd_ps(weight, x1, b5);
        }
    }
    _mm256_store_ps(partial, a0); _mm256_store_ps(partial + 8, b0);
    if constexpr (Rows > 1) { _mm256_store_ps(partial + 16, a1); _mm256_store_ps(partial + 24, b1); }
    if constexpr (Rows > 2) { _mm256_store_ps(partial + 32, a2); _mm256_store_ps(partial + 40, b2); }
    if constexpr (Rows > 3) { _mm256_store_ps(partial + 48, a3); _mm256_store_ps(partial + 56, b3); }
    if constexpr (Rows > 4) { _mm256_store_ps(partial + 64, a4); _mm256_store_ps(partial + 72, b4); }
    if constexpr (Rows > 5) { _mm256_store_ps(partial + 80, a5); _mm256_store_ps(partial + 88, b5); }
}

LEAF_TOKEN_PANEL_TARGET inline void gemm_kblocked(const float* weights, std::size_t rows,
        std::size_t columns, std::size_t weight_stride, const TokenPanelsF32& input,
        float* output, std::size_t output_stride, const float* bias) {
    constexpr std::size_t row_block = 96, column_block = 256;
    // Fixed 6 KiB per call, not proportional to the model, sequence, or total
    // output size. Worker row slices use independent scratch and shared const X.
    alignas(32) float partial[row_block][16];
    for (std::size_t panel = 0; panel < input.panels(); ++panel) {
        const auto first = panel * 16, count = std::min<std::size_t>(16, input.tokens() - first);
        const auto* packed = input.panel(panel);
        auto* destination = output + first * output_stride;
        for (std::size_t first_row = 0; first_row < rows;) {
            const auto row_count = std::min(row_block, rows - first_row);
            const auto* row_weights = weights + first_row * weight_stride;
            for (std::size_t first_col = 0; first_col < columns;) {
                const auto column_count = std::min(column_block, columns - first_col);
                const auto* block_input = packed + first_col * 16;
                const auto* block_weights = row_weights + first_col;
                const bool initialize = first_col == 0;
                std::size_t row = 0;
                for (; row_count - row >= 6; row += 6)
                    accumulate_kblock<6>(block_weights + row * weight_stride, column_count,
                        weight_stride, block_input, partial[row], initialize);
                switch (row_count - row) {
                case 5: accumulate_kblock<5>(block_weights + row * weight_stride, column_count,
                            weight_stride, block_input, partial[row], initialize); break;
                case 4: accumulate_kblock<4>(block_weights + row * weight_stride, column_count,
                            weight_stride, block_input, partial[row], initialize); break;
                case 3: accumulate_kblock<3>(block_weights + row * weight_stride, column_count,
                            weight_stride, block_input, partial[row], initialize); break;
                case 2: accumulate_kblock<2>(block_weights + row * weight_stride, column_count,
                            weight_stride, block_input, partial[row], initialize); break;
                case 1: accumulate_kblock<1>(block_weights + row * weight_stride, column_count,
                            weight_stride, block_input, partial[row], initialize); break;
                }
                first_col += column_count;
            }
            // Bias is applied only once; store only real tokens and requested
            // rows, preserving token stride padding and independent row slices.
            for (std::size_t token = 0; token < count; ++token)
                for (std::size_t row = 0; row < row_count; ++row)
                    destination[token * output_stride + first_row + row] = partial[row][token] +
                        (bias ? bias[first_row + row] : 0.0f);
            first_row += row_count;
        }
    }
}
#endif
}  // namespace token_panel_detail

// Y[tokens, rows] = X[tokens, columns] * W[rows, columns]^T + bias[rows].
// Explicit strides permit caller-owned output buffers and independent row
// slices, including later threaded/CNN callers. The caller packs X once and
// may share its immutable panels across row slices. This FP32-only prototype
// intentionally changes reduction order versus feature-lane/horizontal sums;
// it must pass model-quality and packing-inclusive speed gates before use.
inline void gemm_token_panels_f32(const float* weights, std::size_t rows,
        std::size_t columns, std::size_t weight_row_stride, const TokenPanelsF32& input,
        float* output, std::size_t output_token_stride, const float* bias = nullptr,
        bool use_avx2 = true) {
    if (!weights || !output || !rows || !columns || columns != input.columns() || !input.tokens() ||
        weight_row_stride < columns || output_token_stride < rows)
        throw std::invalid_argument("token-panel GEMM requires compatible nonempty strided FP32 matrices");
    token_panel_detail::extent(rows, weight_row_stride, columns);
    token_panel_detail::extent(input.tokens(), output_token_stride, rows);
    const bool vector = use_avx2 && token_panel_f32_avx2_available();
    for (std::size_t panel = 0; panel < input.panels(); ++panel) {
        const auto first = panel * 16, count = std::min<std::size_t>(16, input.tokens() - first);
        const auto* packed = input.panel(panel);
        auto* destination = output + first * output_token_stride;
#if LEAF_TOKEN_PANEL_AVX2
        if (vector) {
            std::size_t row = 0;
            for (; rows - row >= 6; row += 6)
                token_panel_detail::tile<6>(weights + row * weight_row_stride, columns, weight_row_stride,
                    packed, count, destination + row, output_token_stride, bias ? bias + row : nullptr);
            switch (rows - row) {
            case 5: token_panel_detail::tile<5>(weights + row * weight_row_stride, columns, weight_row_stride, packed,
                        count, destination + row, output_token_stride, bias ? bias + row : nullptr); break;
            case 4: token_panel_detail::tile<4>(weights + row * weight_row_stride, columns, weight_row_stride, packed,
                        count, destination + row, output_token_stride, bias ? bias + row : nullptr); break;
            case 3: token_panel_detail::tile<3>(weights + row * weight_row_stride, columns, weight_row_stride, packed,
                        count, destination + row, output_token_stride, bias ? bias + row : nullptr); break;
            case 2: token_panel_detail::tile<2>(weights + row * weight_row_stride, columns, weight_row_stride, packed,
                        count, destination + row, output_token_stride, bias ? bias + row : nullptr); break;
            case 1: token_panel_detail::tile<1>(weights + row * weight_row_stride, columns, weight_row_stride, packed,
                        count, destination + row, output_token_stride, bias ? bias + row : nullptr); break;
            }
            continue;
        }
#else
        (void)vector;
#endif
        for (std::size_t token = 0; token < count; ++token) {
            for (std::size_t row = 0; row < rows; ++row) {
                float result = 0.0f;
                for (std::size_t col = 0; col < columns; ++col)
                    result += weights[row * weight_row_stride + col] * packed[col * 16 + token];
                destination[token * output_token_stride + row] = result + (bias ? bias[row] : 0.0f);
            }
        }
    }
}

// Separate bounded K-blocked candidate; the established full-K API above is
// unchanged. AVX2 keeps the same increasing-column FMA reduction as full-K.
// Small matrices and portable/explicit scalar calls use the original path.
// Inputs and weight storage must not overlap the caller-owned output buffer.
inline void gemm_token_panels_f32_kblocked(const float* weights, std::size_t rows,
        std::size_t columns, std::size_t weight_row_stride, const TokenPanelsF32& input,
        float* output, std::size_t output_token_stride, const float* bias = nullptr,
        bool use_avx2 = true) {
    if (!weights || !output || !rows || !columns || columns != input.columns() || !input.tokens() ||
        weight_row_stride < columns || output_token_stride < rows)
        throw std::invalid_argument("token-panel GEMM requires compatible nonempty strided FP32 matrices");
    token_panel_detail::extent(rows, weight_row_stride, columns);
    token_panel_detail::extent(input.tokens(), output_token_stride, rows);
#if LEAF_TOKEN_PANEL_AVX2
    if (use_avx2 && token_panel_f32_avx2_available() && rows > 6 && columns > 256) {
        token_panel_detail::gemm_kblocked(weights, rows, columns, weight_row_stride, input,
                                         output, output_token_stride, bias);
        return;
    }
#endif
    gemm_token_panels_f32(weights, rows, columns, weight_row_stride, input,
                          output, output_token_stride, bias, use_avx2);
}
}  // namespace leaf::kernels

#undef LEAF_TOKEN_PANEL_AVX2
#if defined(LEAF_TOKEN_PANEL_TARGET)
#undef LEAF_TOKEN_PANEL_TARGET
#endif
