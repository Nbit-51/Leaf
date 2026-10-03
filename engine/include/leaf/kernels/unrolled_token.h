#pragma once
#include "leaf/kernels/token_panel.h"
namespace leaf::kernels {
#if defined(__x86_64__) && defined(__GNUC__)
namespace unrolled_token_detail {
template <unsigned Rows>
__attribute__((target("avx2,fma"))) inline void tile(const float* weights, std::size_t columns,
        std::size_t weight_stride, const float* packed, std::size_t count,
        float* output, std::size_t output_stride, const float* bias) {
    static_assert(Rows >= 1 && Rows <= 6, "FP32 token tile contains one to six rows");
    __m256 a0 = _mm256_setzero_ps(), b0 = a0;
    __m256 a1 = a0, b1 = a0, a2 = a0, b2 = a0;
    __m256 a3 = a0, b3 = a0, a4 = a0, b4 = a0, a5 = a0, b5 = a0;
    // Full-K prototype: twelve accumulators, two input vectors and one reused
    // weight broadcast. K-blocking across row tiles requires a separate
    // partial-output workspace and should be measured as another candidate.
    #pragma GCC unroll 4
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

}
#endif
inline void gemm_token_panels_f32_unrolled(const float* weights, std::size_t rows,
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
#if defined(__x86_64__) && defined(__GNUC__)
        if (vector) {
            std::size_t row = 0;
            for (; rows - row >= 6; row += 6)
                unrolled_token_detail::tile<6>(weights + row * weight_row_stride, columns, weight_row_stride,
                    packed, count, destination + row, output_token_stride, bias ? bias + row : nullptr);
            switch (rows - row) {
            case 5: unrolled_token_detail::tile<5>(weights + row * weight_row_stride, columns, weight_row_stride, packed,
                        count, destination + row, output_token_stride, bias ? bias + row : nullptr); break;
            case 4: unrolled_token_detail::tile<4>(weights + row * weight_row_stride, columns, weight_row_stride, packed,
                        count, destination + row, output_token_stride, bias ? bias + row : nullptr); break;
            case 3: unrolled_token_detail::tile<3>(weights + row * weight_row_stride, columns, weight_row_stride, packed,
                        count, destination + row, output_token_stride, bias ? bias + row : nullptr); break;
            case 2: unrolled_token_detail::tile<2>(weights + row * weight_row_stride, columns, weight_row_stride, packed,
                        count, destination + row, output_token_stride, bias ? bias + row : nullptr); break;
            case 1: unrolled_token_detail::tile<1>(weights + row * weight_row_stride, columns, weight_row_stride, packed,
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

}
