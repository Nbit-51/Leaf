#pragma once
#include "leaf/kernels/token_panel.h"
namespace leaf::kernels {
class WideTokenPanelsF32 {
public:
    void pack(const float* input, std::size_t tokens, std::size_t columns, std::size_t stride) {
        if (!input || !tokens || !columns || stride < columns) throw std::invalid_argument("invalid wide token input");
        token_panel_detail::extent(tokens, stride, columns);
        const auto panels = (tokens - 1) / 32 + 1;
        if (panels > values_.max_size() / 32 || columns > values_.max_size() / (panels * 32))
            throw std::overflow_error("wide token workspace overflows");
        values_.resize(panels * 32 * columns);
        tokens_ = tokens; columns_ = columns;
        for (std::size_t p = 0; p < panels; ++p) {
            auto* out = values_.data() + p * 32 * columns;
            const auto count = std::min<std::size_t>(32, tokens - p * 32);
            for (std::size_t k = 0; k < columns; k += 8) {
                const auto end = std::min(columns, k + 8);
                for (std::size_t t = 0; t < count; ++t)
                    for (std::size_t j = k; j < end; ++j) out[j * 32 + t] = input[(p * 32 + t) * stride + j];
                for (std::size_t j = k; j < end; ++j) std::fill(out + j * 32 + count, out + (j + 1) * 32, 0.0f);
            }
        }
    }
    std::size_t tokens() const { return tokens_; }
    std::size_t columns() const { return columns_; }
    const float* data() const { return values_.data(); }
private:
    std::size_t tokens_ = 0, columns_ = 0;
    std::vector<float> values_;
};
#if defined(__x86_64__) && defined(__GNUC__)
namespace wide_token_detail {
template<unsigned Rows>
__attribute__((target("avx2,fma"))) inline void tile(const float* weights, std::size_t columns,
        std::size_t stride, const float* packed, std::size_t count, float* output,
        std::size_t output_stride, const float* bias) {
    __m256 a00 = _mm256_setzero_ps();
    __m256 a01 = _mm256_setzero_ps();
    __m256 a02 = _mm256_setzero_ps();
    __m256 a03 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps();
    __m256 a11 = _mm256_setzero_ps();
    __m256 a12 = _mm256_setzero_ps();
    __m256 a13 = _mm256_setzero_ps();
    __m256 a20 = _mm256_setzero_ps();
    __m256 a21 = _mm256_setzero_ps();
    __m256 a22 = _mm256_setzero_ps();
    __m256 a23 = _mm256_setzero_ps();
    for (std::size_t k = 0; k < columns; ++k) {
        const auto w0 = _mm256_broadcast_ss(weights + k);
        const auto w1 = _mm256_broadcast_ss(weights + (1 < Rows ? 1 : 0) * stride + k);
        const auto w2 = _mm256_broadcast_ss(weights + (2 < Rows ? 2 : 0) * stride + k);
        { const auto x = _mm256_loadu_ps(packed + k * 32 + 0);
          if constexpr (Rows > 0) a00 = _mm256_fmadd_ps(w0, x, a00);
          if constexpr (Rows > 1) a10 = _mm256_fmadd_ps(w1, x, a10);
          if constexpr (Rows > 2) a20 = _mm256_fmadd_ps(w2, x, a20);
        }
        { const auto x = _mm256_loadu_ps(packed + k * 32 + 8);
          if constexpr (Rows > 0) a01 = _mm256_fmadd_ps(w0, x, a01);
          if constexpr (Rows > 1) a11 = _mm256_fmadd_ps(w1, x, a11);
          if constexpr (Rows > 2) a21 = _mm256_fmadd_ps(w2, x, a21);
        }
        { const auto x = _mm256_loadu_ps(packed + k * 32 + 16);
          if constexpr (Rows > 0) a02 = _mm256_fmadd_ps(w0, x, a02);
          if constexpr (Rows > 1) a12 = _mm256_fmadd_ps(w1, x, a12);
          if constexpr (Rows > 2) a22 = _mm256_fmadd_ps(w2, x, a22);
        }
        { const auto x = _mm256_loadu_ps(packed + k * 32 + 24);
          if constexpr (Rows > 0) a03 = _mm256_fmadd_ps(w0, x, a03);
          if constexpr (Rows > 1) a13 = _mm256_fmadd_ps(w1, x, a13);
          if constexpr (Rows > 2) a23 = _mm256_fmadd_ps(w2, x, a23);
        }
    }
    alignas(32) float result[Rows][32];
    if constexpr (Rows > 0) _mm256_store_ps(result[0] + 0, a00);
    if constexpr (Rows > 0) _mm256_store_ps(result[0] + 8, a01);
    if constexpr (Rows > 0) _mm256_store_ps(result[0] + 16, a02);
    if constexpr (Rows > 0) _mm256_store_ps(result[0] + 24, a03);
    if constexpr (Rows > 1) _mm256_store_ps(result[1] + 0, a10);
    if constexpr (Rows > 1) _mm256_store_ps(result[1] + 8, a11);
    if constexpr (Rows > 1) _mm256_store_ps(result[1] + 16, a12);
    if constexpr (Rows > 1) _mm256_store_ps(result[1] + 24, a13);
    if constexpr (Rows > 2) _mm256_store_ps(result[2] + 0, a20);
    if constexpr (Rows > 2) _mm256_store_ps(result[2] + 8, a21);
    if constexpr (Rows > 2) _mm256_store_ps(result[2] + 16, a22);
    if constexpr (Rows > 2) _mm256_store_ps(result[2] + 24, a23);
    for (std::size_t t = 0; t < count; ++t)
        for (std::size_t r = 0; r < Rows; ++r)
            output[t * output_stride + r] = result[r][t] + (bias ? bias[r] : 0.0f);
}
}
#endif
inline void gemm_wide_tokens_f32(const float* weights, std::size_t rows, std::size_t columns,
        std::size_t stride, const WideTokenPanelsF32& input, float* output,
        std::size_t output_stride, const float* bias = nullptr, bool use_avx2 = true,
        bool reuse_rows = false) {
    if (!weights || !output || !rows || !columns || columns != input.columns() || !input.tokens() ||
            stride < columns || output_stride < rows) throw std::invalid_argument("invalid wide token GEMM");
    token_panel_detail::extent(rows, stride, columns);
    token_panel_detail::extent(input.tokens(), output_stride, rows);
#if defined(__x86_64__) && defined(__GNUC__)
    if (reuse_rows && use_avx2 && token_panel_f32_avx2_available()) {
        // Three weight rows, reused across every 32-token panel, rather than
        // traversing the entire weight matrix once per panel. Same microtile.
        for (std::size_t row = 0; row < rows; row += 3) {
            const auto row_count = std::min<std::size_t>(3, rows - row);
            for (std::size_t p = 0; p < (input.tokens() - 1) / 32 + 1; ++p) {
                const auto count = std::min<std::size_t>(32, input.tokens() - p * 32);
                const auto* x = input.data() + p * columns * 32;
                const auto* w = weights + row * stride;
                auto* y = output + p * 32 * output_stride + row;
                const auto* b = bias ? bias + row : nullptr;
                switch (row_count) {
                case 1: wide_token_detail::tile<1>(w, columns, stride, x, count, y, output_stride, b); break;
                case 2: wide_token_detail::tile<2>(w, columns, stride, x, count, y, output_stride, b); break;
                case 3: wide_token_detail::tile<3>(w, columns, stride, x, count, y, output_stride, b); break;
                }
            }
        }
        return;
    }
#else
    (void)reuse_rows;
#endif
    for (std::size_t p = 0; p < (input.tokens() - 1) / 32 + 1; ++p) {
        const auto count = std::min<std::size_t>(32, input.tokens() - p * 32);
        const auto* x = input.data() + p * columns * 32;
        auto* y = output + p * 32 * output_stride;
#if defined(__x86_64__) && defined(__GNUC__)
        if (use_avx2 && token_panel_f32_avx2_available()) {
            std::size_t r = 0;
            for (; rows - r >= 3; r += 3)
                wide_token_detail::tile<3>(weights + r * stride, columns, stride, x, count,
                                           y + r, output_stride, bias ? bias + r : nullptr);
            if (rows - r == 2) wide_token_detail::tile<2>(weights + r * stride, columns, stride, x, count,
                                           y + r, output_stride, bias ? bias + r : nullptr);
            if (rows - r == 1) wide_token_detail::tile<1>(weights + r * stride, columns, stride, x, count,
                                           y + r, output_stride, bias ? bias + r : nullptr);
            continue;
        }
#else
        (void)use_avx2;
#endif
        for (std::size_t t = 0; t < count; ++t)
            for (std::size_t r = 0; r < rows; ++r) {
                float sum = 0;
                for (std::size_t k = 0; k < columns; ++k) sum += weights[r * stride + k] * x[k * 32 + t];
                y[t * output_stride + r] = sum + (bias ? bias[r] : 0.0f);
            }
    }
}
}
