#pragma once
#include "leaf/kernels/token_panel.h"

// Pack a bounded W[16, K<=256] panel into [K, 16]. Vector lanes span
// output channels, so a 6-token x 16-channel tile writes contiguous output.
// Packing is per call, not a second persistent copy of model weights.
namespace leaf::kernels {
#if defined(__x86_64__) && defined(__GNUC__)
namespace weight_panel_detail {
#define LEAF_WEIGHT_TARGET __attribute__((target("avx2,fma")))
template<unsigned Tokens>
LEAF_WEIGHT_TARGET inline void tile(const float* input, std::size_t input_stride,
        const float* packed, std::size_t columns, std::size_t rows,
        float* output, std::size_t output_stride, const float* bias,
        bool initialize, bool finish) {
    const auto mask0 = _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(rows)),
                                         _mm256_setr_epi32(0,1,2,3,4,5,6,7));
    const auto mask1 = _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(rows)),
                                         _mm256_setr_epi32(8,9,10,11,12,13,14,15));
    __m256 a0 = _mm256_setzero_ps(), b0 = a0;
    __m256 a1 = _mm256_setzero_ps(), b1 = a1;
    __m256 a2 = _mm256_setzero_ps(), b2 = a2;
    __m256 a3 = _mm256_setzero_ps(), b3 = a3;
    __m256 a4 = _mm256_setzero_ps(), b4 = a4;
    __m256 a5 = _mm256_setzero_ps(), b5 = a5;
    if (!initialize) {
        if constexpr (Tokens > 0) {
            a0 = rows == 16 ? _mm256_loadu_ps(output + 0 * output_stride)
                             : _mm256_maskload_ps(output + 0 * output_stride, mask0);
            if (rows > 8) b0 = rows == 16 ? _mm256_loadu_ps(output + 0 * output_stride + 8)
                                            : _mm256_maskload_ps(output + 0 * output_stride + 8, mask1);
        }
        if constexpr (Tokens > 1) {
            a1 = rows == 16 ? _mm256_loadu_ps(output + 1 * output_stride)
                             : _mm256_maskload_ps(output + 1 * output_stride, mask0);
            if (rows > 8) b1 = rows == 16 ? _mm256_loadu_ps(output + 1 * output_stride + 8)
                                            : _mm256_maskload_ps(output + 1 * output_stride + 8, mask1);
        }
        if constexpr (Tokens > 2) {
            a2 = rows == 16 ? _mm256_loadu_ps(output + 2 * output_stride)
                             : _mm256_maskload_ps(output + 2 * output_stride, mask0);
            if (rows > 8) b2 = rows == 16 ? _mm256_loadu_ps(output + 2 * output_stride + 8)
                                            : _mm256_maskload_ps(output + 2 * output_stride + 8, mask1);
        }
        if constexpr (Tokens > 3) {
            a3 = rows == 16 ? _mm256_loadu_ps(output + 3 * output_stride)
                             : _mm256_maskload_ps(output + 3 * output_stride, mask0);
            if (rows > 8) b3 = rows == 16 ? _mm256_loadu_ps(output + 3 * output_stride + 8)
                                            : _mm256_maskload_ps(output + 3 * output_stride + 8, mask1);
        }
        if constexpr (Tokens > 4) {
            a4 = rows == 16 ? _mm256_loadu_ps(output + 4 * output_stride)
                             : _mm256_maskload_ps(output + 4 * output_stride, mask0);
            if (rows > 8) b4 = rows == 16 ? _mm256_loadu_ps(output + 4 * output_stride + 8)
                                            : _mm256_maskload_ps(output + 4 * output_stride + 8, mask1);
        }
        if constexpr (Tokens > 5) {
            a5 = rows == 16 ? _mm256_loadu_ps(output + 5 * output_stride)
                             : _mm256_maskload_ps(output + 5 * output_stride, mask0);
            if (rows > 8) b5 = rows == 16 ? _mm256_loadu_ps(output + 5 * output_stride + 8)
                                            : _mm256_maskload_ps(output + 5 * output_stride + 8, mask1);
        }
    }
    for (std::size_t k = 0; k < columns; ++k) {
        const auto w0 = _mm256_load_ps(packed + k * 16);
        const auto w1 = _mm256_load_ps(packed + k * 16 + 8);
        if constexpr (Tokens > 0) {
            const auto x = _mm256_broadcast_ss(input + 0 * input_stride + k);
            a0 = _mm256_fmadd_ps(x, w0, a0);
            b0 = _mm256_fmadd_ps(x, w1, b0);
        }
        if constexpr (Tokens > 1) {
            const auto x = _mm256_broadcast_ss(input + 1 * input_stride + k);
            a1 = _mm256_fmadd_ps(x, w0, a1);
            b1 = _mm256_fmadd_ps(x, w1, b1);
        }
        if constexpr (Tokens > 2) {
            const auto x = _mm256_broadcast_ss(input + 2 * input_stride + k);
            a2 = _mm256_fmadd_ps(x, w0, a2);
            b2 = _mm256_fmadd_ps(x, w1, b2);
        }
        if constexpr (Tokens > 3) {
            const auto x = _mm256_broadcast_ss(input + 3 * input_stride + k);
            a3 = _mm256_fmadd_ps(x, w0, a3);
            b3 = _mm256_fmadd_ps(x, w1, b3);
        }
        if constexpr (Tokens > 4) {
            const auto x = _mm256_broadcast_ss(input + 4 * input_stride + k);
            a4 = _mm256_fmadd_ps(x, w0, a4);
            b4 = _mm256_fmadd_ps(x, w1, b4);
        }
        if constexpr (Tokens > 5) {
            const auto x = _mm256_broadcast_ss(input + 5 * input_stride + k);
            a5 = _mm256_fmadd_ps(x, w0, a5);
            b5 = _mm256_fmadd_ps(x, w1, b5);
        }
    }
    const auto bias0 = finish && bias ? _mm256_maskload_ps(bias, mask0) : _mm256_setzero_ps();
    const auto bias1 = finish && bias && rows > 8 ? _mm256_maskload_ps(bias + 8, mask1) : _mm256_setzero_ps();
    if constexpr (Tokens > 0) {
        if (finish) { a0 = _mm256_add_ps(a0, bias0); b0 = _mm256_add_ps(b0, bias1); }
        if (rows == 16) {
            _mm256_storeu_ps(output + 0 * output_stride, a0);
            _mm256_storeu_ps(output + 0 * output_stride + 8, b0);
        } else {
            _mm256_maskstore_ps(output + 0 * output_stride, mask0, a0);
            if (rows > 8) _mm256_maskstore_ps(output + 0 * output_stride + 8, mask1, b0);
        }
    }
    if constexpr (Tokens > 1) {
        if (finish) { a1 = _mm256_add_ps(a1, bias0); b1 = _mm256_add_ps(b1, bias1); }
        if (rows == 16) {
            _mm256_storeu_ps(output + 1 * output_stride, a1);
            _mm256_storeu_ps(output + 1 * output_stride + 8, b1);
        } else {
            _mm256_maskstore_ps(output + 1 * output_stride, mask0, a1);
            if (rows > 8) _mm256_maskstore_ps(output + 1 * output_stride + 8, mask1, b1);
        }
    }
    if constexpr (Tokens > 2) {
        if (finish) { a2 = _mm256_add_ps(a2, bias0); b2 = _mm256_add_ps(b2, bias1); }
        if (rows == 16) {
            _mm256_storeu_ps(output + 2 * output_stride, a2);
            _mm256_storeu_ps(output + 2 * output_stride + 8, b2);
        } else {
            _mm256_maskstore_ps(output + 2 * output_stride, mask0, a2);
            if (rows > 8) _mm256_maskstore_ps(output + 2 * output_stride + 8, mask1, b2);
        }
    }
    if constexpr (Tokens > 3) {
        if (finish) { a3 = _mm256_add_ps(a3, bias0); b3 = _mm256_add_ps(b3, bias1); }
        if (rows == 16) {
            _mm256_storeu_ps(output + 3 * output_stride, a3);
            _mm256_storeu_ps(output + 3 * output_stride + 8, b3);
        } else {
            _mm256_maskstore_ps(output + 3 * output_stride, mask0, a3);
            if (rows > 8) _mm256_maskstore_ps(output + 3 * output_stride + 8, mask1, b3);
        }
    }
    if constexpr (Tokens > 4) {
        if (finish) { a4 = _mm256_add_ps(a4, bias0); b4 = _mm256_add_ps(b4, bias1); }
        if (rows == 16) {
            _mm256_storeu_ps(output + 4 * output_stride, a4);
            _mm256_storeu_ps(output + 4 * output_stride + 8, b4);
        } else {
            _mm256_maskstore_ps(output + 4 * output_stride, mask0, a4);
            if (rows > 8) _mm256_maskstore_ps(output + 4 * output_stride + 8, mask1, b4);
        }
    }
    if constexpr (Tokens > 5) {
        if (finish) { a5 = _mm256_add_ps(a5, bias0); b5 = _mm256_add_ps(b5, bias1); }
        if (rows == 16) {
            _mm256_storeu_ps(output + 5 * output_stride, a5);
            _mm256_storeu_ps(output + 5 * output_stride + 8, b5);
        } else {
            _mm256_maskstore_ps(output + 5 * output_stride, mask0, a5);
            if (rows > 8) _mm256_maskstore_ps(output + 5 * output_stride + 8, mask1, b5);
        }
    }
}
LEAF_WEIGHT_TARGET inline void gemm(const float* input, std::size_t tokens,
        std::size_t input_stride, const float* weights, std::size_t rows,
        std::size_t columns, std::size_t weight_stride, float* output,
        std::size_t output_stride, const float* bias) {
    constexpr std::size_t block_k = 256;
    alignas(32) float packed[block_k * 16];
    for (std::size_t n = 0; n < rows; n += 16) {
        const auto nr = std::min<std::size_t>(16, rows - n);
        for (std::size_t k = 0; k < columns; k += block_k) {
            const auto kc = std::min(block_k, columns - k);
            // Block the transpose to avoid many simultaneous strided cache lines.
            for (std::size_t c = 0; c < kc; c += 8) {
                const auto ce = std::min(kc, c + 8);
                for (std::size_t r = 0; r < nr; ++r)
                    for (std::size_t j = c; j < ce; ++j)
                        packed[j * 16 + r] = weights[(n + r) * weight_stride + k + j];
                for (std::size_t j = c; j < ce; ++j)
                    std::fill(packed + j * 16 + nr, packed + (j + 1) * 16, 0.0f);
            }
            const bool initialize = k == 0, finish = k + kc == columns;
            std::size_t m = 0;
            for (; tokens - m >= 6; m += 6)
                tile<6>(input + m * input_stride + k, input_stride, packed, kc, nr,
                        output + m * output_stride + n, output_stride,
                        bias ? bias + n : nullptr, initialize, finish);
            switch (tokens - m) {
            case 1: tile<1>(input + m * input_stride + k, input_stride, packed, kc, nr,
                        output + m * output_stride + n, output_stride,
                        bias ? bias + n : nullptr, initialize, finish); break;
            case 2: tile<2>(input + m * input_stride + k, input_stride, packed, kc, nr,
                        output + m * output_stride + n, output_stride,
                        bias ? bias + n : nullptr, initialize, finish); break;
            case 3: tile<3>(input + m * input_stride + k, input_stride, packed, kc, nr,
                        output + m * output_stride + n, output_stride,
                        bias ? bias + n : nullptr, initialize, finish); break;
            case 4: tile<4>(input + m * input_stride + k, input_stride, packed, kc, nr,
                        output + m * output_stride + n, output_stride,
                        bias ? bias + n : nullptr, initialize, finish); break;
            case 5: tile<5>(input + m * input_stride + k, input_stride, packed, kc, nr,
                        output + m * output_stride + n, output_stride,
                        bias ? bias + n : nullptr, initialize, finish); break;
            }
        }
    }
}
#undef LEAF_WEIGHT_TARGET
} // namespace weight_panel_detail
#endif

// Input and weights must not overlap output. Row slices may execute in
// parallel: each call has independent 16 KiB stack scratch and disjoint output.
// Keep increasing-K FMA order and add bias once, after the final K block.
inline void gemm_weight_panels_f32(const float* input, std::size_t tokens,
        std::size_t input_stride, const float* weights, std::size_t rows,
        std::size_t columns, std::size_t weight_stride, float* output,
        std::size_t output_stride, const float* bias = nullptr, bool use_avx2 = true) {
    if (!input || !weights || !output || !tokens || !rows || !columns ||
            input_stride < columns || weight_stride < columns || output_stride < rows)
        throw std::invalid_argument("weight-panel GEMM requires compatible nonempty matrices");
    token_panel_detail::extent(tokens, input_stride, columns);
    token_panel_detail::extent(rows, weight_stride, columns);
    token_panel_detail::extent(tokens, output_stride, rows);
#if defined(__x86_64__) && defined(__GNUC__)
    if (use_avx2 && token_panel_f32_avx2_available()) {
        weight_panel_detail::gemm(input, tokens, input_stride, weights, rows, columns,
                                  weight_stride, output, output_stride, bias);
        return;
    }
#else
    (void)use_avx2;
#endif
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t n = 0; n < rows; ++n) {
            float sum = 0;
            for (std::size_t k = 0; k < columns; ++k)
                sum += input[t * input_stride + k] * weights[n * weight_stride + k];
            output[t * output_stride + n] = sum + (bias ? bias[n] : 0.0f);
        }
}
} // namespace leaf::kernels
