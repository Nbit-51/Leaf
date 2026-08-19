#include "gemm.h"

#include <immintrin.h>
#include <cstring>

namespace leaf {

// Naive-but-vectorized GEMM: for each output row, walk K accumulating
// into an 8-wide AVX2 register per 8-column block of N. Not blocked for
// cache (no tiling yet) -- this is the correctness baseline. Tiling/
// blocking for L1/L2 cache reuse is the next optimization pass once this
// is proven correct and benchmarked.
void gemm_f32(
    const float* A, const float* B, float* C,
    size_t M, size_t K, size_t N,
    float alpha, float beta
) {
    // Apply beta*C first (or zero it out if beta == 0), matching ONNX
    // Gemm semantics where beta scales the existing C values before
    // accumulation.
    if (beta == 0.0f) {
        std::memset(C, 0, M * N * sizeof(float));
    } else if (beta != 1.0f) {
        for (size_t i = 0; i < M * N; ++i) {
            C[i] *= beta;
        }
    }

    const __m256 alpha_vec = _mm256_set1_ps(alpha);

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * N;

        size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            __m256 acc = _mm256_setzero_ps();

            for (size_t k = 0; k < K; ++k) {
                __m256 a_broadcast = _mm256_set1_ps(a_row[k]);
                __m256 b_vec = _mm256_loadu_ps(B + k * N + j);
                acc = _mm256_fmadd_ps(a_broadcast, b_vec, acc);
            }

            acc = _mm256_mul_ps(acc, alpha_vec);
            __m256 existing = _mm256_loadu_ps(c_row + j);
            __m256 result = _mm256_add_ps(existing, acc);
            _mm256_storeu_ps(c_row + j, result);
        }

        // Scalar tail for N not divisible by 8.
        for (; j < N; ++j) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += a_row[k] * B[k * N + j];
            }
            c_row[j] += alpha * acc;
        }
    }
}

}  // namespace leaf
