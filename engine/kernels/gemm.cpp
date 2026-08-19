#include "gemm.h"

#include <immintrin.h>
#include <cstring>

namespace leaf {

// AVX2 GEMM with a 4x8 micro-kernel.  Convolution maps weights to A(M,K)
// and im2col patches to B(K,N); M is the output-channel count and N is the
// spatial extent.  Computing four output channels together lets every B
// vector loaded from im2col feed four FMAs, rather than rereading it once
// per output channel as the earlier 1x8 kernel did.
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

    size_t i = 0;
    for (; i + 4 <= M; i += 4) {
        const float* a0 = A + (i + 0) * K;
        const float* a1 = A + (i + 1) * K;
        const float* a2 = A + (i + 2) * K;
        const float* a3 = A + (i + 3) * K;
        float* c0 = C + (i + 0) * N;
        float* c1 = C + (i + 1) * N;
        float* c2 = C + (i + 2) * N;
        float* c3 = C + (i + 3) * N;

        size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            for (size_t k = 0; k < K; ++k) {
                const __m256 b_vec = _mm256_loadu_ps(B + k * N + j);
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[k]), b_vec, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[k]), b_vec, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[k]), b_vec, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[k]), b_vec, acc3);
            }
            _mm256_storeu_ps(c0 + j, _mm256_add_ps(_mm256_loadu_ps(c0 + j), _mm256_mul_ps(acc0, alpha_vec)));
            _mm256_storeu_ps(c1 + j, _mm256_add_ps(_mm256_loadu_ps(c1 + j), _mm256_mul_ps(acc1, alpha_vec)));
            _mm256_storeu_ps(c2 + j, _mm256_add_ps(_mm256_loadu_ps(c2 + j), _mm256_mul_ps(acc2, alpha_vec)));
            _mm256_storeu_ps(c3 + j, _mm256_add_ps(_mm256_loadu_ps(c3 + j), _mm256_mul_ps(acc3, alpha_vec)));
        }
        for (; j < N; ++j) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                const float b = B[k * N + j];
                acc0 += a0[k] * b;
                acc1 += a1[k] * b;
                acc2 += a2[k] * b;
                acc3 += a3[k] * b;
            }
            c0[j] += alpha * acc0;
            c1[j] += alpha * acc1;
            c2[j] += alpha * acc2;
            c3[j] += alpha * acc3;
        }
    }

    // Remaining one to three rows use the same 1x8 vector path.
    for (; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * N;
        size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (size_t k = 0; k < K; ++k) {
                acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[k]),
                                       _mm256_loadu_ps(B + k * N + j), acc);
            }
            _mm256_storeu_ps(c_row + j, _mm256_add_ps(_mm256_loadu_ps(c_row + j),
                                                       _mm256_mul_ps(acc, alpha_vec)));
        }
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
