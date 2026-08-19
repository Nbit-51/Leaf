// Standalone correctness check for gemm_f32 -- no framework dependency
// yet, just asserts. Run directly; nonzero exit = failure.

#include "../kernels/gemm.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

bool close(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) < tol;
}

int main() {
    // Simple known case: 2x3 @ 3x2 = 2x2, alpha=1, beta=0.
    // A = [[1, 2, 3], [4, 5, 6]]
    // B = [[7, 8], [9, 10], [11, 12]]
    // Expected: A@B = [[58, 64], [139, 154]]
    {
        std::vector<float> A = {1, 2, 3, 4, 5, 6};
        std::vector<float> B = {7, 8, 9, 10, 11, 12};
        std::vector<float> C(4, 0.0f);

        leaf::gemm_f32(A.data(), B.data(), C.data(), 2, 3, 2, 1.0f, 0.0f);

        assert(close(C[0], 58.0f));
        assert(close(C[1], 64.0f));
        assert(close(C[2], 139.0f));
        assert(close(C[3], 154.0f));
        std::cout << "PASS: small known-value GEMM\n";
    }

    // Wider N to exercise the AVX2 8-wide path plus scalar tail
    // (N=10 -> one full 8-wide block + 2-element tail).
    {
        size_t M = 1, K = 4, N = 10;
        std::vector<float> A(K, 1.0f);           // row of ones
        std::vector<float> B(K * N, 1.0f);        // all ones
        std::vector<float> C(M * N, 0.0f);

        leaf::gemm_f32(A.data(), B.data(), C.data(), M, K, N, 1.0f, 0.0f);

        // Each output = sum over K of 1*1 = K = 4.
        for (size_t j = 0; j < N; ++j) {
            assert(close(C[j], 4.0f));
        }
        std::cout << "PASS: AVX2 8-wide + scalar tail path (N=10)\n";
    }

    // beta != 0: C should accumulate onto existing values.
    {
        std::vector<float> A = {2.0f};
        std::vector<float> B = {3.0f};
        std::vector<float> C = {10.0f};

        leaf::gemm_f32(A.data(), B.data(), C.data(), 1, 1, 1, 1.0f, 2.0f);
        // Expected: alpha*(A@B) + beta*C = 1*6 + 2*10 = 26
        assert(close(C[0], 26.0f));
        std::cout << "PASS: alpha/beta accumulation\n";
    }

    // General non-square case: exercises the 4x8 micro-kernel, its M-row
    // remainder, its N-column remainder, and non-trivial alpha/beta values.
    {
        const size_t M = 5, K = 13, N = 17;
        std::vector<float> A(M * K);
        std::vector<float> B(K * N);
        std::vector<float> C(M * N);
        std::vector<float> expected(M * N);
        for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>(static_cast<int>((i * 7) % 19) - 9) / 7.0f;
        for (size_t i = 0; i < B.size(); ++i) B[i] = static_cast<float>(static_cast<int>((i * 5) % 23) - 11) / 11.0f;
        for (size_t i = 0; i < C.size(); ++i) C[i] = static_cast<float>(static_cast<int>((i * 3) % 13) - 6) / 13.0f;
        expected = C;

        const float alpha = 0.75f;
        const float beta = -0.5f;
        for (size_t row = 0; row < M; ++row) {
            for (size_t column = 0; column < N; ++column) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += A[row * K + k] * B[k * N + column];
                }
                expected[row * N + column] = alpha * sum + beta * expected[row * N + column];
            }
        }
        leaf::gemm_f32(A.data(), B.data(), C.data(), M, K, N, alpha, beta);
        for (size_t index = 0; index < C.size(); ++index) {
            assert(close(C[index], expected[index], 1e-4f));
        }
        std::cout << "PASS: tiled GEMM general-shape correctness\n";
    }

    std::cout << "\nAll GEMM tests passed.\n";
    return 0;
}
