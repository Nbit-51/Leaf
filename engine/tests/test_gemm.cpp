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

    std::cout << "\nAll GEMM tests passed.\n";
    return 0;
}
