#pragma once

#include <cstddef>

namespace leaf {

// C = alpha * (A @ B) + beta * C
// A: (M, K) row-major, B: (K, N) row-major, C: (M, N) row-major.
//
// This is the shared compute core for Conv2D (via im2col) and the Gemm
// op (final FC layers). AVX2/FMA accelerated inner loop -- processes 8
// float32 accumulators at once per FMA instruction.
void gemm_f32(
    const float* A, const float* B, float* C,
    size_t M, size_t K, size_t N,
    float alpha, float beta
);

}  // namespace leaf
