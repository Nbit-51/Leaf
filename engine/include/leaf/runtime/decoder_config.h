#pragma once

// Qualified Windows W8A8 SiLU prefill policy. FP32 and INT4 stay unchanged.
// The experiment flag retains its broader, opt-in precision coverage.
#ifndef LEAF_OPTIMIZED_SILU_W8A8
#if defined(_WIN32) && !defined(LEAF_EXPERIMENTAL_SILU_GATE)
#define LEAF_OPTIMIZED_SILU_W8A8 1
#else
#define LEAF_OPTIMIZED_SILU_W8A8 0
#endif
#endif

// The qualified FP32 stack is the Windows release default. Other platforms
// retain their current policy until they receive their own release validation.
// Explicit research builds preserve the historical opt-in dispatch so old
// experiments remain reproducible. A conservative build is always available.
#ifndef LEAF_OPTIMIZED_FP32
#if defined(_WIN32) && !defined(LEAF_EXPERIMENTAL_VECTOR_GELU) && \
    !defined(LEAF_EXPERIMENTAL_ATTENTION_AVX2) && !defined(LEAF_EXPERIMENTAL_ROW_REUSE) && \
    !defined(LEAF_EXPERIMENTAL_GEMV_PAIR) && !defined(LEAF_EXPERIMENTAL_MLP_PACK) && \
    !defined(LEAF_TEST_GEMV)
#define LEAF_OPTIMIZED_FP32 1
#else
#define LEAF_OPTIMIZED_FP32 0
#endif
#endif
