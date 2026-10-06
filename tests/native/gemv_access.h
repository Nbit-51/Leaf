#pragma once
#include <cstddef>

// Linked only when decoder.cpp is compiled with LEAF_TEST_GEMV.
namespace leaf::runtime::testing {
void gemv_f32(const float* weights, const float* input, float* output,
              std::size_t rows, std::size_t columns, unsigned variant);
}
