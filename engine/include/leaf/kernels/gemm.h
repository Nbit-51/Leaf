#pragma once

#include <cstddef>
#include <cstdint>

namespace leaf::kernels {

enum class Activation { None, Relu };

void gemm_f32_reference(const float* a, const float* b, const float* bias,
                        float* output, std::size_t m, std::size_t k,
                        std::size_t n, Activation activation = Activation::None);

void gemm_f32_optimized(const float* a, const float* b, const float* bias,
                        float* output, std::size_t m, std::size_t k,
                        std::size_t n, Activation activation = Activation::None);

void gemm_i8_reference(const std::int8_t* a, const std::int8_t* b,
                       const float* bias, float input_scale,
                       const float* weight_scales, float* output,
                       std::size_t m, std::size_t k, std::size_t n,
                       Activation activation = Activation::None);

void gemm_i8_per_channel(const std::int8_t* a, const std::int8_t* b,
                         const float* bias, float input_scale,
                         const float* weight_scales, float* output,
                         std::size_t m, std::size_t k, std::size_t n,
                         Activation activation = Activation::None);

bool compiled_with_avx2();

}  // namespace leaf::kernels
