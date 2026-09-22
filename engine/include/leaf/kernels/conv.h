#pragma once

#include "leaf/kernels/gemm.h"

#include <cstddef>
#include <cstdint>

namespace leaf::kernels {

struct Conv2DShape {
  std::size_t batch;
  std::size_t input_channels;
  std::size_t input_height;
  std::size_t input_width;
  std::size_t output_channels;
  std::size_t kernel_height;
  std::size_t kernel_width;
  std::size_t pad_height{0};
  std::size_t pad_width{0};
  std::size_t stride_height{1};
  std::size_t stride_width{1};

  std::size_t output_height() const;
  std::size_t output_width() const;
  std::size_t patch_size() const;
};

void pack_conv_weights_f32(const float* oihw, float* packed_kxn,
                           const Conv2DShape& shape);
void pack_conv_weights_i8(const std::int8_t* oihw, std::int8_t* packed_kxn,
                          const Conv2DShape& shape);

void conv2d_f32_reference(const float* input, const float* weights_oihw,
                          const float* bias, float* output,
                          const Conv2DShape& shape,
                          Activation activation = Activation::None);
void conv2d_f32_im2col(const float* input, const float* packed_weights,
                       const float* bias, float* output, float* workspace,
                       float* gemm_output, const Conv2DShape& shape,
                       Activation activation = Activation::None);

void conv2d_i8_reference(const std::int8_t* input,
                         const std::int8_t* weights_oihw, const float* bias,
                         float input_scale, const float* weight_scales,
                         float* output, const Conv2DShape& shape,
                         Activation activation = Activation::None);
void conv2d_i8_im2col(const std::int8_t* input,
                      const std::int8_t* packed_weights, const float* bias,
                      float input_scale, const float* weight_scales,
                      float* output, std::int8_t* workspace,
                      float* gemm_output, const Conv2DShape& shape,
                      Activation activation = Activation::None);

}  // namespace leaf::kernels
