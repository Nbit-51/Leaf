#include "leaf/kernels/conv.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace leaf::kernels {

std::size_t Conv2DShape::output_height() const {
  return (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
}
std::size_t Conv2DShape::output_width() const {
  return (input_width + 2 * pad_width - kernel_width) / stride_width + 1;
}
std::size_t Conv2DShape::patch_size() const {
  return input_channels * kernel_height * kernel_width;
}

template <typename T>
void pack_weights(const T* source, T* destination, const Conv2DShape& shape) {
  const std::size_t patch = shape.patch_size();
  for (std::size_t oc = 0; oc < shape.output_channels; ++oc) {
    for (std::size_t item = 0; item < patch; ++item) {
      destination[item * shape.output_channels + oc] = source[oc * patch + item];
    }
  }
}

void pack_conv_weights_f32(const float* source, float* destination,
                           const Conv2DShape& shape) {
  pack_weights(source, destination, shape);
}
void pack_conv_weights_i8(const std::int8_t* source, std::int8_t* destination,
                          const Conv2DShape& shape) {
  pack_weights(source, destination, shape);
}

template <typename T>
void im2col(const T* input, T* workspace, const Conv2DShape& shape) {
  const auto oh = shape.output_height();
  const auto ow = shape.output_width();
  std::size_t row = 0;
  for (std::size_t batch = 0; batch < shape.batch; ++batch) {
    for (std::size_t y = 0; y < oh; ++y) {
      for (std::size_t x = 0; x < ow; ++x, ++row) {
        std::size_t column = 0;
        for (std::size_t channel = 0; channel < shape.input_channels; ++channel) {
          for (std::size_t ky = 0; ky < shape.kernel_height; ++ky) {
            for (std::size_t kx = 0; kx < shape.kernel_width; ++kx, ++column) {
              const auto input_y = static_cast<std::ptrdiff_t>(y * shape.stride_height + ky) -
                                   static_cast<std::ptrdiff_t>(shape.pad_height);
              const auto input_x = static_cast<std::ptrdiff_t>(x * shape.stride_width + kx) -
                                   static_cast<std::ptrdiff_t>(shape.pad_width);
              T value{};
              if (input_y >= 0 && input_x >= 0 &&
                  input_y < static_cast<std::ptrdiff_t>(shape.input_height) &&
                  input_x < static_cast<std::ptrdiff_t>(shape.input_width)) {
                const auto index = ((batch * shape.input_channels + channel) * shape.input_height +
                                    static_cast<std::size_t>(input_y)) * shape.input_width +
                                   static_cast<std::size_t>(input_x);
                value = input[index];
              }
              workspace[row * shape.patch_size() + column] = value;
            }
          }
        }
      }
    }
  }
}

void nhwc_to_nchw(const float* source, float* destination, const Conv2DShape& shape) {
  const auto spatial = shape.output_height() * shape.output_width();
  for (std::size_t batch = 0; batch < shape.batch; ++batch) {
    for (std::size_t position = 0; position < spatial; ++position) {
      for (std::size_t channel = 0; channel < shape.output_channels; ++channel) {
        destination[(batch * shape.output_channels + channel) * spatial + position] =
            source[(batch * spatial + position) * shape.output_channels + channel];
      }
    }
  }
}

void conv2d_f32_reference(const float* input, const float* weights,
                          const float* bias, float* output,
                          const Conv2DShape& shape, Activation activation) {
  const auto oh = shape.output_height(), ow = shape.output_width();
  for (std::size_t batch = 0; batch < shape.batch; ++batch) {
    for (std::size_t oc = 0; oc < shape.output_channels; ++oc) {
      for (std::size_t y = 0; y < oh; ++y) {
        for (std::size_t x = 0; x < ow; ++x) {
          float sum = bias == nullptr ? 0.0F : bias[oc];
          for (std::size_t ic = 0; ic < shape.input_channels; ++ic) {
            for (std::size_t ky = 0; ky < shape.kernel_height; ++ky) {
              for (std::size_t kx = 0; kx < shape.kernel_width; ++kx) {
                const auto iy = static_cast<std::ptrdiff_t>(y * shape.stride_height + ky) -
                                static_cast<std::ptrdiff_t>(shape.pad_height);
                const auto ix = static_cast<std::ptrdiff_t>(x * shape.stride_width + kx) -
                                static_cast<std::ptrdiff_t>(shape.pad_width);
                if (iy >= 0 && ix >= 0 && iy < static_cast<std::ptrdiff_t>(shape.input_height) &&
                    ix < static_cast<std::ptrdiff_t>(shape.input_width)) {
                  const auto input_index = ((batch * shape.input_channels + ic) * shape.input_height +
                                            static_cast<std::size_t>(iy)) * shape.input_width +
                                           static_cast<std::size_t>(ix);
                  const auto weight_index = ((oc * shape.input_channels + ic) * shape.kernel_height + ky) *
                                            shape.kernel_width + kx;
                  sum += input[input_index] * weights[weight_index];
                }
              }
            }
          }
          if (activation == Activation::Relu) sum = std::max(sum, 0.0F);
          output[(batch * shape.output_channels + oc) * oh * ow + y * ow + x] = sum;
        }
      }
    }
  }
}

void conv2d_f32_im2col(const float* input, const float* weights,
                       const float* bias, float* output, float* workspace,
                       float* gemm_output, const Conv2DShape& shape,
                       Activation activation) {
  im2col(input, workspace, shape);
  const auto rows = shape.batch * shape.output_height() * shape.output_width();
  gemm_f32_optimized(workspace, weights, bias, gemm_output, rows,
                     shape.patch_size(), shape.output_channels, activation);
  nhwc_to_nchw(gemm_output, output, shape);
}

void conv2d_i8_reference(const std::int8_t* input, const std::int8_t* weights,
                         const float* bias, float input_scale,
                         const float* weight_scales, float* output,
                         const Conv2DShape& shape, Activation activation) {
  const auto patch = shape.patch_size();
  const auto rows = shape.batch * shape.output_height() * shape.output_width();
  std::vector<std::int8_t> workspace(rows * patch), packed(patch * shape.output_channels);
  std::vector<float> temporary(rows * shape.output_channels);
  im2col(input, workspace.data(), shape);
  pack_conv_weights_i8(weights, packed.data(), shape);
  gemm_i8_reference(workspace.data(), packed.data(), bias, input_scale, weight_scales,
                    temporary.data(), rows, patch, shape.output_channels, activation);
  nhwc_to_nchw(temporary.data(), output, shape);
}

void conv2d_i8_im2col(const std::int8_t* input, const std::int8_t* weights,
                      const float* bias, float input_scale,
                      const float* weight_scales, float* output,
                      std::int8_t* workspace, float* gemm_output,
                      const Conv2DShape& shape, Activation activation) {
  im2col(input, workspace, shape);
  const auto rows = shape.batch * shape.output_height() * shape.output_width();
  gemm_i8_per_channel(workspace, weights, bias, input_scale, weight_scales,
                      gemm_output, rows, shape.patch_size(), shape.output_channels, activation);
  nhwc_to_nchw(gemm_output, output, shape);
}

}  // namespace leaf::kernels
