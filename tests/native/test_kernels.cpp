#include "leaf/kernels/gemm.h"
#include "leaf/kernels/conv.h"
#include "leaf/runtime/arena.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void test_f32() {
  constexpr std::size_t m = 5, k = 19, n = 23;
  std::mt19937 generator(42);
  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
  std::vector<float> a(m * k), b(k * n), bias(n), expected(m * n), actual(m * n);
  for (float& value : a) value = distribution(generator);
  for (float& value : b) value = distribution(generator);
  for (float& value : bias) value = distribution(generator);
  leaf::kernels::gemm_f32_reference(a.data(), b.data(), bias.data(), expected.data(), m, k, n,
                                    leaf::kernels::Activation::Relu);
  leaf::kernels::gemm_f32_optimized(a.data(), b.data(), bias.data(), actual.data(), m, k, n,
                                    leaf::kernels::Activation::Relu);
  float maximum_error = 0.0F;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    maximum_error = std::max(maximum_error, std::abs(actual[index] - expected[index]));
  }
  require(maximum_error < 1e-4F, "optimized FP32 GEMM disagrees with reference");
}

void test_i8() {
  constexpr std::size_t m = 4, k = 31;
  for (const std::size_t n : {8U, 23U, 35U, 40U}) {
    std::mt19937 generator(7);
    std::uniform_int_distribution<int> distribution(-127, 127);
    std::vector<std::int8_t> a(m * k), b(k * n);
    std::vector<float> bias(n), scales(n), expected(m * n), actual(m * n);
    for (auto& value : a) value = static_cast<std::int8_t>(distribution(generator));
    for (auto& value : b) value = static_cast<std::int8_t>(distribution(generator));
    for (std::size_t index = 0; index < n; ++index) {
      bias[index] = static_cast<float>(index) * 0.01F;
      scales[index] = 0.002F + static_cast<float>(index) * 0.00001F;
    }
    leaf::kernels::gemm_i8_reference(a.data(), b.data(), bias.data(), 0.003F, scales.data(),
                                     expected.data(), m, k, n, leaf::kernels::Activation::Relu);
    leaf::kernels::gemm_i8_per_channel(a.data(), b.data(), bias.data(), 0.003F, scales.data(),
                                       actual.data(), m, k, n, leaf::kernels::Activation::Relu);
    for (std::size_t index = 0; index < actual.size(); ++index) {
      require(std::abs(actual[index] - expected[index]) < 1e-5F,
              "optimized INT8 GEMM disagrees with reference");
    }
  }
}

void test_arena() {
  leaf::runtime::Arena arena(256);
  auto* first = static_cast<std::byte*>(arena.at(0, 64));
  auto* second = static_cast<std::byte*>(arena.at(64, 128));
  require(second - first == 64, "arena offsets do not match exported plan");
  bool rejected = false;
  try {
    arena.at(32, 16);
  } catch (const std::out_of_range&) {
    rejected = true;
  }
  require(rejected, "arena accepted a misaligned offset");
}

void test_convolution() {
  const leaf::kernels::Conv2DShape shape{1, 3, 7, 7, 5, 3, 3, 1, 1, 1, 1};
  const auto output_size = shape.batch * shape.output_channels *
                           shape.output_height() * shape.output_width();
  const auto rows = shape.batch * shape.output_height() * shape.output_width();
  std::mt19937 generator(19);
  std::uniform_real_distribution<float> distribution(-0.5F, 0.5F);
  std::vector<float> input(shape.batch * shape.input_channels * shape.input_height * shape.input_width);
  std::vector<float> weights(shape.output_channels * shape.patch_size());
  std::vector<float> packed(shape.patch_size() * shape.output_channels), bias(shape.output_channels);
  std::vector<float> expected(output_size), actual(output_size), workspace(rows * shape.patch_size());
  std::vector<float> gemm_output(rows * shape.output_channels);
  for (float& value : input) value = distribution(generator);
  for (float& value : weights) value = distribution(generator);
  for (float& value : bias) value = distribution(generator);
  leaf::kernels::pack_conv_weights_f32(weights.data(), packed.data(), shape);
  leaf::kernels::conv2d_f32_reference(input.data(), weights.data(), bias.data(), expected.data(), shape,
                                      leaf::kernels::Activation::Relu);
  leaf::kernels::conv2d_f32_im2col(input.data(), packed.data(), bias.data(), actual.data(),
                                   workspace.data(), gemm_output.data(), shape,
                                   leaf::kernels::Activation::Relu);
  for (std::size_t index = 0; index < output_size; ++index) {
    require(std::abs(actual[index] - expected[index]) < 1e-4F,
            "im2col convolution disagrees with direct convolution");
  }
}

void test_int8_convolution() {
  const leaf::kernels::Conv2DShape shape{1, 2, 6, 6, 17, 3, 3, 1, 1, 1, 1};
  const auto rows = shape.batch * shape.output_height() * shape.output_width();
  const auto output_size = rows * shape.output_channels;
  std::mt19937 generator(29);
  std::uniform_int_distribution<int> distribution(-20, 20);
  std::vector<std::int8_t> input(shape.batch * shape.input_channels * shape.input_height * shape.input_width);
  std::vector<std::int8_t> weights(shape.output_channels * shape.patch_size());
  std::vector<std::int8_t> packed(shape.patch_size() * shape.output_channels);
  std::vector<std::int8_t> workspace(rows * shape.patch_size());
  std::vector<float> bias(shape.output_channels, 0.1F), scales(shape.output_channels, 0.02F);
  std::vector<float> expected(output_size), actual(output_size), gemm_output(output_size);
  for (auto& value : input) value = static_cast<std::int8_t>(distribution(generator));
  for (auto& value : weights) value = static_cast<std::int8_t>(distribution(generator));
  leaf::kernels::pack_conv_weights_i8(weights.data(), packed.data(), shape);
  leaf::kernels::conv2d_i8_reference(input.data(), weights.data(), bias.data(), 0.03F,
                                     scales.data(), expected.data(), shape,
                                     leaf::kernels::Activation::Relu);
  leaf::kernels::conv2d_i8_im2col(input.data(), packed.data(), bias.data(), 0.03F,
                                  scales.data(), actual.data(), workspace.data(),
                                  gemm_output.data(), shape, leaf::kernels::Activation::Relu);
  for (std::size_t index = 0; index < output_size; ++index) {
    require(std::abs(actual[index] - expected[index]) < 1e-5F,
            "INT8 im2col convolution disagrees with reference");
  }
}

}  // namespace

int main() {
  try {
    test_f32();
    test_i8();
    test_convolution();
    test_int8_convolution();
    test_arena();
    std::cout << "native kernel tests passed (AVX2="
              << (leaf::kernels::compiled_with_avx2() ? "yes" : "no") << ")\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native kernel test failed: " << error.what() << '\n';
    return 1;
  }
}
