#include "leaf/kernels/gemm.h"
#include "leaf/kernels/conv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

template <class Function>
double median_ms(Function&& function, int iterations) {
  std::vector<double> samples;
  function();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    function();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  bool enforce = false;
  std::string output_path;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--enforce-speedup") {
      enforce = true;
    } else if (argument == "--output" && index + 1 < argc) {
      output_path = argv[++index];
    }
  }
  constexpr std::size_t m = 64, k = 384, n = 384;
  std::mt19937 generator(51);
  std::uniform_real_distribution<float> real_distribution(-1.0F, 1.0F);
  std::uniform_int_distribution<int> int_distribution(-127, 127);
  std::vector<float> a(m * k), b(k * n), bias(n), output(m * n), weight_scales(n, 0.004F);
  std::vector<std::int8_t> aq(m * k), bq(k * n);
  for (float& value : a) value = real_distribution(generator);
  for (float& value : b) value = real_distribution(generator);
  for (float& value : bias) value = real_distribution(generator);
  for (auto& value : aq) value = static_cast<std::int8_t>(int_distribution(generator));
  for (auto& value : bq) value = static_cast<std::int8_t>(int_distribution(generator));

  const double f32_reference = median_ms([&] {
    leaf::kernels::gemm_f32_reference(a.data(), b.data(), bias.data(), output.data(), m, k, n);
  }, 7);
  const double f32_optimized = median_ms([&] {
    leaf::kernels::gemm_f32_optimized(a.data(), b.data(), bias.data(), output.data(), m, k, n);
  }, 11);
  const double i8_reference = median_ms([&] {
    leaf::kernels::gemm_i8_reference(aq.data(), bq.data(), bias.data(), 0.004F,
                                     weight_scales.data(), output.data(), m, k, n);
  }, 7);
  const double i8_optimized = median_ms([&] {
    leaf::kernels::gemm_i8_per_channel(aq.data(), bq.data(), bias.data(), 0.004F,
                                       weight_scales.data(), output.data(), m, k, n);
  }, 11);

  const leaf::kernels::Conv2DShape conv_shape{1, 16, 24, 24, 32, 3, 3, 1, 1, 1, 1};
  const auto conv_rows = conv_shape.batch * conv_shape.output_height() * conv_shape.output_width();
  std::vector<float> conv_input(conv_shape.batch * conv_shape.input_channels *
                                conv_shape.input_height * conv_shape.input_width);
  std::vector<float> conv_weights(conv_shape.output_channels * conv_shape.patch_size());
  std::vector<float> conv_packed(conv_shape.patch_size() * conv_shape.output_channels);
  std::vector<float> conv_output(conv_rows * conv_shape.output_channels);
  std::vector<float> conv_workspace(conv_rows * conv_shape.patch_size());
  std::vector<float> conv_gemm_output(conv_rows * conv_shape.output_channels);
  for (float& value : conv_input) value = real_distribution(generator);
  for (float& value : conv_weights) value = real_distribution(generator);
  leaf::kernels::pack_conv_weights_f32(conv_weights.data(), conv_packed.data(), conv_shape);
  const double conv_reference = median_ms([&] {
    leaf::kernels::conv2d_f32_reference(conv_input.data(), conv_weights.data(), bias.data(),
                                        conv_output.data(), conv_shape);
  }, 3);
  const double conv_optimized = median_ms([&] {
    leaf::kernels::conv2d_f32_im2col(conv_input.data(), conv_packed.data(), bias.data(),
                                     conv_output.data(), conv_workspace.data(),
                                     conv_gemm_output.data(), conv_shape);
  }, 7);

  std::ostringstream report;
  report << std::fixed << std::setprecision(3)
            << "{\n"
#if defined(__GNUC__)
            << "  \"compiler\": \"GCC " << __VERSION__ << "\",\n"
            << "  \"compile_flags\": \"-O3 -mavx2 -mfma\",\n"
#elif defined(_MSC_VER)
            << "  \"compiler\": \"MSVC " << _MSC_VER << "\",\n"
            << "  \"compile_flags\": \"/O2 /arch:AVX2\",\n"
#endif
            << "  \"avx2\": " << (leaf::kernels::compiled_with_avx2() ? "true" : "false") << ",\n"
            << "  \"shape\": [" << m << ", " << k << ", " << n << "],\n"
            << "  \"f32_reference_ms\": " << f32_reference << ",\n"
            << "  \"f32_optimized_ms\": " << f32_optimized << ",\n"
            << "  \"f32_speedup\": " << f32_reference / f32_optimized << ",\n"
            << "  \"int8_reference_ms\": " << i8_reference << ",\n"
            << "  \"int8_optimized_ms\": " << i8_optimized << ",\n"
            << "  \"int8_speedup\": " << i8_reference / i8_optimized << ",\n"
            << "  \"conv_reference_ms\": " << conv_reference << ",\n"
            << "  \"conv_im2col_ms\": " << conv_optimized << ",\n"
            << "  \"conv_speedup\": " << conv_reference / conv_optimized << "\n"
            << "}\n";
  std::cout << report.str();
  if (!output_path.empty()) {
    std::ofstream output_file(output_path);
    if (!output_file) {
      std::cerr << "could not open benchmark output: " << output_path << '\n';
      return 3;
    }
    output_file << report.str();
  }

  if (enforce && (f32_optimized > f32_reference || i8_optimized > i8_reference ||
                  conv_optimized > conv_reference)) {
    std::cerr << "speed gate failed: an optimized kernel regressed its scalar baseline\n";
    return 2;
  }
  return 0;
}
