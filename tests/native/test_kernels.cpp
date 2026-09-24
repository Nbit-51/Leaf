#include "leaf/kernels/gemm.h"
#include "leaf/kernels/conv.h"
#include "leaf/kernels/transformer.h"
#include "leaf/runtime/arena.h"
#include "leaf/runtime/kv_cache.h"

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

void test_rmsnorm() {
  for (const std::size_t hidden : {15U, 32U, 896U}) {
    constexpr std::size_t rows = 7;
    std::mt19937 generator(static_cast<unsigned>(hidden));
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
    std::vector<float> input(rows * hidden), weight(hidden), expected(rows * hidden),
        actual(rows * hidden);
    for (float& value : input) value = distribution(generator);
    for (float& value : weight) value = distribution(generator);
    leaf::kernels::rmsnorm_f32_reference(input.data(), weight.data(), expected.data(),
                                         rows, hidden, 1e-6f);
    leaf::kernels::rmsnorm_f32(input.data(), weight.data(), actual.data(),
                               rows, hidden, 1e-6f);
    for (std::size_t index = 0; index < actual.size(); ++index) {
      require(std::abs(actual[index] - expected[index]) < 2e-5f,
              "RMSNorm kernel disagrees with scalar reference");
    }
  }
}

void test_attention() {
  constexpr std::size_t batch = 2, heads = 3, queries = 4, keys = 5, dim = 15;
  const std::size_t mask_shape[4] = {1, 1, queries, keys};
  std::mt19937 generator(112);
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  std::vector<float> query(batch * heads * queries * dim);
  std::vector<float> key(batch * heads * keys * dim), value(key.size());
  std::vector<float> mask(queries * keys, 1.0f);
  std::vector<float> expected(query.size()), actual(query.size());
  for (float& item : query) item = distribution(generator);
  for (float& item : key) item = distribution(generator);
  for (float& item : value) item = distribution(generator);
  for (std::size_t q = 0; q < queries; ++q) {
    for (std::size_t k = q + 1; k < keys; ++k) mask[q * keys + k] = 0.0f;
  }
  leaf::kernels::attention_f32_reference(query.data(), key.data(), value.data(),
                                         mask.data(), expected.data(), batch, heads,
                                         queries, keys, dim, mask_shape, 0.5f);
  leaf::kernels::attention_f32(query.data(), key.data(), value.data(), mask.data(),
                               actual.data(), batch, heads, queries, keys, dim,
                               mask_shape, 0.5f);
  for (std::size_t index = 0; index < actual.size(); ++index) {
    require(std::abs(actual[index] - expected[index]) < 1e-5f,
            "Attention kernel disagrees with scalar reference");
  }
  for (float& item : mask) item = 1.0f - item;
  leaf::kernels::attention_f32(query.data(), key.data(), value.data(), mask.data(),
                               actual.data(), batch, heads, queries, keys, dim,
                               mask_shape, 0.5f, false);
  for (std::size_t index = 0; index < actual.size(); ++index) {
    require(std::abs(actual[index] - expected[index]) < 1e-5f,
            "Attention inverted mask disagrees with scalar reference");
  }
  std::fill(mask.begin(), mask.begin() + keys, 1.0f);
  leaf::kernels::attention_f32(query.data(), key.data(), value.data(), mask.data(),
                               actual.data(), batch, heads, queries, keys, dim,
                               mask_shape, 0.5f, false);
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t h = 0; h < heads; ++h) {
      for (std::size_t d = 0; d < dim; ++d) {
        require(actual[(b * queries * heads + h) * dim + d] == 0.0f,
                "fully masked attention row must return zero");
      }
    }
  }
}

void test_rope_table() {
  constexpr std::size_t batch = 2, half_dim = 7, tokens = 5;
  std::vector<float> frequency(batch * half_dim);
  std::vector<float> position(batch * tokens);
  std::vector<float> cosine(batch * tokens * half_dim * 2);
  std::vector<float> sine(cosine.size());
  for (std::size_t index = 0; index < frequency.size(); ++index) {
    frequency[index] = static_cast<float>(index + 1) * 0.01f;
  }
  for (std::size_t index = 0; index < position.size(); ++index) {
    position[index] = static_cast<float>(index);
  }
  leaf::kernels::rope_table_f32(frequency.data(), position.data(), cosine.data(), sine.data(),
                                 batch, half_dim, tokens, 0.75f, 1.25f);
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t token = 0; token < tokens; ++token) {
      for (std::size_t dimension = 0; dimension < half_dim; ++dimension) {
        const float angle = frequency[b * half_dim + dimension] * position[b * tokens + token];
        const std::size_t index = (b * tokens + token) * half_dim * 2 + dimension;
        require(std::abs(cosine[index] - std::cos(angle) * 0.75f) < 1e-6f,
                "RoPE cosine table mismatch");
        require(std::abs(sine[index] - std::sin(angle) * 1.25f) < 1e-6f,
                "RoPE sine table mismatch");
        require(cosine[index] == cosine[index + half_dim] &&
                sine[index] == sine[index + half_dim],
                "RoPE table halves are not duplicated");
      }
    }
  }
}

void test_repeat_kv() {
  constexpr std::size_t batch = 2, heads = 3, tokens = 4, dim = 5, repeats = 7;
  std::vector<float> input(batch * heads * tokens * dim);
  std::vector<float> output(input.size() * repeats);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(index);
  }
  leaf::kernels::repeat_kv_f32(input.data(), output.data(), batch, heads,
                                tokens, dim, repeats);
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t h = 0; h < heads; ++h) {
      for (std::size_t r = 0; r < repeats; ++r) {
        for (std::size_t item = 0; item < tokens * dim; ++item) {
          const std::size_t source = (b * heads + h) * tokens * dim + item;
          const std::size_t target =
              (b * heads * repeats + h * repeats + r) * tokens * dim + item;
          require(output[target] == input[source], "RepeatKV head ordering mismatch");
        }
      }
    }
  }
}

void test_dynamic_kv_cache() {
  constexpr std::size_t kv_heads = 2, query_heads = 14, dim = 16;
  constexpr std::size_t prefix = 64, total = 65;
  std::mt19937 generator(722);
  std::uniform_real_distribution<float> distribution(-0.3f, 0.3f);
  std::vector<float> prefix_keys(kv_heads * prefix * dim);
  std::vector<float> prefix_values(prefix_keys.size());
  std::vector<float> next_keys(kv_heads * dim), next_values(next_keys.size());
  std::vector<float> query(query_heads * dim);
  for (float& value : prefix_keys) value = distribution(generator);
  for (float& value : prefix_values) value = distribution(generator);
  for (float& value : next_keys) value = distribution(generator);
  for (float& value : next_values) value = distribution(generator);
  for (float& value : query) value = distribution(generator);

  leaf::runtime::KVCache cache(1, kv_heads, dim);
  cache.append(prefix_keys.data(), prefix_values.data(), prefix);
  require(cache.length() == prefix && cache.capacity() >= prefix,
          "KVCache prefix length/capacity mismatch");
  cache.append(next_keys.data(), next_values.data(), 1);
  require(cache.length() == total && cache.capacity() >= total,
          "KVCache failed to grow at capacity boundary");

  std::vector<float> dense_keys(kv_heads * total * dim), dense_values(dense_keys.size());
  for (std::size_t head = 0; head < kv_heads; ++head) {
    std::copy_n(prefix_keys.data() + head * prefix * dim, prefix * dim,
                dense_keys.data() + head * total * dim);
    std::copy_n(prefix_values.data() + head * prefix * dim, prefix * dim,
                dense_values.data() + head * total * dim);
    std::copy_n(next_keys.data() + head * dim, dim,
                dense_keys.data() + (head * total + prefix) * dim);
    std::copy_n(next_values.data() + head * dim, dim,
                dense_values.data() + (head * total + prefix) * dim);
  }
  std::vector<float> repeated_keys(dense_keys.size() * (query_heads / kv_heads));
  std::vector<float> repeated_values(repeated_keys.size());
  leaf::kernels::repeat_kv_f32(dense_keys.data(), repeated_keys.data(), 1,
                                kv_heads, total, dim, query_heads / kv_heads);
  leaf::kernels::repeat_kv_f32(dense_values.data(), repeated_values.data(), 1,
                                kv_heads, total, dim, query_heads / kv_heads);
  const std::size_t mask_shape[4] = {1, 1, 1, 1};
  const float mask = 1.0f;
  std::vector<float> expected(query_heads * dim), actual(expected.size());
  leaf::kernels::attention_f32_reference(query.data(), repeated_keys.data(),
                                         repeated_values.data(), &mask, expected.data(),
                                         1, query_heads, 1, total, dim, mask_shape, 0.5f);
  cache.attend(query.data(), &mask, actual.data(), query_heads, 1, mask_shape, 0.5f);
  for (std::size_t index = 0; index < actual.size(); ++index) {
    require(std::abs(actual[index] - expected[index]) < 1e-5f,
            "cached GQA attention disagrees with dense reference");
  }
  cache.reset();
  require(cache.length() == 0 && cache.capacity() >= total,
          "KVCache reset must clear state without discarding capacity");
}

}  // namespace

int main() {
  try {
    test_f32();
    test_i8();
    test_convolution();
    test_int8_convolution();
    test_rmsnorm();
    test_attention();
    test_rope_table();
    test_repeat_kv();
    test_dynamic_kv_cache();
    test_arena();
    std::cout << "native kernel tests passed (AVX2="
              << (leaf::kernels::compiled_with_avx2() ? "yes" : "no") << ")\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native kernel test failed: " << error.what() << '\n';
    return 1;
  }
}
