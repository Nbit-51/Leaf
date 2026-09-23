#include "leaf/kernels/gemm.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace leaf::kernels {
namespace {

inline float activate(float value, Activation activation) {
  return activation == Activation::Relu ? std::max(value, 0.0F) : value;
}

void initialize_output(const float* bias, float* output, std::size_t m,
                       std::size_t n) {
  for (std::size_t row = 0; row < m; ++row) {
    if (bias != nullptr) {
      std::memcpy(output + row * n, bias, n * sizeof(float));
    } else {
      std::fill_n(output + row * n, n, 0.0F);
    }
  }
}

}  // namespace

void gemm_f32_reference(const float* a, const float* b, const float* bias,
                        float* output, std::size_t m, std::size_t k,
                        std::size_t n, Activation activation) {
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t column = 0; column < n; ++column) {
      float sum = bias == nullptr ? 0.0F : bias[column];
      for (std::size_t inner = 0; inner < k; ++inner) {
        sum += a[row * k + inner] * b[inner * n + column];
      }
      output[row * n + column] = activate(sum, activation);
    }
  }
}

void gemm_f32_optimized(const float* a, const float* b, const float* bias,
                        float* output, std::size_t m, std::size_t k,
                        std::size_t n, Activation activation) {
  initialize_output(bias, output, m, n);
  for (std::size_t row = 0; row < m; ++row) {
    float* destination = output + row * n;
    for (std::size_t inner = 0; inner < k; ++inner) {
      const float scalar = a[row * k + inner];
      const float* weights = b + inner * n;
      std::size_t column = 0;
#if defined(__AVX2__)
      const __m256 av = _mm256_set1_ps(scalar);
      for (; column + 8 <= n; column += 8) {
        const __m256 bv = _mm256_loadu_ps(weights + column);
        __m256 cv = _mm256_loadu_ps(destination + column);
#if defined(__FMA__)
        cv = _mm256_fmadd_ps(av, bv, cv);
#else
        cv = _mm256_add_ps(cv, _mm256_mul_ps(av, bv));
#endif
        _mm256_storeu_ps(destination + column, cv);
      }
#endif
      for (; column < n; ++column) {
        destination[column] += scalar * weights[column];
      }
    }
    if (activation == Activation::Relu) {
      std::transform(destination, destination + n, destination,
                     [](float value) { return std::max(value, 0.0F); });
    }
  }
}

void gemm_i8_reference(const std::int8_t* a, const std::int8_t* b,
                       const float* bias, float input_scale,
                       const float* weight_scales, float* output,
                       std::size_t m, std::size_t k, std::size_t n,
                       Activation activation) {
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t column = 0; column < n; ++column) {
      std::int32_t accumulator = 0;
      for (std::size_t inner = 0; inner < k; ++inner) {
        accumulator += static_cast<std::int32_t>(a[row * k + inner]) *
                       static_cast<std::int32_t>(b[inner * n + column]);
      }
      float value = static_cast<float>(accumulator) * input_scale *
                    weight_scales[column];
      if (bias != nullptr) value += bias[column];
      output[row * n + column] = activate(value, activation);
    }
  }
}

void gemm_i8_per_channel(const std::int8_t* a, const std::int8_t* b,
                         const float* bias, float input_scale,
                         const float* weight_scales, float* output,
                         std::size_t m, std::size_t k, std::size_t n,
                         Activation activation) {
  for (std::size_t row = 0; row < m; ++row) {
    std::size_t column = 0;
#if defined(__AVX2__)
    for (; column + 16 <= n; column += 16) {
      __m256i sum_lo = _mm256_setzero_si256();
      __m256i sum_hi = _mm256_setzero_si256();
      for (std::size_t inner = 0; inner < k; ++inner) {
        const __m128i packed = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(b + inner * n + column));
        const __m256i weights16 = _mm256_cvtepi8_epi16(packed);
        const __m256i activation16 = _mm256_set1_epi16(
            static_cast<std::int16_t>(a[row * k + inner]));
        const __m256i products16 = _mm256_mullo_epi16(weights16, activation16);
        const __m128i low16 = _mm256_castsi256_si128(products16);
        const __m128i high16 = _mm256_extracti128_si256(products16, 1);
        sum_lo = _mm256_add_epi32(sum_lo, _mm256_cvtepi16_epi32(low16));
        sum_hi = _mm256_add_epi32(sum_hi, _mm256_cvtepi16_epi32(high16));
      }
      alignas(32) std::int32_t accumulators[16];
      _mm256_store_si256(reinterpret_cast<__m256i*>(accumulators), sum_lo);
      _mm256_store_si256(reinterpret_cast<__m256i*>(accumulators + 8), sum_hi);
      for (std::size_t lane = 0; lane < 16; ++lane) {
        float value = static_cast<float>(accumulators[lane]) * input_scale *
                      weight_scales[column + lane];
        if (bias != nullptr) value += bias[column + lane];
        output[row * n + column + lane] = activate(value, activation);
      }
    }
    for (; column + 8 <= n; column += 8) {
      __m256i sum = _mm256_setzero_si256();
      for (std::size_t inner = 0; inner < k; ++inner) {
        const __m128i packed = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(b + inner * n + column));
        const __m128i weights16 = _mm_cvtepi8_epi16(packed);
        const __m128i activation16 = _mm_set1_epi16(
            static_cast<std::int16_t>(a[row * k + inner]));
        const __m128i products16 = _mm_mullo_epi16(weights16, activation16);
        sum = _mm256_add_epi32(sum, _mm256_cvtepi16_epi32(products16));
      }
      alignas(32) std::int32_t accumulators[8];
      _mm256_store_si256(reinterpret_cast<__m256i*>(accumulators), sum);
      for (std::size_t lane = 0; lane < 8; ++lane) {
        float value = static_cast<float>(accumulators[lane]) * input_scale *
                      weight_scales[column + lane];
        if (bias != nullptr) value += bias[column + lane];
        output[row * n + column + lane] = activate(value, activation);
      }
    }
#endif
    for (; column < n; ++column) {
      std::int32_t accumulator = 0;
      for (std::size_t inner = 0; inner < k; ++inner) {
        accumulator += static_cast<std::int32_t>(a[row * k + inner]) *
                       static_cast<std::int32_t>(b[inner * n + column]);
      }
      float value = static_cast<float>(accumulator) * input_scale *
                    weight_scales[column];
      if (bias != nullptr) value += bias[column];
      output[row * n + column] = activate(value, activation);
    }
  }
}

bool compiled_with_avx2() {
#if defined(__AVX2__)
  return true;
#else
  return false;
#endif
}

}  // namespace leaf::kernels
