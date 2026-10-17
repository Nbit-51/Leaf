#include "leaf/kernels/gelu_erf.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {
std::size_t checked = 0;
void check(const std::vector<float>& input, bool vector = true) {
    auto actual = input;
    leaf::kernels::gelu_erf_inplace(actual.data(), actual.size(), vector);
    for (std::size_t i = 0; i < input.size(); ++i) {
        const float expected = leaf::kernels::gelu_erf_scalar(input[i]);
        if (std::memcmp(&actual[i], &expected, sizeof(float)) &&
                !(std::isnan(actual[i]) && std::isnan(expected))) {
            std::cerr << "Mismatch at " << input[i] << ": " << std::hexfloat
                      << actual[i] << " vs " << expected << '\n';
            throw std::runtime_error("GELU-erf output differs from scalar libm");
        }
    }
    checked += input.size();
}
}

int main() {
    try {
        leaf::kernels::gelu_erf_inplace(nullptr, 0);
        bool rejected = false;
        try { leaf::kernels::gelu_erf_inplace(nullptr, 1); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("Invalid storage was accepted");
        std::vector<float> inputs;
        for (int i = 0; i <= 1000000; ++i) inputs.push_back(-12.0f + i * (24.0f / 1000000));
        std::mt19937 random(15711);
        for (int i = 0; i < 400000; ++i) {
            const std::uint32_t bits = random();
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            inputs.push_back(value);
        }
        for (float boundary : {0x1p-28f, 0.84375f, 1.25f, 6.0f}) {
            const float center = boundary / 0.7071067811865475f;
            for (float sign : {-1.0f, 1.0f}) {
                float lower = sign * center, upper = lower;
                for (int i = 0; i < 256; ++i) {
                    inputs.push_back(lower); inputs.push_back(upper);
                    lower = std::nextafter(lower, -INFINITY);
                    upper = std::nextafter(upper, INFINITY);
                }
            }
        }
        inputs.insert(inputs.end(), {0.0f, -0.0f, INFINITY, -INFINITY,
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min()});
        check(inputs);
        check(inputs, false);
        // Different batch lengths, unaligned pointers and protected storage.
        for (std::size_t size = 0; size < 65; ++size) {
            std::vector<float> values(size + 2, 173.0f), expected = values;
            for (std::size_t i = 0; i < size; ++i) values[i + 1] = expected[i + 1] = (static_cast<int>(i) - 17) * 0.09f;
            for (std::size_t i = 0; i < size; ++i) expected[i + 1] = leaf::kernels::gelu_erf_scalar(expected[i + 1]);
            leaf::kernels::gelu_erf_inplace(values.data() + 1, size);
            if (std::memcmp(values.data(), expected.data(), values.size() * sizeof(float)))
                throw std::runtime_error("Tail or storage guard mismatch");
            checked += size;
        }
        const auto rounding = std::fegetround();
        for (int mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
            if (std::fesetround(mode) != 0) throw std::runtime_error("Cannot set rounding mode");
            check({-5.0f,-1.77f,-1.5f,-1.2f,-1.0f,-0.7f,-0.0f,0.0f,0.7f,1.0f,1.2f,1.5f,1.77f,5.0f});
        }
        std::fesetround(rounding);
        std::cout << "{\"passed\":true,\"checked_values\":" << checked
                  << ",\"finite_outputs_bit_exact\":true,\"exceptional_classes_passed\":true,\"vector_available\":"
                  << (leaf::kernels::gelu_erf_vector_available() ? "true" : "false") << "}\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
