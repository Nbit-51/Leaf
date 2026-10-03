#include "leaf/kernels/gelu.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

int main() {
    try {
        float maximum = 0;
        for (std::size_t count : {0, 1, 7, 8, 9, 15, 16, 17, 100001}) {
            std::vector<float> source(count + 2, 9876.0f);
            for (std::size_t i = 0; i < count; ++i)
                source[i + 1] = -20.0f + 40.0f * static_cast<float>(i) / std::max<std::size_t>(1, count - 1);
            auto output = source, scalar = source;
            leaf::kernels::gelu_new_inplace(output.data() + 1, count);
            leaf::kernels::gelu_new_inplace(scalar.data() + 1, count, false);
            if (output.front() != 9876 || output.back() != 9876) throw std::runtime_error("GELU overwrote guards");
            for (std::size_t i = 1; i <= count; ++i) {
                const float expected = leaf::kernels::gelu_new_scalar(source[i]);
                const float error = std::abs(output[i] - expected);
                maximum = std::max(maximum, error);
                if (!std::isfinite(output[i]) || error > 2e-6f + 1e-6f * std::abs(expected))
                    throw std::runtime_error("vector GELU exceeded tolerance");
                if (std::memcmp(&scalar[i], &expected, sizeof(float))) throw std::runtime_error("scalar GELU changed");
            }
        }
        // Special values and very large finite values in each SIMD lane.
        for (float value : {0.0f, -0.0f, 1e30f, -1e30f, std::numeric_limits<float>::max(),
                            -std::numeric_limits<float>::max(), INFINITY, -INFINITY, NAN}) {
            std::vector<float> output(8, value);
            leaf::kernels::gelu_new_inplace(output.data(), output.size());
            const float expected = leaf::kernels::gelu_new_scalar(value);
            for (float result : output)
                if (std::isnan(expected) ? !std::isnan(result) : std::memcmp(&result, &expected, sizeof(float)) != 0)
                    throw std::runtime_error("GELU special-value behavior changed");
        }
        leaf::kernels::gelu_new_inplace(nullptr, 0);
        bool rejected = false;
        try { leaf::kernels::gelu_new_inplace(nullptr, 1); } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("null GELU storage accepted");
        std::cout << "GELU correctness passed; maximum absolute error=" << maximum << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
