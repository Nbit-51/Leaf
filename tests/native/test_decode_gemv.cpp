#include "gemv_access.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    try {
        std::mt19937 random(815);
        std::uniform_real_distribution<float> draw(-2, 2);
        for (std::size_t columns : {1, 7, 8, 15, 31, 32, 33, 63, 64, 65, 768, 3072}) {
            for (std::size_t rows : {1, 2, 3, 7, 32, 33}) {
                std::vector<float> weights(rows * columns + 1), input(columns + 1);
                for (auto& x : weights) x = draw(random);
                for (auto& x : input) x = draw(random);
                const auto saved_weights = weights, saved_input = input;
                std::vector<float> before(rows), candidate(rows + 2, 9876);
                leaf::runtime::testing::gemv_f32(weights.data() + 1, input.data() + 1,
                    before.data(), rows, columns, 0);
                for (unsigned variant : {1, 2}) {
                    // Odd worker-like row slices must preserve the tail path.
                    for (std::size_t first = 0; first < rows; first += 3) {
                        const auto count = std::min<std::size_t>(3, rows - first);
                        leaf::runtime::testing::gemv_f32(weights.data() + 1 + first * columns,
                            input.data() + 1, candidate.data() + 1 + first, count, columns, variant);
                    }
                    if (variant == 2 && std::memcmp(before.data(), candidate.data() + 1, rows * sizeof(float)))
                        throw std::runtime_error("GEMV bit-exact parity failed: variant=" + std::to_string(variant) +
                            " rows=" + std::to_string(rows) + " columns=" + std::to_string(columns));
                    if (candidate.front() != 9876 || candidate.back() != 9876)
                        throw std::runtime_error("GEMV output guards overwritten");
                    for (std::size_t row = 0; row < rows; ++row) {
                        double expected = 0;
                        for (std::size_t col = 0; col < columns; ++col)
                            expected += double(weights[1 + row * columns + col]) * input[1 + col];
                        if (!std::isfinite(candidate[row + 1]) ||
                            std::abs(candidate[row + 1] - expected) > 2e-5 * (1 + std::abs(expected)))
                            throw std::runtime_error("GEMV FP64 tolerance failed");
                    }
                }
                if (weights != saved_weights || input != saved_input)
                    throw std::runtime_error("GEMV modified input");
            }
        }
        std::cout << "Decode GEMV correctness passed: real kernels, tails, unaligned input, row slices, guards and FP64\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
