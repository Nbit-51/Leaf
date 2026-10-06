// Calls the actual decoder's dot kernels through a test-only access point.
#include "../tests/native/gemv_access.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
volatile float checksum = 0;

void shape(std::size_t rows, std::size_t columns, unsigned copies,
           unsigned candidate, unsigned runs, unsigned warmup) {
    std::vector<float> weights(rows * columns * copies), input(columns);
    std::vector<float> reference(rows), output(rows);
    for (std::size_t i = 0; i < weights.size(); ++i)
        weights[i] = float(int((i % (rows * columns)) % 127) - 63) / 71.0f;
    for (std::size_t i = 0; i < columns; ++i) input[i] = float(int(i % 101) - 50) / 53.0f;
    leaf::runtime::testing::gemv_f32(weights.data(), input.data(), reference.data(), rows, columns, 0);
    leaf::runtime::testing::gemv_f32(weights.data(), input.data(), output.data(), rows, columns, candidate);
    if (std::memcmp(output.data(), reference.data(), rows * sizeof(float)))
        throw std::runtime_error("GEMV candidate is not bit-exact");
    double max_error = 0;
    for (std::size_t row = 0; row < rows; ++row) {
        double expected = 0;
        for (std::size_t col = 0; col < columns; ++col)
            expected += double(weights[row * columns + col]) * input[col];
        max_error = std::max(max_error, std::abs(expected - output[row]));
        if (!std::isfinite(output[row]) || std::abs(expected - output[row]) > 2e-5 * (1 + std::abs(expected)))
            throw std::runtime_error("GEMV exceeds FP64 tolerance");
    }
    std::cout << "{\"n\":" << rows << ",\"k\":" << columns << ",\"weight_copies\":" << copies
              << ",\"bit_exact\":true,\"max_abs_error_fp64\":" << max_error << ",\"passes\":[";
    for (unsigned pass = 0; pass < 4; ++pass) {
        const unsigned variant = pass == 1 || pass == 2 ? candidate : 0;
        std::cout << (pass ? "," : "") << "{\"stage\":\"" << (variant ? "optimized" : "full_k")
                  << "\",\"samples_ms\":[";
        for (unsigned iteration = 0; iteration < runs + warmup; ++iteration) {
            const auto start = Clock::now();
            for (unsigned copy = 0; copy < copies; ++copy) {
                leaf::runtime::testing::gemv_f32(weights.data() + copy * rows * columns,
                    input.data(), output.data(), rows, columns, variant);
            }
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count() / copies;
            checksum = output[iteration % rows];
            if (iteration >= warmup) std::cout << (iteration > warmup ? "," : "") << ms;
        }
        std::cout << "]}";
    }
    std::cout << "]}";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("expected candidate, runs, warmup");
        const unsigned candidate = std::stoul(argv[1]), runs = std::stoul(argv[2]), warmup = std::stoul(argv[3]);
        if ((candidate != 1 && candidate != 2) || runs < 5 || runs > 10000 || warmup > 10000)
            throw std::runtime_error("invalid benchmark arguments");
        std::cout << std::setprecision(10) << "{\"candidate\":" << candidate << ",\"shapes\":[";
        shape(768, 768, 12, candidate, runs, warmup); std::cout << ',';
        shape(3072, 768, 12, candidate, runs, warmup); std::cout << ',';
        shape(768, 3072, 12, candidate, runs, warmup); std::cout << ',';
        shape(50257, 768, 1, candidate, runs, warmup);
        std::cout << "]}\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
