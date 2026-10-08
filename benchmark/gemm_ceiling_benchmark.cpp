// Diagnostic only: separates the shipped row-reuse GEMM's compute efficiency
// from weight-memory exposure. The FMA ceiling is measured on this core with
// independent register accumulators; no nominal clock frequency is assumed.
#include "leaf/kernels/row_reuse.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
volatile float sink = 0;

__attribute__((target("avx2,fma"), noinline)) double fma_ns_per_instruction(std::size_t iterations) {
    __m256 a0 = _mm256_set1_ps(1.000f), a1 = _mm256_set1_ps(1.001f), a2 = _mm256_set1_ps(1.002f);
    __m256 a3 = _mm256_set1_ps(1.003f), a4 = _mm256_set1_ps(1.004f), a5 = _mm256_set1_ps(1.005f);
    __m256 a6 = _mm256_set1_ps(1.006f), a7 = _mm256_set1_ps(1.007f), a8 = _mm256_set1_ps(1.008f);
    __m256 a9 = _mm256_set1_ps(1.009f), a10 = _mm256_set1_ps(1.010f), a11 = _mm256_set1_ps(1.011f);
    const auto m = _mm256_set1_ps(0.9999999f), c = _mm256_set1_ps(1e-7f);
    const auto start = Clock::now();
    for (std::size_t n = 0; n < iterations; ++n) {
        a0 = _mm256_fmadd_ps(a0, m, c); a1 = _mm256_fmadd_ps(a1, m, c); a2 = _mm256_fmadd_ps(a2, m, c);
        a3 = _mm256_fmadd_ps(a3, m, c); a4 = _mm256_fmadd_ps(a4, m, c); a5 = _mm256_fmadd_ps(a5, m, c);
        a6 = _mm256_fmadd_ps(a6, m, c); a7 = _mm256_fmadd_ps(a7, m, c); a8 = _mm256_fmadd_ps(a8, m, c);
        a9 = _mm256_fmadd_ps(a9, m, c); a10 = _mm256_fmadd_ps(a10, m, c); a11 = _mm256_fmadd_ps(a11, m, c);
        asm volatile("" : "+x"(a0), "+x"(a1), "+x"(a2), "+x"(a3), "+x"(a4), "+x"(a5),
                          "+x"(a6), "+x"(a7), "+x"(a8), "+x"(a9), "+x"(a10), "+x"(a11));
    }
    const double ns = std::chrono::duration<double, std::nano>(Clock::now() - start).count();
    const __m256 all[] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11};
    float total = 0;
    for (auto v : all) total += _mm256_cvtss_f32(v);
    sink = total;
    return ns / (iterations * 12.0);
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Returns median ms per matrix. copies > 1 rotates distinct resident weights.
double gemm_ms(std::size_t tokens, std::size_t rows, std::size_t columns, unsigned copies,
               bool include_pack, unsigned runs, unsigned warmup) {
    std::vector<float> input(tokens * columns), weights(copies * rows * columns), bias(rows), out(tokens * rows);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int((i * 17 + 3) % 101) - 50) / 53.0f;
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int((i * 13 + 7) % 97) - 48) / 71.0f;
    for (std::size_t i = 0; i < rows; ++i) bias[i] = float(i % 7) / 11.0f;
    leaf::kernels::TokenPanelsF32 panels;
    panels.pack(input.data(), tokens, columns, columns);
    std::vector<double> samples;
    for (unsigned it = 0; it < warmup + runs; ++it) {
        const auto start = Clock::now();
        for (unsigned c = 0; c < copies; ++c) {
            if (include_pack) panels.pack(input.data(), tokens, columns, columns);
            leaf::kernels::gemm_token_panels_f32_row_reuse(weights.data() + c * rows * columns, rows, columns,
                columns, panels, out.data(), rows, bias.data());
        }
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count() / copies;
        sink = out[tokens * rows - 1];
        if (it >= warmup) samples.push_back(ms);
    }
    return median(samples);
}

int main(int argc, char** argv) {
    const unsigned runs = argc > 1 ? std::stoul(argv[1]) : 31, warmup = argc > 2 ? std::stoul(argv[2]) : 10;
    std::cout << std::setprecision(8) << "{\"fma_ns_per_256bit_instruction\":[";
    for (int i = 0; i < 5; ++i) std::cout << (i ? "," : "") << fma_ns_per_instruction(50'000'000);
    std::cout << "],\"cases\":[";
    struct Case { const char* name; std::size_t m, n, k; unsigned copies; bool pack; };
    const Case cases[] = {
        {"l1_resident_m64_n96_k256", 64, 96, 256, 1, false},
        {"l2_resident_m64_n384_k768", 64, 384, 768, 1, false},
        {"mlp_up_hot", 63, 3072, 768, 1, true},
        {"mlp_up_rotating12", 63, 3072, 768, 12, true},
        {"mlp_down_hot", 63, 768, 3072, 1, true},
        {"mlp_down_rotating12", 63, 768, 3072, 12, true},
    };
    bool first = true;
    for (const auto& c : cases) {
        const double ceiling = fma_ns_per_instruction(20'000'000);
        const double ms = gemm_ms(c.m, c.n, c.k, c.copies, c.pack, runs, warmup);
        // 256-bit FMA instructions actually issued, including padded token lanes.
        const double instructions = double((c.m + 15) / 16 * 2) * c.n * c.k;
        std::cout << (first ? "" : ",") << "{\"name\":\"" << c.name << "\",\"m\":" << c.m << ",\"n\":" << c.n
                  << ",\"k\":" << c.k << ",\"weight_copies\":" << c.copies << ",\"packing_included\":"
                  << (c.pack ? "true" : "false") << ",\"median_ms\":" << ms
                  << ",\"fma_instructions\":" << instructions
                  << ",\"adjacent_fma_ceiling_ns\":" << ceiling
                  << ",\"fraction_of_measured_ceiling\":" << instructions * ceiling * 1e-6 / ms << "}";
        first = false;
    }
    std::cout << "]}\n";
}
