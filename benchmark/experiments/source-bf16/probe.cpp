// Storage-only hypothesis: expand original BF16 weights to FP32 in registers.
// Not a runtime format or dispatch change. Match the existing GEMV reduction.
#include "leaf/kernels/gemv_pair.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Clock = std::chrono::steady_clock;
volatile float observed;

__attribute__((target("avx2,fma"), always_inline)) inline __m256 expand(const std::uint16_t* w) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w))), 16));
}

__attribute__((target("avx2,fma"))) void pair_bf16(
        const std::uint16_t* weights, const float* input, std::size_t columns, float* output) {
    const auto* second = weights + columns;
    auto a0 = _mm256_setzero_ps(), b0 = a0, c0 = a0, d0 = a0;
    auto a1 = a0, b1 = a0, c1 = a0, d1 = a0;
    std::size_t col = 0;
    for (; columns - col >= 32; col += 32) {
        auto x = _mm256_loadu_ps(input + col);
        a0 = _mm256_fmadd_ps(expand(weights + col), x, a0);
        a1 = _mm256_fmadd_ps(expand(second + col), x, a1);
        x = _mm256_loadu_ps(input + col + 8);
        b0 = _mm256_fmadd_ps(expand(weights + col + 8), x, b0);
        b1 = _mm256_fmadd_ps(expand(second + col + 8), x, b1);
        x = _mm256_loadu_ps(input + col + 16);
        c0 = _mm256_fmadd_ps(expand(weights + col + 16), x, c0);
        c1 = _mm256_fmadd_ps(expand(second + col + 16), x, c1);
        x = _mm256_loadu_ps(input + col + 24);
        d0 = _mm256_fmadd_ps(expand(weights + col + 24), x, d0);
        d1 = _mm256_fmadd_ps(expand(second + col + 24), x, d1);
    }
    a0 = _mm256_add_ps(_mm256_add_ps(a0, b0), _mm256_add_ps(c0, d0));
    a1 = _mm256_add_ps(_mm256_add_ps(a1, b1), _mm256_add_ps(c1, d1));
    for (; columns - col >= 8; col += 8) {
        const auto x = _mm256_loadu_ps(input + col);
        a0 = _mm256_fmadd_ps(expand(weights + col), x, a0);
        a1 = _mm256_fmadd_ps(expand(second + col), x, a1);
    }
    output[0] = 0.0f + leaf::kernels::gemv_pair_detail::reduce(a0);
    output[1] = 0.0f + leaf::kernels::gemv_pair_detail::reduce(a1);
}

void require_ac() {
    SYSTEM_POWER_STATUS power{};
    if (!GetSystemPowerStatus(&power) || power.ACLineStatus != 1)
        throw std::runtime_error("AC power missing or unknown; stop timing");
}

int main(int argc, char** argv) {
    try {
        const bool check_only = argc == 5 && std::string(argv[4]) == "--check-only";
        if (argc != 4 && !check_only) throw std::runtime_error("SOURCE_BF16_FILE ROWS COLUMNS [--check-only]");
        const auto rows = std::stoull(argv[2]), columns = std::stoull(argv[3]);
        if (!rows || rows % 2 || columns < 8 || columns % 8 || rows > SIZE_MAX / columns / 4)
            throw std::runtime_error("This isolated probe requires even rows and K divisible by eight");
        if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma"))
            throw std::runtime_error("Probe requires AVX2/FMA");
        if (!SetProcessAffinityMask(GetCurrentProcess(), DWORD_PTR(1) << 2) ||
            !SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS))
            throw std::runtime_error("Cannot apply the benchmark CPU/priority policy");
        std::vector<std::uint16_t> compact(rows * columns);
        std::vector<float> full(compact.size()), input(columns), before(rows), after(rows);
        std::ifstream file(argv[1], std::ios::binary);
        if (!file.read(reinterpret_cast<char*>(compact.data()), compact.size()*2) ||
            file.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("Invalid raw BF16 tensor length");
        for (std::size_t i = 0; i < compact.size(); ++i) {
            const std::uint32_t bits = std::uint32_t(compact[i]) << 16;
            std::memcpy(&full[i], &bits, 4);
        }
        for (std::size_t i = 0; i < columns; ++i)
            input[i] = (static_cast<int>((i*19)%257)-128)/128.0f;
        auto run = [&](bool compressed, float* output) {
            for (std::size_t row = 0; row < rows; row += 2) {
                if (compressed) pair_bf16(compact.data()+row*columns, input.data(), columns, output+row);
                else leaf::kernels::gemv_pair_f32(full.data()+row*columns, input.data(), columns, output+row);
            }
        };
        run(false, before.data()); run(true, after.data());
        if (std::memcmp(before.data(), after.data(), rows*4))
            throw std::runtime_error("Storage expansion changed FP32 output bits");
        if (check_only) {
            std::cout << "{\"exact_fp32_output\":true,\"timing_measured\":false,\"rows\":"
                      << rows << ",\"columns\":" << columns << "}\n";
            return 0;
        }
        std::cout << std::setprecision(12) << "{\"exact_fp32_output\":true,\"cpu\":2,\"rows\":"
                  << rows << ",\"columns\":" << columns << ",\"fp32_weight_bytes\":" << full.size()*4
                  << ",\"bf16_weight_bytes\":" << compact.size()*2 << ",\"passes\":[";
        for (unsigned pass = 0; pass < 4; ++pass) {
            const bool compressed = pass == 1 || pass == 2;
            std::cout << (pass ? "," : "") << "{\"stage\":\"" << (compressed ? "bf16" : "fp32") << "\",\"ms\":[";
            for (unsigned i = 0; i < 14; ++i) {
                require_ac();
                const auto start = Clock::now();
                run(compressed, after.data());
                const auto ms = std::chrono::duration<double,std::milli>(Clock::now()-start).count();
                require_ac();
                observed = after[i%rows];
                if (i >= 3) std::cout << (i > 3 ? "," : "") << ms;
            }
            std::cout << "]}";
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
