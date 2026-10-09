#include "leaf/kernels/silu_gate.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {
std::uint32_t maximum_ulp = 0;
std::size_t checked = 0;
float reference(float up, float gate) {
    return up * (gate / (1.0f + std::exp(-gate)));
}
std::uint32_t ordered(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}
void compare(float actual, float expected, bool exact = false) {
    ++checked;
    if (std::isnan(expected)) {
        if (!std::isnan(actual)) throw std::runtime_error("NaN classification changed");
        return;
    }
    if (exact || !std::isfinite(expected) || expected == 0.0f) {
        if (std::memcmp(&actual, &expected, sizeof(float)))
            throw std::runtime_error("scalar/special-value result changed");
        return;
    }
    if (!std::isfinite(actual)) throw std::runtime_error("finite result became nonfinite");
    const auto a = ordered(actual), b = ordered(expected);
    const auto ulp = a > b ? a - b : b - a;
    maximum_ulp = std::max(maximum_ulp, ulp);
    if (ulp > 3) throw std::runtime_error("SiLU gating exceeds 3 ULP vs scalar reference");
}
void exercise(const std::vector<float>& gates, const std::vector<float>& inputs) {
    const auto n = gates.size();
    std::vector<float> g(n + 2, 8765.0f), u(n + 2, 7654.0f);
    std::copy(gates.begin(), gates.end(), g.begin() + 1);
    std::copy(inputs.begin(), inputs.end(), u.begin() + 1);
    const auto saved_g = g;
    auto scalar = u;
    leaf::kernels::silu_gate_inplace(u.data() + 1, g.data() + 1, n);
    leaf::kernels::silu_gate_inplace(scalar.data() + 1, g.data() + 1, n, false);
    if (u.front() != 7654 || u.back() != 7654 || scalar.front() != 7654 || scalar.back() != 7654 ||
        std::memcmp(g.data(), saved_g.data(), g.size() * sizeof(float)))
        throw std::runtime_error("guards or read-only gate buffer changed");
    for (std::size_t i = 0; i < n; ++i) {
        const float expected = reference(inputs[i], gates[i]);
        compare(u[i + 1], expected);
        compare(scalar[i + 1], expected, true);
    }
    auto alias = gates;
    leaf::kernels::silu_gate_inplace(alias.data(), alias.data(), n);
    for (std::size_t i = 0; i < n; ++i) compare(alias[i], reference(gates[i], gates[i]));
}
}

int main() {
    try {
        for (std::size_t n : {0, 1, 3, 4, 5, 7, 8, 9, 31, 63, 2048, 5632, 354816, 1000001}) {
            std::vector<float> g(n), u(n);
            for (std::size_t i = 0; i < n; ++i) {
                g[i] = -80.0f + 160.0f * static_cast<float>(i) / std::max<std::size_t>(1, n - 1);
                u[i] = (static_cast<int>(i % 113) - 56) * 0.125f;
            }
            exercise(g, u);
        }
        std::mt19937 rng(512);
        std::uniform_real_distribution<float> gates(-80, 80), mantissa(-1, 1);
        std::uniform_int_distribution<int> exponents(-110, 110);
        std::vector<float> g(262147), u(g.size());
        for (std::size_t i = 0; i < g.size(); ++i) {
            g[i] = gates(rng);
            u[i] = std::ldexp(mantissa(rng), exponents(rng));
        }
        exercise(g, u);
        const float specials[] = {0.0f, -0.0f, std::numeric_limits<float>::denorm_min(),
            -std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::min(),
            -std::numeric_limits<float>::min(), -80.0f, 80.0f,
            std::nextafter(-80.0f, -INFINITY), std::nextafter(80.0f, INFINITY),
            -88.8f, 88.8f, -104.0f, 104.0f, -1e30f, 1e30f,
            -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
            -INFINITY, INFINITY, NAN};
        for (float gate : specials) for (float input : specials) {
            // Every SIMD lane and a scalar tail, mixed with ordinary inputs.
            for (unsigned lane = 0; lane < 5; ++lane) {
                g.assign(5, 0.375f); u.assign(5, -2.5f);
                g[lane] = gate; u[lane] = input;
                exercise(g, u);
            }
        }
        leaf::kernels::silu_gate_inplace(nullptr, nullptr, 0);
        float value = 1;
        for (bool missing_up : {false, true}) {
            bool rejected = false;
            try { leaf::kernels::silu_gate_inplace(missing_up ? nullptr : &value,
                                                  missing_up ? &value : nullptr, 1); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("null storage accepted");
        }
        bool vector = false;
#if defined(__x86_64__) && defined(__GNUC__)
        vector = __builtin_cpu_supports("avx2");
#endif
        std::cout << "SiLU gating correctness passed; checked=" << checked
                  << " max_ulp=" << maximum_ulp << " avx2_exercised=" << vector << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
