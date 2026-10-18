// External reference only. This executable is never loaded by Leaf.
#include "tree_sum.h"
#include <windows.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using I = long long; // The runner requires the explicitly named ILP64 exports.
using Sgemm = void (*)(int, int, int, I, I, I, float, const float*, I,
                      const float*, I, float, float*, I);
using SetThreads = void (*)(int);
using GetThreads = int (*)();
using Config = const char* (*)();
using Clock = std::chrono::steady_clock;
volatile float observed = 0;

template<class T> T symbol(HMODULE dll, const char* name) {
    auto value = GetProcAddress(dll, name);
    if (!value) throw std::runtime_error(std::string("Missing ILP64 export: ") + name);
    return reinterpret_cast<T>(value);
}

float random_value(std::uint32_t& seed) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return static_cast<float>(static_cast<int>(seed % 20001) - 10000) / 10000.0f;
}

struct Difference {
    double max_abs = 0, max_ratio = 0;
    std::size_t violations = 0;
    void add(float actual, double expected) {
        if (!std::isfinite(actual)) throw std::runtime_error("Nonfinite projection output");
        const auto delta = std::abs(static_cast<double>(actual) - expected);
        const auto ratio = delta / (2e-4 + 2e-4 * std::abs(expected));
        max_abs = std::max(max_abs, delta); max_ratio = std::max(max_ratio, ratio);
        violations += ratio > 1;
    }
    void print() const {
        std::cout << "{\"violations\":" << violations << ",\"max_abs_error\":" << max_abs
                  << ",\"max_tolerance_ratio\":" << max_ratio << '}';
    }
};

void shape(Sgemm blas, I m, I k, I n) {
    std::vector<float> x(m*k), w(n*k), baseline(m*n), tree(m*n), external(m*n);
    std::uint32_t seed = 0x672613af;
    for (auto& v : x) v = random_value(seed);
    for (auto& v : w) v = random_value(seed) / 32;
    leaf::kernels::TokenPanelsF32 panels;
    auto run = [&](unsigned algorithm) -> std::vector<float>& {
        auto& output = algorithm == 0 ? baseline : algorithm == 1 ? tree : external;
        if (algorithm == 2) {
            // Row-major Y[M,N] = X[M,K] * W[N,K]^T. No layout conversion.
            blas(101, 111, 112, m, n, k, 1, x.data(), k, w.data(), k, 0, output.data(), n);
        } else {
            panels.pack(x.data(), m, k, k, algorithm == 1);
            if (algorithm == 0)
                leaf::kernels::gemm_token_panels_f32_row_reuse(w.data(), n, k, k, panels, output.data(), n);
            else
                leaf::experiments::gemm_tree_sum(w.data(), n, k, k, panels, output.data(), n);
        }
        return output;
    };
    for (unsigned algorithm = 0; algorithm < 3; ++algorithm) run(algorithm);
    Difference errors[3];
    for (I token = 0; token < m; ++token) for (I row = 0; row < n; ++row) {
        double expected = 0;
        for (I col = 0; col < k; ++col)
            expected += static_cast<double>(x[token*k+col]) * w[row*k+col];
        const auto index = token*n+row;
        errors[0].add(baseline[index], expected);
        errors[1].add(tree[index], expected);
        errors[2].add(external[index], expected);
    }
    std::cout << "{\"m\":" << m << ",\"k\":" << k << ",\"n\":" << n << ",\"quality\":[";
    for (unsigned a = 0; a < 3; ++a) {
        if (a) std::cout << ',';
        errors[a].print();
        if (errors[a].violations) throw std::runtime_error("Projection tolerance failed");
    }
    std::cout << "],\"passes\":[";
    // Reverse ordering in the second round, keeping all algorithms adjacent.
    const unsigned order[] = {0, 1, 2, 2, 1, 0};
    const char* names[] = {"leaf_release", "leaf_accurate_tree", "openblas"};
    for (unsigned pass = 0; pass < 6; ++pass) {
        const auto a = order[pass];
        std::cout << (pass ? "," : "") << "{\"algorithm\":\"" << names[a] << "\",\"ms\":[";
        for (unsigned iteration = 0; iteration < 34; ++iteration) {
            const auto start = Clock::now();
            auto& values = run(a);
            const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            observed = values[(iteration*17) % values.size()];
            if (iteration >= 3) std::cout << (iteration > 3 ? "," : "") << ms;
        }
        std::cout << "]}";
    }
    std::cout << "]}";
}

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Pass an absolute OpenBLAS ILP64 DLL path");
        if (!SetProcessAffinityMask(GetCurrentProcess(), DWORD_PTR(1) << 2) ||
            !SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS))
            throw std::runtime_error("Cannot apply requested process policy");
        DWORD_PTR actual, available;
        if (!GetProcessAffinityMask(GetCurrentProcess(), &actual, &available) || actual != 4)
            throw std::runtime_error("Affinity verification failed");
        auto dll = LoadLibraryExW(argv[1], nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!dll) throw std::runtime_error("Cannot load external OpenBLAS DLL: " + std::to_string(GetLastError()));
        auto blas = symbol<Sgemm>(dll, "scipy_cblas_sgemm64_");
        symbol<SetThreads>(dll, "scipy_openblas_set_num_threads64_")(1);
        const auto threads = symbol<GetThreads>(dll, "scipy_openblas_get_num_threads64_")();
        if (threads != 1) throw std::runtime_error("OpenBLAS thread count did not become one");
        std::cout << std::setprecision(12) << "{\"external_reference_only\":true,\"integrated_into_leaf\":false,"
                     "\"cpu\":2,\"threads\":" << threads << ",\"openblas_config\":"
                  << std::quoted(symbol<Config>(dll, "scipy_openblas_get_config64_")())
                  << ",\"packing_included\":true,\"synthetic\":true,\"shapes\":[";
        shape(blas, 63, 960, 2560); std::cout << ',';
        shape(blas, 63, 896, 4864); std::cout << ',';
        shape(blas, 63, 2560, 960); std::cout << ',';
        shape(blas, 63, 4864, 896);
        std::cout << "]}\n";
        FreeLibrary(dll);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
