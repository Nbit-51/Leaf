// Paired full-range versus finite-span attention. Writes every raw sample.
#include "leaf/kernels/attention_vector.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
    try {
        if (argc != 2 || !__builtin_cpu_supports("avx2")) throw std::runtime_error("Require AVX2 and output path");
        std::ofstream record(argv[1]);
        if (!record) throw std::runtime_error("Cannot create record");
        std::mt19937 random(924);
        std::uniform_real_distribution<float> draw(-3, 3);
        for (std::size_t n : {17, 64, 257, 669}) for (unsigned kind = 0; kind < 3; ++kind) {
            constexpr std::size_t dim = 256, heads = 3;
            std::vector<float> q(heads*n*dim), k(n*dim), v(n*dim), mask(n*n), a(q.size()), b(q.size());
            for (auto* data : {&q, &k, &v}) for (auto& x : *data) x = draw(random);
            for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j)
                mask[i*n+j] = kind == 0 || (kind == 1 ? j <= i : (i > j ? i-j : j-i) <= 256);
            const std::size_t shape[] = {1, 1, n, n};
            auto run = [&](bool trimmed) {
                auto start = std::chrono::steady_clock::now();
                if (trimmed) leaf::kernels::attention_vector_detail::attend<true>(q.data(), k.data(), v.data(), mask.data(),
                    b.data(), 1, heads, 1, n, n, n, dim, shape, .25f, true);
                else leaf::kernels::attention_vector_detail::attend<false>(q.data(), k.data(), v.data(), mask.data(),
                    a.data(), 1, heads, 1, n, n, n, dim, shape, .25f, true);
                return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            };
            run(false); run(true);
            if (std::memcmp(a.data(), b.data(), a.size()*sizeof(float))) throw std::runtime_error("Attention bits changed");
            std::vector<double> before, after;
            for (unsigned i = 0; i < 24; ++i) {
                double x, y;
                if (i % 2) { y = run(true); x = run(false); } else { x = run(false); y = run(true); }
                if (i >= 3) { before.push_back(x); after.push_back(y); }
            }
            record << "{\"tokens\":" << n << ",\"mask\":\"" << (kind==0 ? "full" : kind==1 ? "causal" : "local256")
                   << "\",\"bit_exact\":true,\"before_ms\":[";
            for (unsigned i = 0; i < before.size(); ++i) record << (i ? "," : "") << before[i];
            record << "],\"after_ms\":[";
            for (unsigned i = 0; i < after.size(); ++i) record << (i ? "," : "") << after[i];
            record << "]}" << std::endl;
            std::cout << n << ' ' << kind << " completed\n" << std::flush;
        }
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
