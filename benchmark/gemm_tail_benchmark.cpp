// Paired shared-kernel experiment. Raw JSONL is flushed after every shape.
#include "leaf/kernels/row_reuse.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

using Clock = std::chrono::steady_clock;
volatile float tail_sink = 0;

template <bool Narrow>
double run(const std::vector<float>& input, const std::vector<float>& weights,
           std::size_t tokens, std::size_t rows, std::size_t columns, unsigned copies,
           leaf::kernels::TokenPanelsF32& panels, std::vector<float>& out) {
    const auto begin = Clock::now();
    for (unsigned c = 0; c < copies; ++c) {
        panels.pack(input.data(), tokens, columns, columns, true);
        leaf::kernels::row_reuse_detail::gemm<Narrow>(weights.data() + c * rows * columns,
            rows, columns, columns, panels, out.data(), rows, nullptr);
        tail_sink = out.back();
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count() / copies;
}

int main(int argc, char** argv) {
    try {
        if (argc != 2 || !leaf::kernels::token_panel_f32_avx2_available())
            throw std::runtime_error("Require AVX2/FMA and output JSONL path");
        std::ofstream record(argv[1]);
        if (!record) throw std::runtime_error("Cannot create record");
        constexpr unsigned copies = 8, runs = 21;
        // Gemma and GPT-style up/down projections; full tiles are controls.
        for (auto shape : {std::pair<std::size_t, std::size_t>{1152, 768}, {768, 1152}, {3072, 768}, {768, 3072}}) {
            const auto rows = shape.first, columns = shape.second;
            for (std::size_t tokens : {8, 16, 17, 18, 24, 25, 32, 33, 63}) {
                std::vector<float> x(tokens * columns), w(copies * rows * columns);
                for (std::size_t i = 0; i < x.size(); ++i) x[i] = float(int(i % 97) - 48) / 53;
                for (std::size_t i = 0; i < w.size(); ++i) w[i] = float(int(i % 89) - 44) / 71;
                std::vector<float> a(tokens * rows), b(a.size());
                leaf::kernels::TokenPanelsF32 panels;
                run<false>(x, w, tokens, rows, columns, copies, panels, a);
                run<true>(x, w, tokens, rows, columns, copies, panels, b);
                if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)))
                    throw std::runtime_error("Tail output changed");
                std::vector<double> before, after;
                for (unsigned i = 0; i < runs + 3; ++i) {
                    double old_ms, new_ms;
                    if (i % 2) {
                        new_ms = run<true>(x, w, tokens, rows, columns, copies, panels, b);
                        old_ms = run<false>(x, w, tokens, rows, columns, copies, panels, a);
                    } else {
                        old_ms = run<false>(x, w, tokens, rows, columns, copies, panels, a);
                        new_ms = run<true>(x, w, tokens, rows, columns, copies, panels, b);
                    }
                    if (i >= 3) { before.push_back(old_ms); after.push_back(new_ms); }
                }
                record << "{\"tokens\":" << tokens << ",\"rows\":" << rows << ",\"columns\":" << columns
                       << ",\"copies\":" << copies << ",\"packing_included\":true,\"bit_exact\":true,\"before_ms\":[";
                for (unsigned i = 0; i < runs; ++i) record << (i ? "," : "") << before[i];
                record << "],\"after_ms\":[";
                for (unsigned i = 0; i < runs; ++i) record << (i ? "," : "") << after[i];
                record << "]}" << std::endl;
                std::cout << tokens << 'x' << rows << 'x' << columns << " complete\n" << std::flush;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
