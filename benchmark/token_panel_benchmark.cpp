// Actual GPT-2 prefill shapes. Synthetic inputs; no model-level speed claim.
#include "leaf/kernels/token_panel.h"
#include "leaf/kernels/weight_panel.h"
#include "leaf/kernels/wide_token.h"
#include "leaf/kernels/unrolled_token.h"
#include "leaf/kernels/row_reuse.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using Gemm = decltype(&leaf::kernels::gemm_token_panels_f32);
volatile double observed_checksum = 0;

void shape(std::size_t rows, std::size_t columns, unsigned runs, unsigned warmup, const std::string& kind, unsigned copies) {
    const bool weight_panel = kind == "weight_panel", wide = kind == "wide_token";
    constexpr std::size_t tokens = 63;
    std::vector<float> input(tokens * columns), weights(copies * rows * columns), bias(rows);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(static_cast<int>((i * 17 + 3) % 101) - 50) / 53.0f;
    for (std::size_t i = 0; i < weights.size(); ++i)
        weights[i] = static_cast<float>(static_cast<int>(((i % (rows * columns)) * 13 + 7) % 97) - 48) / 71.0f;
    for (std::size_t i = 0; i < rows; ++i) bias[i] = static_cast<float>(i % 7) / 11.0f;
    leaf::kernels::TokenPanelsF32 panels;
    panels.pack(input.data(), tokens, columns, columns);
    leaf::kernels::WideTokenPanelsF32 wide_panels;
    wide_panels.pack(input.data(), tokens, columns, columns);
    std::vector<float> full(tokens * rows), blocked(full.size());
    leaf::kernels::gemm_token_panels_f32(weights.data(), rows, columns, columns,
        panels, full.data(), rows, bias.data());
    if (kind == "row_reuse")
        leaf::kernels::gemm_token_panels_f32_row_reuse(weights.data(), rows, columns, columns,
            panels, blocked.data(), rows, bias.data());
    else if (kind == "unrolled")
        leaf::kernels::gemm_token_panels_f32_unrolled(weights.data(), rows, columns, columns,
            panels, blocked.data(), rows, bias.data());
    else if (wide)
        leaf::kernels::gemm_wide_tokens_f32(weights.data(), rows, columns, columns,
            wide_panels, blocked.data(), rows, bias.data());
    else if (weight_panel)
        leaf::kernels::gemm_weight_panels_f32(input.data(), tokens, columns, weights.data(), rows,
            columns, columns, blocked.data(), rows, bias.data());
    else
        leaf::kernels::gemm_token_panels_f32_kblocked(weights.data(), rows, columns, columns,
            panels, blocked.data(), rows, bias.data());
    if (std::memcmp(full.data(), blocked.data(), full.size() * sizeof(float)))
        throw std::runtime_error("candidate output is not bit-exact to full-K");
    double max_error = 0;
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t r = 0; r < rows; ++r) {
            double expected = bias[r];
            for (std::size_t k = 0; k < columns; ++k)
                expected += static_cast<double>(input[t * columns + k]) * weights[r * columns + k];
            const double error = std::abs(full[t * rows + r] - expected);
            max_error = std::max(max_error, error);
            if (!std::isfinite(full[t * rows + r]) || error > 1e-3 + 1e-4 * std::abs(expected))
                throw std::runtime_error("FP64 reference tolerance exceeded");
        }
    std::cout << "{\"m\":63,\"n\":" << rows << ",\"k\":" << columns
              << ",\"bit_exact\":true,\"max_absolute_error_fp64\":" << max_error
              << ",\"packed_input_bytes\":" << panels.storage_size() * sizeof(float)
              << ",\"passes\":[";
    // Same resident matrices and reusable output storage in both stages.
    // Include each algorithm's packing: input panels for token candidates,
    // bounded transient weight panels for the weight-panel candidate.
    for (unsigned pass = 0; pass < 4; ++pass) {
        const bool candidate = pass == 1 || pass == 2;
        Gemm gemm = candidate ? &leaf::kernels::gemm_token_panels_f32_kblocked
                              : &leaf::kernels::gemm_token_panels_f32;
        if (candidate && kind == "row_reuse") gemm = &leaf::kernels::gemm_token_panels_f32_row_reuse;
        if (candidate && kind == "unrolled") gemm = &leaf::kernels::gemm_token_panels_f32_unrolled;
        std::cout << (pass ? "," : "") << "{\"stage\":\""
                  << (candidate ? kind : "full_k") << "\",\"samples_ms\":[";
        for (unsigned iteration = 0; iteration < warmup + runs; ++iteration) {
            const auto start = Clock::now();
            for (unsigned copy = 0; copy < copies; ++copy) {
            const auto* current_weights = weights.data() + copy * rows * columns;
            if (candidate && wide) {
                wide_panels.pack(input.data(), tokens, columns, columns);
                leaf::kernels::gemm_wide_tokens_f32(current_weights, rows, columns, columns,
                    wide_panels, blocked.data(), rows, bias.data());
            } else if (candidate && weight_panel)
                leaf::kernels::gemm_weight_panels_f32(input.data(), tokens, columns, current_weights, rows,
                    columns, columns, blocked.data(), rows, bias.data());
            else {
                panels.pack(input.data(), tokens, columns, columns);
                gemm(current_weights, rows, columns, columns, panels, blocked.data(), rows, bias.data(), true);
            }
            }
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count() / copies;
            double checksum = 0;
            for (float value : blocked) checksum += value;
            observed_checksum = checksum;
            if (iteration >= warmup) std::cout << (iteration > warmup ? "," : "") << ms;
        }
        std::cout << "]}";
    }
    std::cout << "]}";
}

int main(int argc, char** argv) {
    try {
        const unsigned runs = argc > 1 ? std::stoul(argv[1]) : 21;
        const unsigned warmup = argc > 2 ? std::stoul(argv[2]) : 10;
        const std::string candidate = argc > 3 ? argv[3] : "kblocked";
        if (candidate != "kblocked" && candidate != "weight_panel" && candidate != "wide_token" && candidate != "unrolled" && candidate != "row_reuse") throw std::runtime_error("invalid candidate");
        const unsigned copies = argc > 4 ? std::stoul(argv[4]) : 1;
        if (copies < 1 || copies > 32) throw std::runtime_error("invalid weight copies");
        if (runs < 5 || runs > 10000 || warmup > 10000) throw std::runtime_error("invalid runs/warmup");
        std::cout << std::setprecision(10) << "{\"avx2_fma\":"
                  << (leaf::kernels::token_panel_f32_avx2_available() ? "true" : "false")
                  << ",\"weight_copies\":" << copies << ",\"candidate\":\"" << candidate << "\",\"packing_included\":true,\"reuses_workspace\":true,\"shapes\":[";
        shape(768, 768, runs, warmup, candidate, copies); std::cout << ',';
        shape(3072, 768, runs, warmup, candidate, copies); std::cout << ',';
        shape(768, 3072, runs, warmup, candidate, copies); std::cout << ',';
        shape(2304, 768, runs, warmup, candidate, copies);
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
