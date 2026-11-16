#include "leaf/kernels/accurate_fp32.h"
#define LEAF_ACCUMULATION_TREE_CHUNK 256
#define LEAF_ACCUMULATION_INLINE_PARTIAL 1
#include "../../benchmark/experiments/linear-accumulation/tree_sum.h"
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <random>

int main() {
    std::mt19937 random(51026);
    std::uniform_real_distribution<float> value(-0.25f, 0.25f);
    std::size_t checked = 0;
    const bool vector = leaf::kernels::token_panel_f32_avx2_available();
    for (auto rows : {1u, 5u, 6u, 7u, 13u})
        for (auto tokens : {1u, 7u, 8u, 15u, 16u, 17u, 33u})
            for (auto columns : {1u, 127u, 255u, 256u, 257u, 513u, 4097u}) {
                const auto ws = columns + 3, ys = rows + 2;
                std::vector<float> w(std::size_t(rows) * ws), x(std::size_t(tokens) * columns), bias(rows);
                for (auto& v : w) v = value(random);
                for (auto& v : x) v = value(random);
                for (auto& v : bias) v = value(random);
                leaf::kernels::TokenPanelsF32 panels;
                panels.pack(x.data(), tokens, columns, columns, vector);
                for (bool has_bias : {false, true}) {
                    std::vector<float> result(std::size_t(tokens) * ys, 12345), reference(result.size(), 12345);
                    auto b = has_bias ? bias.data() : nullptr;
                    leaf::kernels::gemm_token_panels_f32_accurate(w.data(), rows, columns, ws, panels, result.data(), ys, b, vector);
                    if (vector) {
                        leaf::experiments::gemm_tree_sum(w.data(), rows, columns, ws, panels, reference.data(), ys, b, true);
                        if (std::memcmp(result.data(), reference.data(), result.size() * sizeof(float))) return 1;
#if defined(__x86_64__) && defined(__GNUC__)
                        std::fill(reference.begin(), reference.end(), 12345.0f);
                        for (std::size_t first = 0; first < panels.panels(); first += 2)
                            leaf::kernels::accurate_fp32_detail::gemm(w.data(), rows, columns, ws, panels,
                                reference.data(), ys, b, first, first + 2);
                        if (std::memcmp(result.data(), reference.data(), result.size() * sizeof(float))) return 6;
#endif
                    }
                    for (unsigned t = 0; t < tokens; ++t) {
                        for (unsigned r = 0; r < rows; ++r) {
                            double dot = 0;
                            for (unsigned k = 0; k < columns; ++k) dot += double(w[r * ws + k]) * x[t * columns + k];
                            float expected = static_cast<float>(dot) + (b ? b[r] : 0);
                            if (std::abs(result[t * ys + r] - expected) > 1e-5f + 2e-5f * std::abs(expected)) return 2;
                            ++checked;
                        }
                        if (result[t * ys + rows] != 12345 || result[t * ys + rows + 1] != 12345) return 3;
                    }
                    // Exercise the portable numerical fallback independently.
                    leaf::kernels::gemm_token_panels_f32_accurate(w.data(), rows, columns, ws, panels, result.data(), ys, b, false);
                    for (unsigned t = 0; t < tokens; ++t) {
                        for (unsigned r = 0; r < rows; ++r) {
                            double dot = 0;
                            for (unsigned k = 0; k < columns; ++k) dot += double(w[r * ws + k]) * x[t * columns + k];
                            float expected = static_cast<float>(dot) + (b ? b[r] : 0);
                            if (std::memcmp(&result[t * ys + r], &expected, sizeof(float))) return 4;
                        }
                        if (result[t * ys + rows] != 12345 || result[t * ys + rows + 1] != 12345) return 7;
                    }
                }
            }
    const float rms[] = {10000.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    if (leaf::kernels::accurate_rms_square_sum(rms, 9) != 100000008.0f) return 5;
    std::cout << "PASS accurate FP32: " << checked << " outputs; AVX/private-reference exact=" << vector << '\n';
}
