// Additional cache-block row boundaries and exact traversal-order parity.
#define main accumulation_probe_main
#include "probe.cpp"
#undef main

int main() {
    try {
        std::cout << "{\"row_boundaries\":[";
        shape(17, 129, 95, true, true, false); std::cout << ',';
        shape(17, 128, 96, true, false, false); std::cout << ',';
        shape(17, 127, 97, true, true, false);
        std::cout << "],\"exact_cache_traversal\":[";
        for (unsigned index = 0; index < 3; ++index) {
            const std::size_t rows = 95 + index, columns = 255 + index;
            const std::size_t tokens = 17, ws = columns + 3, ys = rows + 7;
            std::vector<float> input(tokens * columns), weights(rows * ws), bias(rows);
            std::vector<float> expected(tokens * ys, 123456.0f), actual = expected;
            std::uint32_t seed = 7319;
            for (auto& value : input) value = random_value(seed);
            for (auto& value : weights) value = random_value(seed);
            for (auto& value : bias) value = random_value(seed);
            leaf::kernels::TokenPanelsF32 packed;
            packed.pack(input.data(), tokens, columns, columns);
#if defined(__x86_64__) && defined(__GNUC__)
            if (leaf::kernels::token_panel_f32_avx2_available()) {
                leaf::experiments::blocked_sum_detail::gemm(
                    weights.data(), rows, columns, ws, packed, expected.data(), ys, bias.data());
                leaf::experiments::blocked_sum_detail::gemm_cached(
                    weights.data(), rows, columns, ws, packed, actual.data(), ys, bias.data());
                if (std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)))
                    throw std::runtime_error("cache traversal changed arithmetic or padding");
            } else throw std::runtime_error("AVX2/FMA is required for the cache traversal check");
#endif
            if (index) std::cout << ',';
            std::cout << "{\"rows\":" << rows << ",\"columns\":" << columns << ",\"exact\":true}";
        }
        std::cout << "]}\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
