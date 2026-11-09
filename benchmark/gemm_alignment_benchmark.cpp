// Diagnostic only. Packing is excluded; all offsets call one raw-panel entry.
// Run pinned/serially under the external AC/effective-power-mode audit.
#include "leaf/kernels/row_reuse.h"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
volatile float alignment_sink = 0;
using Sizes = std::array<std::size_t, 3>; // tokens, output rows, K

static std::uint64_t hash(const float* p, std::size_t n) {
    std::uint64_t h = 14695981039346656037ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n * sizeof(float); ++i) h = (h ^ bytes[i]) * 1099511628211ULL;
    return h;
}
static std::string hex(std::uint64_t n) {
    std::ostringstream s; s << std::hex << std::setfill('0') << std::setw(16) << n; return s.str();
}

// Standard vector storage owns live float objects; overallocate and select an
// interior 64-byte boundary, then apply the controlled offset. No allocator or
// production packing change is needed for this experiment.
struct PackedCopy {
    std::vector<float> storage;
    float* data;
    PackedCopy(const float* source, std::size_t n, std::size_t offset) : storage(n + 32) {
        const auto address = reinterpret_cast<std::uintptr_t>(storage.data());
        data = storage.data() + ((64 - address % 64) % 64 + offset) / sizeof(float);
        if (reinterpret_cast<std::uintptr_t>(data) % 64 != offset)
            throw std::runtime_error("Controlled packed alignment failed");
        std::memcpy(data, source, n * sizeof(float));
    }
};

#if defined(__x86_64__) && defined(__GNUC__)
// Target this independent scalar recurrence explicitly. A baseline MinGW fmaf
// library call differs from hardware fusion on the tested inputs; it is not an
// exact oracle for the production AVX2/FMA tile. No horizontal/vector reduction.
__attribute__((target("avx2,fma"), noinline, noclone))
static float scalar_fma_oracle(const float* weights, const float* input, std::size_t columns) {
    float value = 0;
    for (std::size_t k = 0; k < columns; ++k) value = std::fma(weights[k], input[k], value);
    return value + 0.0f;
}

__attribute__((target("avx2,fma"), noinline, noclone))
static void raw_gemm(const float* weights, const Sizes& s, const float* packed, float* output) {
    const auto tokens = s[0], rows = s[1], columns = s[2];
    for (std::size_t row = 0; row < rows;) {
        const auto live = std::min<std::size_t>(6, rows - row);
        for (std::size_t panel = 0; panel < (tokens + 15) / 16; ++panel) {
            const auto first = panel * 16, count = std::min<std::size_t>(16, tokens - first);
            const auto* w = weights + row * columns;
            const auto* x = packed + panel * columns * 16;
            auto* y = output + first * rows + row;
            // Same runtime switch and full-K tile as the ordinary shipped GEMM.
            switch (live) {
            case 1: leaf::kernels::token_panel_detail::tile<1>(w, columns, columns, x, count, y, rows, nullptr); break;
            case 2: leaf::kernels::token_panel_detail::tile<2>(w, columns, columns, x, count, y, rows, nullptr); break;
            case 3: leaf::kernels::token_panel_detail::tile<3>(w, columns, columns, x, count, y, rows, nullptr); break;
            case 4: leaf::kernels::token_panel_detail::tile<4>(w, columns, columns, x, count, y, rows, nullptr); break;
            case 5: leaf::kernels::token_panel_detail::tile<5>(w, columns, columns, x, count, y, rows, nullptr); break;
            case 6: leaf::kernels::token_panel_detail::tile<6>(w, columns, columns, x, count, y, rows, nullptr); break;
            }
        }
        row += live;
    }
}

__attribute__((target("avx2,fma"), noinline))
static double fma_ns(std::size_t iterations) {
    __m256 a0 = _mm256_set1_ps(1.000f), a1 = _mm256_set1_ps(1.001f), a2 = _mm256_set1_ps(1.002f);
    __m256 a3 = _mm256_set1_ps(1.003f), a4 = _mm256_set1_ps(1.004f), a5 = _mm256_set1_ps(1.005f);
    __m256 a6 = _mm256_set1_ps(1.006f), a7 = _mm256_set1_ps(1.007f), a8 = _mm256_set1_ps(1.008f);
    __m256 a9 = _mm256_set1_ps(1.009f), a10 = _mm256_set1_ps(1.010f), a11 = _mm256_set1_ps(1.011f);
    const auto m = _mm256_set1_ps(0.9999999f), c = _mm256_set1_ps(1e-7f);
    const auto begin = Clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        a0 = _mm256_fmadd_ps(a0, m, c); a1 = _mm256_fmadd_ps(a1, m, c); a2 = _mm256_fmadd_ps(a2, m, c);
        a3 = _mm256_fmadd_ps(a3, m, c); a4 = _mm256_fmadd_ps(a4, m, c); a5 = _mm256_fmadd_ps(a5, m, c);
        a6 = _mm256_fmadd_ps(a6, m, c); a7 = _mm256_fmadd_ps(a7, m, c); a8 = _mm256_fmadd_ps(a8, m, c);
        a9 = _mm256_fmadd_ps(a9, m, c); a10 = _mm256_fmadd_ps(a10, m, c); a11 = _mm256_fmadd_ps(a11, m, c);
        asm volatile("" : "+x"(a0), "+x"(a1), "+x"(a2), "+x"(a3), "+x"(a4), "+x"(a5),
                          "+x"(a6), "+x"(a7), "+x"(a8), "+x"(a9), "+x"(a10), "+x"(a11));
    }
    const double ns = std::chrono::duration<double, std::nano>(Clock::now() - begin).count();
    const __m256 all[] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11};
    float total = 0; for (auto v : all) total += _mm256_cvtss_f32(v); alignment_sink = total;
    return ns / (iterations * 12.0);
}
#else
static float scalar_fma_oracle(const float*, const float*, std::size_t) {
    throw std::runtime_error("This diagnostic requires GCC-compatible x86-64 AVX2/FMA");
}
static void raw_gemm(const float*, const Sizes&, const float*, float*) {
    throw std::runtime_error("This diagnostic requires GCC-compatible x86-64 AVX2/FMA");
}
static double fma_ns(std::size_t) { return 0; }
#endif

static void array(std::ostream& out, const std::vector<double>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
    out << ']';
}

static void shape(std::ostream& record, const Sizes& s, unsigned copies, unsigned runs, unsigned warmup) {
    const auto tokens = s[0], rows = s[1], columns = s[2];
    const auto matrix = rows * columns, packed_size = ((tokens + 15) / 16) * columns * 16;
    std::vector<float> input(tokens * columns), weights(copies * matrix), output(tokens * rows);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int((i * 17 + 3) % 101) - 50) / 53.0f;
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = float(int((i * 13 + 7) % 97) - 48) / 71.0f;
    leaf::kernels::TokenPanelsF32 panels;
    panels.pack(input.data(), tokens, columns, columns, true);
    const auto* source = panels.panel(0);
    const auto input_hash = hash(input.data(), input.size()), weight_hash = hash(weights.data(), weights.size());
    const auto packed_hash = hash(source, packed_size);
    const std::array<std::size_t, 4> offsets = {0, 16, 32, 48};
    std::array<PackedCopy, 4> phases = {PackedCopy(source, packed_size, 0), PackedCopy(source, packed_size, 16),
                                      PackedCopy(source, packed_size, 32), PackedCopy(source, packed_size, 48)};
    std::size_t oracle_checks = 0, exact_outputs = 0;
    std::vector<std::string> output_hashes;
    for (unsigned c = 0; c < copies; ++c) {
        const auto* w = weights.data() + c * matrix;
        leaf::kernels::gemm_token_panels_f32_row_reuse(w, rows, columns, columns, panels, output.data(), rows);
        const auto reference = output;
        output_hashes.push_back(hex(hash(reference.data(), reference.size())));
        // Cross-sample the full increasing-K fused scalar oracle, including tails.
        const std::array<std::size_t, 6> ts = {0, 1, 7, tokens / 2, tokens - 2, tokens - 1};
        const std::array<std::size_t, 6> rs = {0, 1, 5, rows / 2, rows - 2, rows - 1};
        for (auto t : ts) for (auto r : rs) {
            const auto value = scalar_fma_oracle(w + r * columns, input.data() + t * columns, columns);
            if (std::memcmp(&value, &reference[t * rows + r], sizeof(float))) {
                std::uint32_t expected, actual;
                std::memcpy(&expected, &value, sizeof(float));
                std::memcpy(&actual, &reference[t * rows + r], sizeof(float));
                throw std::runtime_error("Scalar full-K oracle mismatch at token=" + std::to_string(t) +
                    " row=" + std::to_string(r) + " copy=" + std::to_string(c) +
                    " expected_bits=" + hex(expected) + " actual_bits=" + hex(actual));
            }
            ++oracle_checks;
        }
        for (const auto& phase : phases) {
            if (std::memcmp(phase.data, source, packed_size * sizeof(float)))
                throw std::runtime_error("Packed copy mismatch");
            raw_gemm(w, s, phase.data, output.data());
            if (std::memcmp(reference.data(), output.data(), output.size() * sizeof(float)))
                throw std::runtime_error("Alignment output mismatch");
            exact_outputs += output.size();
        }
    }
    auto measure = [&](unsigned phase) {
        const auto begin = Clock::now();
        for (unsigned c = 0; c < copies; ++c) {
            raw_gemm(weights.data() + c * matrix, s, phases[phase].data, output.data());
            alignment_sink = output.back();
        }
        return std::chrono::duration<double, std::milli>(Clock::now() - begin).count() / copies;
    };
    for (unsigned i = 0; i < warmup; ++i) for (unsigned phase = 0; phase < 4; ++phase) measure(phase);
    std::vector<double> before, after;
    for (unsigned i = 0; i < 3; ++i) before.push_back(fma_ns(2000000));
    std::array<std::vector<double>, 4> samples;
    std::vector<unsigned> order;
    for (unsigned round = 0; round < runs; ++round) {
        // Palindromic phase order balances early/late position; reverse each round.
        const std::array<unsigned, 8> forward = {0, 1, 2, 3, 3, 2, 1, 0};
        for (auto p : forward) {
            const auto phase = round % 2 ? 3 - p : p;
            samples[phase].push_back(measure(phase)); order.push_back(phase);
        }
    }
    for (unsigned i = 0; i < 3; ++i) after.push_back(fma_ns(2000000));
    if (hash(input.data(), input.size()) != input_hash || hash(weights.data(), weights.size()) != weight_hash ||
        hash(source, packed_size) != packed_hash) throw std::runtime_error("Source changed during benchmark");
    for (const auto& phase : phases) if (hash(phase.data, packed_size) != packed_hash)
        throw std::runtime_error("Packed phase changed during benchmark");
    const auto instructions = double((tokens + 15) / 16) * 2 * rows * columns;
    record << "{\"type\":\"case\",\"complete\":true,\"tokens\":" << tokens << ",\"rows\":" << rows
           << ",\"columns\":" << columns << ",\"weight_copies\":" << copies << ",\"rounds\":" << runs
           << ",\"warmup_sweeps\":" << warmup << ",\"packing_included\":false,\"fp32_math\":\"fast\""
           << ",\"raw_entry\":\"single_noinline_runtime_shape\",\"whole_model_speed_claim\":false"
           << ",\"packed_bytes\":" << packed_size * sizeof(float) << ",\"weight_bytes\":" << weights.size() * sizeof(float)
           << ",\"original_packed_mod64\":" << reinterpret_cast<std::uintptr_t>(source) % 64
           << ",\"issued_fma_instructions_per_matrix\":" << instructions
           << ",\"bit_exact_values\":" << exact_outputs << ",\"scalar_oracle_positions\":" << oracle_checks
           << ",\"oracle_kind\":\"hardware_scalar_fma_direct_input\""
           << ",\"hash_kind\":\"fnv1a64_raw_bytes\",\"input_hash\":\"" << hex(input_hash)
           << "\",\"weight_hash\":\"" << hex(weight_hash) << "\",\"packed_hash\":\"" << hex(packed_hash)
           << "\",\"output_hashes\":[";
    for (unsigned i = 0; i < copies; ++i) record << (i ? "," : "") << '"' << output_hashes[i] << '"';
    record << "],\"fma_ns_before\":"; array(record, before);
    record << ",\"fma_ns_after\":"; array(record, after);
    record << ",\"sample_order\":[";
    for (std::size_t i = 0; i < order.size(); ++i) record << (i ? "," : "") << offsets[order[i]];
    record << "],\"phases\":[";
    for (unsigned p = 0; p < 4; ++p) {
        record << (p ? "," : "") << "{\"packed_mod64\":" << offsets[p] << ",\"ms_per_matrix\":";
        array(record, samples[p]); record << '}';
    }
    record << "]}" << std::endl;
    std::cout << tokens << 'x' << rows << 'x' << columns << " copies=" << copies << " complete\n" << std::flush;
}

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 4 || !leaf::kernels::token_panel_f32_avx2_available())
            throw std::runtime_error("Usage: gemm_alignment_benchmark OUTPUT.jsonl [ROUNDS=7] [WARMUP_SWEEPS=3]; requires AVX2/FMA");
        const unsigned runs = argc > 2 ? std::stoul(argv[2]) : 7, warmup = argc > 3 ? std::stoul(argv[3]) : 3;
        if (!runs || runs > 1000 || warmup > 1000) throw std::runtime_error("Invalid rounds/warmup");
        std::ifstream existing(argv[1]);
        if (existing.good()) throw std::runtime_error("Output already exists; use a fresh file");
        std::ofstream record(argv[1]);
        if (!record) throw std::runtime_error("Cannot create output");
        record.exceptions(std::ios::badbit | std::ios::failbit);
        record << std::setprecision(12);
        unsigned cases = 0;
        for (std::size_t m : {63, 31}) for (const auto nk : {std::pair<std::size_t, std::size_t>{2560, 960},
                    {960, 2560}, {4864, 896}, {896, 4864}}) for (unsigned copies : {1, 4}) {
            shape(record, Sizes{m, nk.first, nk.second}, copies, runs, warmup); ++cases;
        }
        record << "{\"type\":\"summary\",\"complete\":true,\"cases\":" << cases
               << ",\"timing_qualified\":false,\"acceptance_timing\":false}" << std::endl;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
