// Diagnostic only: same immutable source and output addresses for both paths.
// Compile -O3 without global AVX flags; run pinned under the external power guard.
#include "leaf/kernels/source_bf16.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;
using Expand = void (*)(const std::uint16_t*, float*, std::size_t);
volatile std::uint64_t expand_sink = 0;
constexpr std::uint32_t guard_bits = 0x7fc12345u;

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#define NOINLINE __attribute__((noinline, noclone))
NOINLINE static void baseline(const std::uint16_t* source, float* output, std::size_t count) {
    leaf::kernels::expand_source_bf16(source, output, count);
}
NOINLINE static void vector_expand(const std::uint16_t* source, float* output, std::size_t count) {
    leaf::kernels::expand_source_bf16(source, output, count, true);
}
static void barrier() { asm volatile("" ::: "memory"); }
#else
static void baseline(const std::uint16_t* s, float* d, std::size_t n) {
    leaf::kernels::expand_source_bf16(s, d, n);
}
static void vector_expand(const std::uint16_t*, float*, std::size_t) {
    throw std::runtime_error("This diagnostic requires GCC-compatible x86 AVX2/FMA");
}
static void barrier() {}
#endif

static std::uint32_t bits(const float& value) {
    std::uint32_t result; std::memcpy(&result, &value, sizeof(result)); return result;
}
static void fill_guards(std::vector<float>& output) {
    for (auto& value : output) std::memcpy(&value, &guard_bits, sizeof(value));
}
static std::uint64_t checksum(const float* output, std::size_t count) {
    return std::uint64_t(bits(output[0])) + bits(output[count / 2]) + bits(output[count - 1]);
}
static std::size_t verify_case(std::size_t count, std::size_t source_offset, std::size_t output_offset) {
    const auto sb = 8 + source_offset, ob = 8 + output_offset;
    std::vector<std::uint16_t> source(sb + count + 16, 0x5a5a);
    for (std::size_t i = 0; i < count; ++i) source[sb + i] = static_cast<std::uint16_t>(i);
    const auto saved = source;
    std::vector<float> output(ob + count + 16);
    for (auto expand : {baseline, vector_expand}) {
        fill_guards(output);
        expand(source.data() + sb, output.data() + ob, count);
        if (source != saved) throw std::runtime_error("Expansion changed source words");
        for (std::size_t i = 0; i < output.size(); ++i) {
            const auto expected = i >= ob && i < ob + count ? std::uint32_t(source[sb + i - ob]) << 16 : guard_bits;
            if (bits(output[i]) != expected)
                throw std::runtime_error("Expansion changed bits, tail or output guard");
        }
    }
    return 2 * count;
}
static std::size_t verify() {
    std::size_t checked = 0;
    // Every BF16 pattern, including both NaN classes, sign payloads and subnormals.
    checked += verify_case(65536, 0, 0);
    checked += verify_case(65536, 7, 7);
    for (std::size_t so = 0; so < 8; ++so)
        for (std::size_t oo = 0; oo < 8; ++oo)
            for (std::size_t n : {1u, 2u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u, 127u, 129u})
                checked += verify_case(n, so, oo);
    return checked;
}
static void array(std::ostream& out, const std::vector<double>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
    out << ']';
}
static void run_case(std::ostream& out, std::size_t rows, std::size_t columns,
                     unsigned rounds, unsigned warmup, std::size_t source_offset, std::size_t output_offset) {
    const auto count = rows * columns;
    // Zero offsets preserve the vector allocator's production-style alignment;
    // offsets 1/3 separately exercise unaligned source/output. Both paths share addresses.
    std::vector<std::uint16_t> source(count + 16);
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = static_cast<std::uint16_t>(i * 113 + 7);
    const auto saved = source;
    std::vector<float> output(count + 16); fill_guards(output);
    auto* destination = output.data() + output_offset;
    const auto* input = source.data() + source_offset;
    const auto repeats = std::max<std::size_t>(1, 4194304 / count);
    auto measure = [&](Expand expand) {
        const auto start = Clock::now();
        for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
            expand(input, destination, count);
            barrier();
        }
        const auto ns = std::chrono::duration<double, std::nano>(Clock::now() - start).count();
        expand_sink = checksum(destination, count); // Integer bit reads outside timing.
        return ns;
    };
    for (unsigned i = 0; i < warmup; ++i) { measure(baseline); measure(vector_expand); }
    std::array<std::vector<double>, 2> samples;
    std::vector<unsigned> order;
    for (unsigned round = 0; round < rounds; ++round) {
        for (unsigned phase : {0u, 1u, 1u, 0u}) {
            if (round % 2) phase = 1 - phase;
            samples[phase].push_back(measure(phase ? vector_expand : baseline));
            order.push_back(phase);
        }
    }
    if (source != saved) throw std::runtime_error("Timed source changed");
    for (std::size_t i = 0; i < output.size(); ++i) {
        const auto expected = i >= output_offset && i < count + output_offset ?
            std::uint32_t(input[i - output_offset]) << 16 : guard_bits;
        if (bits(output[i]) != expected) throw std::runtime_error("Timed output/guard mismatch");
    }
    out << "{\"type\":\"case\",\"complete\":true,\"rows\":" << rows << ",\"columns\":" << columns
        << ",\"values\":" << count << ",\"source_bytes\":" << count * 2 << ",\"output_bytes\":" << count * 4
        << ",\"rounds\":" << rounds << ",\"warmup_pairs\":" << warmup << ",\"repeats_per_sample\":" << repeats
        << ",\"cache_policy\":\"same_reused_buffers\",\"source_offset_values\":" << source_offset
        << ",\"output_offset_values\":" << output_offset
        << ",\"layout\":\"" << (source_offset == 0 && output_offset == 0 ? "production_offsets" : "unaligned_offsets")
        << "\",\"source_mod64\":" << reinterpret_cast<std::uintptr_t>(input) % 64
        << ",\"output_mod64\":" << reinterpret_cast<std::uintptr_t>(destination) % 64
        << ",\"baseline\":\"current_expand_source_bf16_baseline_ISA\",\"candidate\":\"production_AVX2_dispatch\""
        << ",\"source_immutable\":true,\"bits_and_guards_exact\":true,\"sample_order\":[";
    for (std::size_t i = 0; i < order.size(); ++i) out << (i ? "," : "") << order[i];
    out << "],\"before_total_ns\":"; array(out, samples[0]);
    out << ",\"after_total_ns\":"; array(out, samples[1]);
    out << ",\"before_ms_per_expansion\":";
    for (auto& v : samples[0]) v /= repeats * 1e6;
    array(out, samples[0]); out << ",\"after_ms_per_expansion\":";
    for (auto& v : samples[1]) v /= repeats * 1e6;
    array(out, samples[1]); out << "}" << std::endl;
    std::cout << count << " values offsets=" << source_offset << '/' << output_offset << " complete\n" << std::flush;
}
int main(int argc, char** argv) {
    try {
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
        if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma"))
            throw std::runtime_error("Requires AVX2/FMA");
#else
        throw std::runtime_error("Requires GCC-compatible x86 AVX2/FMA");
#endif
        if (argc < 2 || argc > 4) throw std::runtime_error("OUTPUT.jsonl [ROUNDS=15] [WARMUP_PAIRS=3]");
        const unsigned rounds = argc > 2 ? std::stoul(argv[2]) : 15, warmup = argc > 3 ? std::stoul(argv[3]) : 3;
        if (!rounds || rounds > 1000 || warmup > 1000) throw std::runtime_error("Invalid rounds/warmup");
        std::ifstream existing(argv[1]);
        if (existing.good()) throw std::runtime_error("Use a fresh output file");
        std::ofstream out(argv[1]);
        if (!out) throw std::runtime_error("Cannot create output");
        out.exceptions(std::ios::badbit | std::ios::failbit); out << std::setprecision(12);
        const auto checked = verify();
        out << "{\"type\":\"correctness\",\"passed\":true,\"values_checked\":" << checked
            << ",\"all_bf16_patterns\":true,\"offsets_tails_guards\":true,\"floating_arithmetic_on_values\":false}" << std::endl;
        unsigned cases = 0;
        for (const auto offsets : {std::pair<std::size_t, std::size_t>{0, 0}, {1, 3}}) {
            for (std::size_t rows : {6u, 96u}) for (std::size_t k : {896u, 960u, 2560u, 4864u}) {
                run_case(out, rows, k, rounds, warmup, offsets.first, offsets.second); ++cases;
            }
            run_case(out, 1, 1048576, rounds, warmup, offsets.first, offsets.second); ++cases;
        }
        out << "{\"type\":\"summary\",\"complete\":true,\"cases\":" << cases
            << ",\"timing_qualified\":false,\"whole_model_speed_claim\":false}" << std::endl;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
