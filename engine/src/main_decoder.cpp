#include "leaf/runtime/decoder.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::uint32_t read_u32(std::istream& stream) {
    unsigned char bytes[4];
    if (!stream.read(reinterpret_cast<char*>(bytes), 4)) throw std::runtime_error("truncated token request");
    return std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8) |
           (std::uint32_t(bytes[2]) << 16) | (std::uint32_t(bytes[3]) << 24);
}
double median(std::vector<double> values) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const auto mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) * 0.5;
}
std::uint64_t peak_rss() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS info{};
    return GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info)) ? info.PeakWorkingSetSize : 0;
#else
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage)) return 0;
#ifdef __APPLE__
    return usage.ru_maxrss;
#else
    return std::uint64_t(usage.ru_maxrss) * 1024;
#endif
#endif
}
template <class T> void array(std::ostream& output, const std::vector<T>& values) {
    output << '[';
    for (std::size_t i = 0; i < values.size(); ++i) output << (i ? "," : "") << values[i];
    output << ']';
}
}

int main(int argc, char** argv) {
    if (argc < 6 || argc > 13) {
        std::cerr << "Usage: leaf_decoder artifact.leaf tokens.bin logits.bin metrics.json mode"
                     " [threads=1] [runs=3] [warmup=1] [generate=16] [scalar=0] [chunk=1] [activation-bits=32]\n";
        return 1;
    }
    try {
        const std::string mode = argv[5];
        if (mode != "verify" && mode != "chunked" && mode != "bench" && mode != "generate")
            throw std::runtime_error("unknown decoder mode");
        const unsigned threads = argc > 6 ? std::stoul(argv[6]) : 1;
        const unsigned runs = argc > 7 ? std::stoul(argv[7]) : 3;
        const unsigned warmup = argc > 8 ? std::stoul(argv[8]) : 1;
        const unsigned generate = argc > 9 ? std::stoul(argv[9]) : 16;
        const bool scalar = argc > 10 && std::stoi(argv[10]) != 0;
        const unsigned chunk = argc > 11 ? std::stoul(argv[11]) : 1;
        const unsigned activation_bits = argc > 12 ? std::stoul(argv[12]) : 32;
        if (!runs || !chunk) throw std::runtime_error("runs and chunk must be positive");
        std::ifstream input(argv[2], std::ios::binary);
        if (!input) throw std::runtime_error("cannot open token request");
        const auto count = read_u32(input);
        if (count == 0 || count > 100000) throw std::runtime_error("invalid request count");
        std::vector<std::vector<std::uint32_t>> requests;
        for (unsigned i = 0; i < count; ++i) {
            const auto length = read_u32(input);
            if (!length || length > 1048576) throw std::runtime_error("invalid token count");
            std::vector<std::uint32_t> ids(length);
            for (auto& id : ids) id = read_u32(input);
            requests.push_back(std::move(ids));
        }
        if (input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing token request bytes");
        const auto load_start = Clock::now();
        leaf::runtime::Decoder decoder(argv[1], threads, scalar, activation_bits);
        const double load_ms = elapsed(load_start);
        std::ofstream outputs(argv[3], std::ios::binary);
        if (!outputs) throw std::runtime_error("cannot open logits output");
        std::vector<double> prefills, decodes;
        std::vector<std::uint32_t> generated;
        for (const auto& ids : requests) {
            decoder.reset();
            if (mode == "verify" || mode == "chunked") {
                const auto step = mode == "verify" ? ids.size() : std::size_t(chunk);
                for (std::size_t pos = 0; pos < ids.size(); pos += step) {
                    std::vector<std::uint32_t> next(ids.begin() + pos, ids.begin() + std::min(ids.size(), pos + step));
                    const auto logits = decoder.forward(next, true);
                    outputs.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
                }
            } else if (mode == "bench") {
                if (ids.size() < 2) throw std::runtime_error("benchmark requires a prefix and final token");
                const std::vector<std::uint32_t> prefix(ids.begin(), ids.end() - 1), last{ids.back()};
                for (unsigned iteration = 0; iteration < runs + warmup; ++iteration) {
                    decoder.reset(); auto start = Clock::now(); decoder.forward(prefix); const double prefill = elapsed(start);
                    start = Clock::now(); decoder.forward(last); const double decode = elapsed(start);
                    if (iteration >= warmup) { prefills.push_back(prefill); decodes.push_back(decode); }
                }
            } else {
                auto start = Clock::now(); auto logits = decoder.forward(ids); prefills.push_back(elapsed(start));
                for (unsigned token = 0; token < generate; ++token) {
                    const auto id = static_cast<std::uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
                    generated.push_back(id);
                    std::cout << "TOKEN " << id << '\n' << std::flush;
                    if (id == decoder.eos_token()) break;
                    if (token + 1 < generate) {
                        start = Clock::now(); logits = decoder.forward({id}); decodes.push_back(elapsed(start));
                    }
                }
            }
        }
        if (!outputs) throw std::runtime_error("failed to write logits");
        std::ofstream metrics(argv[4]);
        if (!metrics) throw std::runtime_error("cannot write metrics");
        metrics.precision(10);
        metrics << "{\"avx2\":" << (decoder.uses_avx2() ? "true" : "false")
#ifdef LEAF_EXPERIMENTAL_VECTOR_GELU
                << ",\"experimental_vector_gelu_build\":true"
#endif
                << ",\"vnni\":" << (decoder.uses_vnni() ? "true" : "false")
                << ",\"activation_bits\":" << decoder.activation_bits()
                << ",\"threads\":" << threads << ",\"artifact_bytes\":" << decoder.weight_bytes()
                << ",\"peak_rss_bytes\":" << peak_rss() << ",\"load_ms\":" << load_ms
                << ",\"prefill_p50_ms\":" << median(prefills) << ",\"decode_p50_ms\":" << median(decodes)
                << ",\"prefill_samples_ms\":"; array(metrics, prefills);
        metrics << ",\"decode_samples_ms\":"; array(metrics, decodes);
        metrics << ",\"generated_tokens\":"; array(metrics, generated);
        metrics << "}\n";
        std::cout << "Leaf decoder completed; AVX2=" << decoder.uses_avx2() << ", peak RSS=" << peak_rss() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Leaf decoder: " << error.what() << '\n'; return 1;
    }
}
