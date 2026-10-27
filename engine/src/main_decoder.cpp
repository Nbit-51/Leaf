#include "leaf/runtime/decoder.h"
#include "leaf/runtime/worker_affinity.h"
#include "leaf/runtime/benchmark_counters.h"
#ifdef LEAF_DIAGNOSTIC_TIMING
#include "../../tests/native/decoder_diagnostic.h"
#include <cstdlib>
#include <filesystem>
#include <thread>
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
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
void check_benchmark_power(bool required) {
    if (!required) return;
#ifdef _WIN32
    SYSTEM_POWER_STATUS power{};
    if (!GetSystemPowerStatus(&power) || power.ACLineStatus != 1)
        throw std::runtime_error("AC power missing or unknown; timing stopped");
#else
    throw std::runtime_error("This benchmark AC guard requires Windows");
#endif
}
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::vector<unsigned> parse_worker_cpus(const std::string& text) {
    std::vector<unsigned> cpus;
    std::size_t first = 0;
    do {
        const auto end = text.find(',', first);
        const auto item = text.substr(first, end == std::string::npos ? end : end-first);
        if (item.empty() || item.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("worker CPUs must be comma-separated unsigned indices");
        const auto cpu = std::stoull(item);
        if (cpu > std::numeric_limits<unsigned>::max())
            throw std::runtime_error("worker CPU index is too large");
        cpus.push_back(static_cast<unsigned>(cpu));
        if (end == std::string::npos) break;
        first = end+1;
    } while (true);
    return cpus;
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
    if (argc == 2 && std::string(argv[1]) == "--runtime-info") {
        std::cout << "{\"runtime\":\"leaf_decoder\",\"artifact_versions\":[2,3,4,5],\"source_bf16_head_supported\":true,\"benchmark_power_guard\":true"
                  << ",\"worker_affinity_supported\":" << (leaf::runtime::detail::worker_affinity_supported() ? "true" : "false")
                  << ",\"worker_spin_limit_us\":50"
                  << ",\"fp32_math_modes\":[\"fast\",\"accurate\"],\"fp32_math_default\":\"fast\""
                  << ",\"reusable_logits_supported\":true"
#ifdef LEAF_DIAGNOSTIC_TIMING
                  << ",\"benchmark_counters_supported\":false"
#else
                  << ",\"benchmark_counters_supported\":" << (leaf::runtime::benchmark::supported() ? "true" : "false")
#endif
                  << ",\"benchmark_counter_version\":1"
#ifdef LEAF_DIAGNOSTIC_TIMING
                  << ",\"diagnostic_start_gate_supported\":true,\"diagnostic_counter_version\":4"
#endif
                  << "}\n";
        return 0;
    }
    if (argc < 6 || argc > 16) {
        std::cerr << "Usage: leaf_decoder artifact.leaf tokens.bin logits.bin metrics.json mode"
                     " [threads=1] [runs=3] [warmup=1] [generate=16] [scalar=0] [chunk=1] [activation-bits=32] [worker-cpus=comma-list] [--fp32-math=fast|accurate] [--benchmark-counters]\n";
        return 1;
    }
    try {
#ifdef LEAF_DIAGNOSTIC_TIMING
        // The monitor releases this gate only after affinity, QoS and telemetry
        // setup. Ordinary builds do not read or wait on this environment key.
        if (const char* gate = std::getenv("LEAF_DIAGNOSTIC_START_GATE")) {
            const auto deadline = Clock::now() + std::chrono::seconds(30);
            while (!std::filesystem::exists(gate)) {
                if (Clock::now() >= deadline) throw std::runtime_error("diagnostic start gate timed out");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
#endif
        const std::string mode = argv[5];
        const bool require_ac = mode == "bench" && std::getenv("LEAF_BENCHMARK_REQUIRE_AC");
        if (mode != "verify" && mode != "chunked" && mode != "bench" && mode != "generate")
            throw std::runtime_error("unknown decoder mode");
        const unsigned threads = argc > 6 ? std::stoul(argv[6]) : 1;
        const unsigned runs = argc > 7 ? std::stoul(argv[7]) : 3;
        const unsigned warmup = argc > 8 ? std::stoul(argv[8]) : 1;
        const unsigned generate = argc > 9 ? std::stoul(argv[9]) : 16;
        const bool scalar = argc > 10 && std::stoi(argv[10]) != 0;
        const unsigned chunk = argc > 11 ? std::stoul(argv[11]) : 1;
        const unsigned activation_bits = argc > 12 ? std::stoul(argv[12]) : 32;
        std::vector<unsigned> worker_cpus;
        auto fp32_math = leaf::runtime::FP32Math::Fast;
        bool math_seen = false, cpus_seen = false, benchmark_counters = false;
        for (int i = 13; i < argc; ++i) {
            const std::string value = argv[i];
            if (value.rfind("--fp32-math=", 0) == 0) {
                if (math_seen) throw std::runtime_error("duplicate FP32 math policy");
                math_seen = true;
                const auto policy = value.substr(12);
                if (policy == "accurate") fp32_math = leaf::runtime::FP32Math::Accurate;
                else if (policy != "fast") throw std::runtime_error("FP32 math must be fast or accurate");
            } else if (value == "--benchmark-counters") {
                if (benchmark_counters) throw std::runtime_error("duplicate benchmark counters flag");
                benchmark_counters = true;
            } else {
                if (value.rfind("--", 0) == 0) throw std::runtime_error("unknown decoder option: " + value);
                if (cpus_seen || math_seen || benchmark_counters)
                    throw std::runtime_error("worker CPU list must precede named decoder options");
                cpus_seen = true;
                worker_cpus = parse_worker_cpus(value);
            }
        }
        if (benchmark_counters && mode != "bench")
            throw std::runtime_error("benchmark counters require bench mode");
        if (benchmark_counters && !leaf::runtime::benchmark::supported())
            throw std::runtime_error("requested benchmark counters require Windows");
#ifdef LEAF_DIAGNOSTIC_TIMING
        if (benchmark_counters)
            throw std::runtime_error("benchmark counters cannot be combined with legacy thread-counter builds");
#endif
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
        leaf::runtime::Decoder decoder(argv[1], threads, scalar, activation_bits, worker_cpus, fp32_math);
        const double load_ms = elapsed(load_start);
        std::ofstream outputs(argv[3], std::ios::binary);
        if (!outputs) throw std::runtime_error("cannot open logits output");
        std::vector<double> prefills, decodes;
        std::vector<float> logits;
        leaf::runtime::benchmark::Samples work_prefill, work_decode;
        if (benchmark_counters) {
            work_prefill.reserve(std::size_t(runs) * requests.size());
            work_decode.reserve(std::size_t(runs) * requests.size());
        }
#ifdef LEAF_DIAGNOSTIC_TIMING
        if (threads != 1) throw std::runtime_error("Thread-counter diagnostics require one inference thread");
        leaf::diagnostics::Samples prefill_counters, decode_counters;
        leaf::diagnostics::ThreadDispatch thread_dispatch;
        prefill_counters.reserve(std::size_t(runs) * requests.size());
        decode_counters.reserve(std::size_t(runs) * requests.size());
#endif
        std::vector<std::uint32_t> generated;
        for (const auto& ids : requests) {
            decoder.reset();
            if (mode == "verify" || mode == "chunked") {
                const auto step = mode == "verify" ? ids.size() : std::size_t(chunk);
                for (std::size_t pos = 0; pos < ids.size(); pos += step) {
                    std::vector<std::uint32_t> next(ids.begin() + pos, ids.begin() + std::min(ids.size(), pos + step));
                    decoder.forward_into(next, logits, true);
                    outputs.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
                }
            } else if (mode == "bench") {
                if (ids.size() < 2) throw std::runtime_error("benchmark requires a prefix and final token");
                const std::vector<std::uint32_t> prefix(ids.begin(), ids.end() - 1), last{ids.back()};
                for (unsigned iteration = 0; iteration < runs + warmup; ++iteration) {
                    check_benchmark_power(require_ac);
#ifdef LEAF_DIAGNOSTIC_TIMING
                    decoder.reset();
                    const auto cpu_start = leaf::diagnostics::stamp();
                    auto start = Clock::now(); decoder.forward_into(prefix, logits); const double prefill = elapsed(start);
                    const auto cpu_middle = leaf::diagnostics::stamp();
                    start = Clock::now(); decoder.forward_into(last, logits); const double decode = elapsed(start);
                    const auto cpu_end = leaf::diagnostics::stamp();
                    if (iteration >= warmup) {
                        prefill_counters.add(cpu_start, cpu_middle);
                        decode_counters.add(cpu_middle, cpu_end);
                    }
#else
                    double prefill, decode;
                    if (benchmark_counters) {
                        decoder.reset();
                        const auto before = leaf::runtime::benchmark::stamp();
                        auto start = Clock::now(); decoder.forward_into(prefix, logits); prefill = elapsed(start);
                        const auto middle = leaf::runtime::benchmark::stamp();
                        start = Clock::now(); decoder.forward_into(last, logits); decode = elapsed(start);
                        const auto after = leaf::runtime::benchmark::stamp();
                        if (iteration >= warmup) {
                            work_prefill.add(before, middle); work_decode.add(middle, after);
                        }
                    } else {
                        decoder.reset(); auto start = Clock::now(); decoder.forward_into(prefix, logits); prefill = elapsed(start);
                        start = Clock::now(); decoder.forward_into(last, logits); decode = elapsed(start);
                    }
#endif
                    check_benchmark_power(require_ac);
                    if (iteration >= warmup) { prefills.push_back(prefill); decodes.push_back(decode); }
                }
            } else {
                auto start = Clock::now(); decoder.forward_into(ids, logits); prefills.push_back(elapsed(start));
                for (unsigned token = 0; token < generate; ++token) {
                    const auto id = static_cast<std::uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
                    generated.push_back(id);
                    std::cout << "TOKEN " << id << '\n' << std::flush;
                    if (decoder.is_eos_token(id)) break;
                    if (token + 1 < generate) {
                        start = Clock::now(); decoder.forward_into({id}, logits); decodes.push_back(elapsed(start));
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
#ifdef LEAF_EXPERIMENTAL_GEMV_PAIR
                << ",\"experimental_gemv_pair_build\":true"
#endif
#ifdef LEAF_EXPERIMENTAL_MLP_PACK
                << ",\"experimental_mlp_pack_build\":true"
#endif
#ifdef LEAF_EXPERIMENTAL_ROW_REUSE
                << ",\"experimental_row_reuse_build\":true"
#endif
#ifdef LEAF_EXPERIMENTAL_ATTENTION_AVX2
                << ",\"experimental_attention_avx2_build\":true"
#endif
#ifdef LEAF_EXPERIMENTAL_SILU_GATE
                << ",\"experimental_silu_gate_build\":true"
#endif
                << ",\"vector_silu_gate_calls\":" << decoder.silu_gate_vector_calls()
                << ",\"source_bf16_tensors\":" << decoder.source_bf16_tensors()
                << ",\"benchmark_ac_checked\":" << (require_ac ? "true" : "false")
                << ",\"source_bf16_linear_calls\":" << decoder.source_bf16_linear_calls()
                << ",\"vector_silu_gate_enabled\":" << (decoder.uses_vector_silu_gate() ? "true" : "false")
                << ",\"optimized_fp32_active\":" << (decoder.uses_optimized_fp32() ? "true" : "false")
                << ",\"fp32_math\":\"" << (decoder.uses_accurate_fp32() ? "accurate" : "fast") << '"'
                << ",\"reusable_logits\":true"
                << ",\"vnni\":" << (decoder.uses_vnni() ? "true" : "false")
                << ",\"activation_bits\":" << decoder.activation_bits()
                << ",\"threads\":" << threads << ",\"artifact_bytes\":" << decoder.weight_bytes()
                << ",\"peak_rss_bytes\":" << peak_rss() << ",\"load_ms\":" << load_ms
                << ",\"prefill_p50_ms\":" << median(prefills) << ",\"decode_p50_ms\":" << median(decodes)
                << ",\"prefill_samples_ms\":"; array(metrics, prefills);
        metrics << ",\"decode_samples_ms\":"; array(metrics, decodes);
        metrics << ",\"worker_cpus\":"; array(metrics, worker_cpus);
        metrics << ",\"worker_spin_us\":" << (worker_cpus.size() > 1 ? 50 : 0);
        metrics << ",\"generated_tokens\":"; array(metrics, generated);
        if (benchmark_counters) leaf::runtime::benchmark::emit(metrics, work_prefill, work_decode);
#ifdef LEAF_DIAGNOSTIC_TIMING
        leaf::diagnostics::emit(metrics, prefill_counters, decode_counters);
#endif
        metrics << "}\n";
        std::cout << "Leaf decoder completed; AVX2=" << decoder.uses_avx2() << ", peak RSS=" << peak_rss() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Leaf decoder: " << error.what() << '\n'; return 1;
    }
}
