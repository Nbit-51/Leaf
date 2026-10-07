#pragma once
// Explicit diagnostic builds only. No counters or scopes in release builds.
#ifndef _WIN32
#error Windows diagnostics require Windows
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace leaf::diagnostics {
struct Stamp { std::uint64_t cpu_100ns, cycles; };
inline Stamp stamp() {
    FILETIME created, exited, kernel, user;
    ULONG64 cycles;
    if (!GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user) ||
        !QueryThreadCycleTime(GetCurrentThread(), &cycles))
        throw std::runtime_error("Cannot read diagnostic thread counters");
    auto value = [](FILETIME t) { return (std::uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime; };
    return {value(kernel) + value(user), cycles};
}
struct Samples {
    std::vector<double> cpu_ms;
    std::vector<std::uint64_t> cycles;
    void add(Stamp before, Stamp after) {
        cpu_ms.push_back(double(after.cpu_100ns - before.cpu_100ns) / 10000);
        cycles.push_back(after.cycles - before.cycles);
    }
};
enum Part : unsigned { MlpTotal, OtherTotal, MlpPack, OtherPack, Parts };
struct Total { double ms = 0; std::size_t calls = 0; };
inline std::array<std::array<Total, Parts>, 2> linear{};
class Scope {
    using Clock = std::chrono::steady_clock;
    Total& total;
    Clock::time_point start = Clock::now();
public:
    Scope(std::size_t tokens, Part part) : total(linear[tokens == 1 ? 1 : 0][part]) {}
    ~Scope() {
        total.ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        ++total.calls;
    }
};
template<class T> inline void values(std::ostream& out, const std::vector<T>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
    out << ']';
}
inline void emit(std::ostream& out, const Samples& prefill, const Samples& decode) {
    out << ",\"diagnostic_timing_build\":true,\"thread_cpu_prefill_ms\":";
    values(out, prefill.cpu_ms);
    out << ",\"thread_cpu_decode_ms\":"; values(out, decode.cpu_ms);
    out << ",\"thread_cycles_prefill\":"; values(out, prefill.cycles);
    out << ",\"thread_cycles_decode\":"; values(out, decode.cycles);
    out << ",\"linear_diagnostic_includes_warmup\":true,\"linear_diagnostic\":{";
    constexpr const char* names[] = {"mlp_total", "other_total", "mlp_pack", "other_pack"};
    for (unsigned bucket = 0; bucket < 2; ++bucket) {
        out << (bucket ? ",\"decode\":{" : "\"prefill\":{");
        for (unsigned part = 0; part < Parts; ++part)
            out << (part ? "," : "") << '"' << names[part] << "\":{\"ms\":" << linear[bucket][part].ms
                << ",\"calls\":" << linear[bucket][part].calls << '}';
        out << '}';
    }
    out << '}';
}
} // namespace leaf::diagnostics
