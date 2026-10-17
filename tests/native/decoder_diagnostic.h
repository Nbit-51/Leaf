#pragma once
// Explicit diagnostic builds only. No counters or scopes in release builds.
#ifndef _WIN32
#error Windows diagnostics require Windows
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace leaf::diagnostics {
inline thread_local HANDLE dispatch_handle = nullptr;
class ThreadDispatch {
public:
    ThreadDispatch() {
        const auto status = EnableThreadProfiling(GetCurrentThread(), THREAD_PROFILING_FLAG_DISPATCH,
                                                  0, &dispatch_handle);
        if (status != ERROR_SUCCESS)
            throw std::runtime_error("Cannot enable diagnostic thread dispatch counters: " + std::to_string(status));
    }
    ThreadDispatch(const ThreadDispatch&) = delete;
    ThreadDispatch& operator=(const ThreadDispatch&) = delete;
    ~ThreadDispatch() { DisableThreadProfiling(dispatch_handle); dispatch_handle = nullptr; }
};
struct Stamp {
    std::uint64_t cpu_100ns, cycles, wait_reasons;
    DWORD page_faults, logical_cpu, context_switches;
};
inline Stamp stamp() {
    FILETIME created, exited, kernel, user;
    ULONG64 cycles;
    PROCESS_MEMORY_COUNTERS memory{};
    memory.cb = sizeof(memory);
    PERFORMANCE_DATA dispatch{};
    dispatch.Size = sizeof(dispatch);
    dispatch.Version = PERFORMANCE_DATA_VERSION;
    if (!GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user) ||
        !QueryThreadCycleTime(GetCurrentThread(), &cycles) ||
        !GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory)))
        throw std::runtime_error("Cannot read diagnostic thread counters");
    const auto status = ReadThreadProfilingData(dispatch_handle, READ_THREAD_PROFILING_FLAG_DISPATCHING, &dispatch);
    if (status != ERROR_SUCCESS)
        throw std::runtime_error("Cannot read diagnostic dispatch counters: " + std::to_string(status));
    auto value = [](FILETIME t) { return (std::uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime; };
    return {value(kernel) + value(user), cycles, dispatch.WaitReasonBitMap,
            memory.PageFaultCount, GetCurrentProcessorNumber(), dispatch.ContextSwitchCount};
}
struct Samples {
    std::vector<double> cpu_ms;
    std::vector<std::uint64_t> cycles;
    std::vector<DWORD> page_faults;
    std::vector<DWORD> logical_cpu_start, logical_cpu_end;
    std::vector<DWORD> context_switches;
    std::vector<std::uint64_t> wait_reasons;
    void reserve(std::size_t count) {
        cpu_ms.reserve(count); cycles.reserve(count); page_faults.reserve(count);
        logical_cpu_start.reserve(count); logical_cpu_end.reserve(count);
        context_switches.reserve(count); wait_reasons.reserve(count);
    }
    void add(Stamp before, Stamp after) {
        cpu_ms.push_back(double(after.cpu_100ns - before.cpu_100ns) / 10000);
        cycles.push_back(after.cycles - before.cycles);
        // Includes soft and hard faults across the process; not a hard-fault
        // counter. DWORD subtraction also handles one counter wrap.
        page_faults.push_back(after.page_faults - before.page_faults);
        logical_cpu_start.push_back(before.logical_cpu);
        logical_cpu_end.push_back(after.logical_cpu);
        context_switches.push_back(after.context_switches - before.context_switches);
        wait_reasons.push_back(after.wait_reasons);
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
    out << ",\"diagnostic_counter_version\":4,\"process_page_faults_prefill\":";
    values(out, prefill.page_faults);
    out << ",\"process_page_faults_decode\":"; values(out, decode.page_faults);
    out << ",\"logical_cpu_start_prefill\":"; values(out, prefill.logical_cpu_start);
    out << ",\"logical_cpu_end_prefill\":"; values(out, prefill.logical_cpu_end);
    out << ",\"logical_cpu_start_decode\":"; values(out, decode.logical_cpu_start);
    out << ",\"logical_cpu_end_decode\":"; values(out, decode.logical_cpu_end);
    out << ",\"thread_context_switches_prefill\":"; values(out, prefill.context_switches);
    out << ",\"thread_context_switches_decode\":"; values(out, decode.context_switches);
    out << ",\"thread_wait_reason_mask_prefill\":"; values(out, prefill.wait_reasons);
    out << ",\"thread_wait_reason_mask_decode\":"; values(out, decode.wait_reasons);
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
