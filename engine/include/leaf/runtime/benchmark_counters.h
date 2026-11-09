#pragma once

// Optional, ordinary-process Windows accounting. These are diagnostic counters,
// not hardware performance counters or a CPU-frequency measurement. No API is
// called until stamp() is explicitly requested by the benchmark executable.
#include <chrono>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace leaf::runtime::benchmark {
using CounterClock = std::chrono::steady_clock;

inline constexpr bool supported() {
#ifdef _WIN32
    return true;
#else
    return false;
#endif
}

struct Stamp {
    CounterClock::time_point read_begin{}, read_end{};
    std::uint64_t process_cpu_100ns = 0, process_cycles = 0;
    std::uint64_t caller_cpu_100ns = 0, caller_cycles = 0;
    std::uint32_t page_faults = 0, caller_cpu = 0, caller_group = 0;
};

inline Stamp stamp() {
#ifdef _WIN32
    Stamp value;
    value.read_begin = CounterClock::now();
    FILETIME created{}, exited{}, process_kernel{}, process_user{}, caller_kernel{}, caller_user{};
    ULONG64 process_cycles = 0, caller_cycles = 0;
    PROCESS_MEMORY_COUNTERS memory{};
    memory.cb = sizeof(memory);
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &process_kernel, &process_user) ||
        !GetThreadTimes(GetCurrentThread(), &created, &exited, &caller_kernel, &caller_user) ||
        !QueryProcessCycleTime(GetCurrentProcess(), &process_cycles) ||
        !QueryThreadCycleTime(GetCurrentThread(), &caller_cycles) ||
        !GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory)))
        throw std::runtime_error("Cannot read requested benchmark counters; Windows error " +
                                 std::to_string(GetLastError()));
    auto ticks = [](FILETIME time) {
        return (std::uint64_t(time.dwHighDateTime) << 32) | time.dwLowDateTime;
    };
    PROCESSOR_NUMBER processor{};
    GetCurrentProcessorNumberEx(&processor);
    value.process_cpu_100ns = ticks(process_kernel) + ticks(process_user);
    value.caller_cpu_100ns = ticks(caller_kernel) + ticks(caller_user);
    value.process_cycles = process_cycles;
    value.caller_cycles = caller_cycles;
    value.page_faults = memory.PageFaultCount;
    value.caller_cpu = processor.Number;
    value.caller_group = processor.Group;
    value.read_end = CounterClock::now();
    return value;
#else
    throw std::runtime_error("Requested benchmark counters require Windows");
#endif
}

inline std::uint64_t delta(std::uint64_t before, std::uint64_t after) {
    if (after < before) throw std::runtime_error("Benchmark counter moved backwards");
    return after - before;
}

struct Samples {
    std::vector<double> interval_ms, process_cpu_ms, caller_cpu_ms;
    std::vector<std::uint64_t> process_cycles, caller_cycles;
    std::vector<std::uint32_t> process_page_faults;
    std::vector<std::uint32_t> caller_cpu_start, caller_cpu_end, caller_group_start, caller_group_end;

    void reserve(std::size_t count) {
        interval_ms.reserve(count); process_cpu_ms.reserve(count); caller_cpu_ms.reserve(count);
        process_cycles.reserve(count); caller_cycles.reserve(count); process_page_faults.reserve(count);
        caller_cpu_start.reserve(count); caller_cpu_end.reserve(count);
        caller_group_start.reserve(count); caller_group_end.reserve(count);
    }

    void add(const Stamp& before, const Stamp& after) {
        if (before.read_end < before.read_begin || after.read_end < after.read_begin ||
            after.read_begin < before.read_end)
            throw std::runtime_error("Benchmark monotonic counter interval moved backwards");
        // Validate all nonwrapping counters before mutating the sample arrays.
        const auto process_cpu = delta(before.process_cpu_100ns, after.process_cpu_100ns);
        const auto thread_cpu = delta(before.caller_cpu_100ns, after.caller_cpu_100ns);
        const auto process_cycle = delta(before.process_cycles, after.process_cycles);
        const auto thread_cycle = delta(before.caller_cycles, after.caller_cycles);
        // DWORD accounting includes both soft and hard page faults. Modular
        // subtraction preserves a single wrap; it is not a hard-fault count.
        const auto faults = static_cast<std::uint32_t>(after.page_faults - before.page_faults);
        interval_ms.push_back(std::chrono::duration<double, std::milli>(after.read_end - before.read_begin).count());
        process_cpu_ms.push_back(double(process_cpu) / 10000);
        caller_cpu_ms.push_back(double(thread_cpu) / 10000);
        process_cycles.push_back(process_cycle); caller_cycles.push_back(thread_cycle);
        process_page_faults.push_back(faults);
        caller_cpu_start.push_back(before.caller_cpu); caller_cpu_end.push_back(after.caller_cpu);
        caller_group_start.push_back(before.caller_group); caller_group_end.push_back(after.caller_group);
    }

    template<class T> static void field(std::ostream& out, const char* name, const std::vector<T>& values) {
        out << '"' << name << "\":[";
        for (std::size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
        out << ']';
    }

    void emit(std::ostream& out) const {
        out << '{';
        field(out, "interval_ms", interval_ms); out << ',';
        field(out, "process_cpu_ms", process_cpu_ms); out << ',';
        field(out, "process_cycles", process_cycles); out << ',';
        field(out, "process_page_faults", process_page_faults); out << ',';
        field(out, "caller_cpu_ms", caller_cpu_ms); out << ',';
        field(out, "caller_cycles", caller_cycles); out << ',';
        field(out, "caller_cpu_start", caller_cpu_start); out << ',';
        field(out, "caller_cpu_end", caller_cpu_end); out << ',';
        field(out, "caller_group_start", caller_group_start); out << ',';
        field(out, "caller_group_end", caller_group_end);
        out << '}';
    }
};

inline void emit(std::ostream& out, const Samples& prefill, const Samples& decode) {
    out << ",\"benchmark_counters\":{\"version\":1,"
           "\"scope\":\"process includes all workers; caller is benchmark thread\","
           "\"interval\":\"monotonic span includes complete counter reads; adjacent phase brackets overlap; not forward wall sample\","
           "\"prefill\":";
    prefill.emit(out);
    out << ",\"decode\":";
    decode.emit(out);
    out << '}';
}
} // namespace leaf::runtime::benchmark
