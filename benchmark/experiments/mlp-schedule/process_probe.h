#pragma once
// Diagnostic-only process totals, including all inference workers. These are
// deliberately absent from the release binary and are not acceptance timing.
#include <windows.h>
#include <psapi.h>
#include <cstdint>
#include <ostream>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace leaf::experiments {
inline std::mutex process_worker_mutex;
inline std::vector<DWORD> process_workers;
inline void register_process_worker() {
    std::lock_guard<std::mutex> lock(process_worker_mutex);
    process_workers.push_back(GetCurrentThreadId());
}
inline std::vector<DWORD> process_worker_ids() {
    std::lock_guard<std::mutex> lock(process_worker_mutex);
    return process_workers;
}
struct ProcessStamp {
    std::uint64_t cpu_100ns, cycles, faults;
    double unix_seconds;
};
inline std::uint64_t file_ticks(FILETIME t) {
    return (std::uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime;
}
inline ProcessStamp process_stamp() {
    FILETIME created{}, exited{}, kernel{}, user{}, now{};
    ULONG64 cycles = 0;
    PROCESS_MEMORY_COUNTERS memory{};
    const auto process = GetCurrentProcess();
    if (!GetProcessTimes(process, &created, &exited, &kernel, &user) ||
        !QueryProcessCycleTime(process, &cycles) ||
        !GetProcessMemoryInfo(process, &memory, sizeof(memory)))
        throw std::runtime_error("Cannot read process diagnostic counters");
    GetSystemTimePreciseAsFileTime(&now);
    return {file_ticks(kernel) + file_ticks(user), cycles, memory.PageFaultCount,
            double(file_ticks(now) - 116444736000000000ull) / 1e7};
}
struct ProcessSamples {
    std::vector<double> cpu_ms, cycles, faults, starts_unix_seconds, ends_unix_seconds;
    void add(const ProcessStamp& a, const ProcessStamp& b) {
        if (b.cpu_100ns < a.cpu_100ns || b.cycles < a.cycles || b.faults < a.faults ||
            b.unix_seconds < a.unix_seconds)
            throw std::runtime_error("Nonmonotonic process diagnostic counters");
        cpu_ms.push_back(double(b.cpu_100ns - a.cpu_100ns) / 1e4);
        cycles.push_back(double(b.cycles - a.cycles));
        faults.push_back(double(b.faults - a.faults));
        starts_unix_seconds.push_back(a.unix_seconds);
        ends_unix_seconds.push_back(b.unix_seconds);
    }
    void emit(std::ostream& output, const char* phase) const {
        const auto field = [&](const char* name, const auto& values) {
            output << ",\"process_" << name << '_' << phase << "\":[";
            for (std::size_t i=0; i<values.size(); ++i) output << (i ? "," : "") << values[i];
            output << ']';
        };
        field("cpu_ms", cpu_ms); field("cycles", cycles); field("faults", faults);
        // Unix timestamps need more precision than the ordinary latency fields.
        const auto precision = output.precision(17);
        field("starts_unix_seconds", starts_unix_seconds);
        field("ends_unix_seconds", ends_unix_seconds);
        output.precision(precision);
    }
};
} // namespace leaf::experiments
