#include "leaf/runtime/benchmark_counters.h"
#include <iostream>
#include <limits>
#include <sstream>

using namespace leaf::runtime::benchmark;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void fails(F&& operation) {
    bool failed = false;
    try { operation(); } catch (const std::runtime_error&) { failed = true; }
    require(failed, "Invalid counters did not fail closed");
}
Stamp make_stamp(long milliseconds, std::uint64_t process_cpu, std::uint64_t caller_cpu,
                 std::uint64_t process_cycle, std::uint64_t caller_cycle,
                 std::uint32_t faults, std::uint32_t cpu, std::uint32_t group) {
    Stamp result;
    result.read_begin = CounterClock::time_point{} + std::chrono::milliseconds(milliseconds);
    result.read_end = result.read_begin + std::chrono::microseconds(100);
    result.process_cpu_100ns = process_cpu; result.caller_cpu_100ns = caller_cpu;
    result.process_cycles = process_cycle; result.caller_cycles = caller_cycle;
    result.page_faults = faults; result.caller_cpu = cpu; result.caller_group = group;
    return result;
}
}

int main() {
    try {
        Samples samples;
        samples.reserve(8);
        const auto first = make_stamp(10, 100000, 10000, 100000000000ull, 30000000000ull,
                                      std::numeric_limits<std::uint32_t>::max() - 2, 2, 0);
        auto last = make_stamp(20, 410000, 90000, 100100000000ull, 30020000000ull, 3, 4, 1);
        samples.add(first, last);
        require(samples.interval_ms == std::vector<double>{10.1}, "Incorrect complete read bracket");
        require(samples.process_cpu_ms == std::vector<double>{31}, "Process CPU units or all-worker sum lost");
        require(samples.caller_cpu_ms == std::vector<double>{8}, "Caller CPU units incorrect");
        require(samples.process_cycles == std::vector<std::uint64_t>{100000000}, "Process cycles converted or truncated");
        require(samples.caller_cycles == std::vector<std::uint64_t>{20000000}, "Caller cycle delta incorrect");
        require(samples.process_page_faults == std::vector<std::uint32_t>{6}, "DWORD page-fault wrap lost");
        require(samples.caller_cpu_start[0] == 2 && samples.caller_cpu_end[0] == 4 &&
                samples.caller_group_start[0] == 0 && samples.caller_group_end[0] == 1,
                "Caller CPU/group endpoints incorrect");
        // Short calls may have zero CPU/cycle/fault deltas. CPU time units do
        // not imply 100ns measurement resolution, and are not a validity gate.
        auto unchanged = first;
        unchanged.read_begin = last.read_begin;
        unchanged.read_end = last.read_end;
        samples.add(first, unchanged);
        require(samples.process_cpu_ms.back() == 0 && samples.caller_cpu_ms.back() == 0 &&
                samples.process_cycles.back() == 0 && samples.caller_cycles.back() == 0 &&
                samples.process_page_faults.back() == 0, "Zero short-call deltas rejected");
        for (int field = 0; field < 4; ++field) {
            auto invalid = last;
            if (field == 0) invalid.process_cpu_100ns = first.process_cpu_100ns - 1;
            if (field == 1) invalid.caller_cpu_100ns = first.caller_cpu_100ns - 1;
            if (field == 2) invalid.process_cycles = first.process_cycles - 1;
            if (field == 3) invalid.caller_cycles = first.caller_cycles - 1;
            fails([&] { samples.add(first, invalid); });
            require(samples.interval_ms.size() == 2 && samples.caller_group_end.size() == 2,
                    "Failed counter delta partially appended a sample");
        }
        auto invalid = first;
        invalid.read_end = invalid.read_begin - std::chrono::microseconds(1);
        fails([&] { samples.add(invalid, last); });
        invalid = last;
        invalid.read_end = invalid.read_begin - std::chrono::microseconds(1);
        fails([&] { samples.add(first, invalid); });
        invalid = first;
        fails([&] { samples.add(first, invalid); });
        std::ostringstream json;
        emit(json, samples, samples);
        const auto text = json.str();
        require(text.find("\"benchmark_counters\":{\"version\":1") != std::string::npos &&
                text.find("\"prefill\":{") != std::string::npos &&
                text.find("\"decode\":{") != std::string::npos &&
                text.find("\"process_cycles\":[100000000,0]") != std::string::npos,
                "Counter JSON schema lost raw phase values");
#ifdef _WIN32
        require(supported(), "Windows counters capability missing");
        const auto begin = stamp();
        volatile std::uint64_t sink = 1;
        for (unsigned i = 1; i < 100000; ++i) sink = sink * 6364136223846793005ull + i;
        (void)sink;
        const auto end = stamp();
        Samples live;
        live.add(begin, end);
        require(live.interval_ms[0] > 0, "Live counter bracket is not positive");
#else
        require(!supported(), "Portable build advertised Windows counters");
        fails([] { stamp(); });
#endif
        std::cout << "PASS benchmark counters: units, wrap, zero deltas, endpoints, failures, JSON and platform probe\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
