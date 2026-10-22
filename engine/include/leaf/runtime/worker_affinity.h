#pragma once
// Optional placement of owned inference threads. Never changes a process mask
// or system power settings; CPU numbers are supplied by the caller.
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace leaf::runtime::detail {
constexpr unsigned unbound_cpu = std::numeric_limits<unsigned>::max();
inline constexpr bool worker_affinity_supported() noexcept {
#if defined(_WIN32) || defined(__linux__)
    return true;
#else
    return false;
#endif
}
inline void affinity_check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(std::string("Leaf worker affinity: ") + message);
}
inline std::vector<unsigned> allowed_worker_cpus() {
    std::vector<unsigned> cpus;
#ifdef _WIN32
    GROUP_AFFINITY thread{};
    DWORD_PTR process = 0, system = 0;
    affinity_check(GetThreadGroupAffinity(GetCurrentThread(), &thread) && thread.Group == 0 &&
                   GetProcessAffinityMask(GetCurrentProcess(), &process, &system),
                   "cannot query allowed CPUs (only processor group zero is supported)");
    const auto mask = thread.Mask & process;
    for (unsigned cpu=0; cpu<sizeof(DWORD_PTR)*8; ++cpu)
        if (mask & (DWORD_PTR(1) << cpu)) cpus.push_back(cpu);
#elif defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    affinity_check(pthread_getaffinity_np(pthread_self(), sizeof(allowed), &allowed) == 0,
                   "cannot query allowed CPUs");
    for (unsigned cpu=0; cpu<CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &allowed)) cpus.push_back(cpu);
#else
    affinity_check(false, "explicit placement is unsupported on this operating system");
#endif
    return cpus;
}
inline void validate_worker_cpus(const std::vector<unsigned>& cpus, unsigned threads) {
    if (cpus.empty()) return;
    affinity_check(cpus.size() == threads, "provide exactly one distinct CPU per inference thread");
    const auto allowed = allowed_worker_cpus();
    for (std::size_t i=0; i<cpus.size(); ++i) {
        affinity_check(std::find(cpus.begin(), cpus.begin()+i, cpus[i]) == cpus.begin()+i,
                       "duplicate CPU");
        affinity_check(std::find(allowed.begin(), allowed.end(), cpus[i]) != allowed.end(),
                       "requested CPU is outside the caller's allowed set");
    }
}

class AffinityScope {
public:
    explicit AffinityScope(unsigned cpu = unbound_cpu) {
        if (cpu == unbound_cpu) return;
        try {
#ifdef _WIN32
            GROUP_AFFINITY before{};
            affinity_check(cpu < sizeof(DWORD_PTR)*8 &&
                           GetThreadGroupAffinity(GetCurrentThread(), &before) && before.Group == 0 &&
                           (before.Mask & (DWORD_PTR(1) << cpu)), "CPU is outside thread affinity");
            if (before.Mask == (DWORD_PTR(1) << cpu)) return;
            previous_ = SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << cpu);
            affinity_check(previous_ != 0, "cannot bind thread");
            active_ = true;
            GROUP_AFFINITY actual{};
            affinity_check(GetThreadGroupAffinity(GetCurrentThread(), &actual) &&
                           actual.Group == 0 && actual.Mask == (DWORD_PTR(1) << cpu),
                           "thread placement verification failed");
#elif defined(__linux__)
            CPU_ZERO(&previous_);
            affinity_check(cpu < CPU_SETSIZE &&
                           pthread_getaffinity_np(pthread_self(), sizeof(previous_), &previous_) == 0 &&
                           CPU_ISSET(cpu, &previous_), "CPU is outside thread affinity");
            cpu_set_t target, actual;
            CPU_ZERO(&target); CPU_SET(cpu, &target); CPU_ZERO(&actual);
            if (CPU_EQUAL(&previous_, &target)) return;
            affinity_check(pthread_setaffinity_np(pthread_self(), sizeof(target), &target) == 0,
                           "cannot bind thread");
            active_ = true;
            affinity_check(pthread_getaffinity_np(pthread_self(), sizeof(actual), &actual) == 0 &&
                           CPU_EQUAL(&actual, &target), "thread placement verification failed");
#else
            affinity_check(false, "explicit placement is unsupported on this operating system");
#endif
        } catch (...) {
            try { restore(); } catch (...) {}
            throw;
        }
    }
    AffinityScope(const AffinityScope&) = delete;
    AffinityScope& operator=(const AffinityScope&) = delete;
    ~AffinityScope() noexcept { try { restore(); } catch (...) {} }
    // Explicit restoration reports failures to the caller. The destructor is
    // the fallback on exceptional exits and during owned-worker termination.
    void restore() {
        if (!active_) return;
#ifdef _WIN32
        affinity_check(SetThreadAffinityMask(GetCurrentThread(), previous_) != 0,
                       "cannot restore caller affinity");
#elif defined(__linux__)
        affinity_check(pthread_setaffinity_np(pthread_self(), sizeof(previous_), &previous_) == 0,
                       "cannot restore caller affinity");
#endif
        active_ = false;
    }
private:
    bool active_ = false;
#ifdef _WIN32
    DWORD_PTR previous_ = 0;
#elif defined(__linux__)
    cpu_set_t previous_{};
#endif
};
} // namespace leaf::runtime::detail
