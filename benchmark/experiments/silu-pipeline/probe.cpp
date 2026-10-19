#include "leaf/kernels/silu_gate.h"
#include "silu_before.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Clock = std::chrono::steady_clock;
volatile float observed;

void shape(std::size_t n) {
    std::vector<float> input(n + 2, 123), gate(n + 2, 456), before, after;
    for (std::size_t i = 0; i < n; ++i) {
        input[i + 1] = (static_cast<int>(i % 127) - 63) / 17.0f;
        gate[i + 1] = (static_cast<int>((i * 19) % 4093) - 2046) / 128.0f;
    }
    before = input; after = input;
    leaf::before_kernels::silu_gate_inplace(before.data()+1, gate.data()+1, n);
    leaf::kernels::silu_gate_inplace(after.data()+1, gate.data()+1, n);
    if (std::memcmp(before.data(), after.data(), before.size()*sizeof(float)))
        throw std::runtime_error("Pipeline changed ordinary output bits or guards");
    auto old_alias = gate, new_alias = gate;
    leaf::before_kernels::silu_gate_inplace(old_alias.data()+1, old_alias.data()+1, n);
    leaf::kernels::silu_gate_inplace(new_alias.data()+1, new_alias.data()+1, n);
    if (std::memcmp(old_alias.data(), new_alias.data(), old_alias.size()*sizeof(float)))
        throw std::runtime_error("Pipeline changed identical-input/output bits");
    std::cout << "{\"elements\":" << n << ",\"exact_previous\":true,\"exact_alias\":true,\"passes\":[";
    for (unsigned pass = 0; pass < 4; ++pass) {
        const bool candidate = pass == 1 || pass == 2;
        auto kernel = candidate ? &leaf::kernels::silu_gate_inplace : &leaf::before_kernels::silu_gate_inplace;
        std::cout << (pass ? "," : "") << "{\"stage\":\"" << (candidate ? "after" : "before") << "\",\"ms\":[";
        for (unsigned run = 0; run < 41; ++run) {
            after = input;
            const auto start = Clock::now();
            kernel(after.data()+1, gate.data()+1, n, true);
            const auto ms = std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            observed = after[1 + run % n];
            if (run >= 10) std::cout << (run > 10 ? "," : "") << ms;
        }
        std::cout << "]}";
    }
    std::cout << "]}";
}

int main() {
    try {
        if (!SetProcessAffinityMask(GetCurrentProcess(), DWORD_PTR(1) << 2) ||
            !SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS))
            throw std::runtime_error("Cannot set benchmark policy");
        DWORD_PTR affinity, available;
        if (!GetProcessAffinityMask(GetCurrentProcess(), &affinity, &available) || affinity != 4)
            throw std::runtime_error("CPU affinity verification failed");
        std::cout << std::setprecision(12) << "{\"cpu\":2,\"threads\":1,\"synthetic\":true,\"shapes\":[";
        bool comma = false;
        for (std::size_t n : {2560, 4864, 63*2560, 63*4864}) {
            if (comma) std::cout << ',';
            comma = true; shape(n);
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
