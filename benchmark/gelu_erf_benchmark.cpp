// Isolated activation timing, including the same input copy on both paths.
#include "leaf/kernels/gelu_erf.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

volatile float gelu_erf_sink = 0;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    try {
        if (argc != 2 || !leaf::kernels::gelu_erf_vector_available())
            throw std::runtime_error("Require output JSONL path and supported MinGW AVX2 runtime");
#ifdef _WIN32
        if (GetPriorityClass(GetCurrentProcess()) != ABOVE_NORMAL_PRIORITY_CLASS)
            throw std::runtime_error("Require Above Normal child priority");
#endif
        std::ofstream output(argv[1]);
        if (!output) throw std::runtime_error("Cannot write samples");
        output << std::setprecision(10);
        constexpr unsigned runs = 31, warmup = 10, repeats = 8;
        struct Case { std::size_t size; float deviation; bool uniform; };
        for (const auto& c : {Case{63*2048,0.5f,false}, Case{63*2048,1.0f,false},
                             Case{63*2048,2.0f,false}, Case{127*3072,1.0f,false},
                             Case{63*2048,6.0f,true}}) {
            std::mt19937 random(15711);
            std::normal_distribution<float> normal(0.0f,c.deviation);
            std::uniform_real_distribution<float> uniform(-c.deviation,c.deviation);
            std::vector<float> input(c.size), workspace(c.size), expected(c.size);
            std::size_t vector_lanes = 0;
            for (auto& value : input) {
                value = c.uniform ? uniform(random) : normal(random);
                const auto a = std::abs(value * 0.7071067811865475f);
                vector_lanes += a >= 0x1p-28f && a < 1.25f;
            }
            expected = input; workspace = input;
            leaf::kernels::gelu_erf_inplace(expected.data(),c.size,false);
            leaf::kernels::gelu_erf_inplace(workspace.data(),c.size,true);
            if (std::memcmp(expected.data(), workspace.data(), c.size*sizeof(float)))
                throw std::runtime_error("Microbenchmark output differs");
            output << "{\"elements\":" << c.size << ",\"distribution\":\""
                   << (c.uniform ? "uniform" : "normal") << "\",\"scale\":" << c.deviation
                   << ",\"vector_fraction\":" << double(vector_lanes)/c.size
                   << ",\"bit_exact\":true,\"copy_included\":true,\"repeats_per_sample\":" << repeats
                   << ",\"runs_per_pass\":" << runs << ",\"warmup_per_pass\":" << warmup << ",\"pairs\":[";
            for (unsigned pair=0; pair<6; ++pair) {
                output << (pair ? ",{" : "{") << "\"order\":["
                       << (pair%2 ? "\"after\",\"before\"" : "\"before\",\"after\"") << "]";
                for (unsigned order=0; order<2; ++order) {
                    const bool candidate = (pair%2) != (order==1);
                    output << ",\"" << (candidate ? "after" : "before") << "_ms\":[";
                    for (unsigned sample=0; sample<warmup+runs; ++sample) {
                        const auto start=Clock::now();
                        for (unsigned repeat=0; repeat<repeats; ++repeat) {
                            std::copy(input.begin(),input.end(),workspace.begin());
                            leaf::kernels::gelu_erf_inplace(workspace.data(),c.size,candidate);
                            gelu_erf_sink = workspace[(sample+repeat)%c.size];
                        }
                        const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-start).count()/repeats;
                        if (sample>=warmup) output << (sample>warmup ? "," : "") << elapsed;
                    }
                    output << "]";
                }
                output << "}";
            }
            output << "]}\n" << std::flush;
            if (!output) throw std::runtime_error("Sample write failed");
            std::cout << "Complete " << c.size << ' ' << c.deviation << ' ' << c.uniform << std::endl;
        }
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
