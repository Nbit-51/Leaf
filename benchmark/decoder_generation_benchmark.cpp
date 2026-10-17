// Matched generation with an explicit cached prompt prefix.
// This adapter links the actual decoder; it is not an alternate inference path.
#include "leaf/runtime/decoder.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

void check_benchmark_power(bool required) {
    if (!required) return;
#ifdef _WIN32
    SYSTEM_POWER_STATUS status{};
    if (!GetSystemPowerStatus(&status) || status.ACLineStatus != 1)
        throw std::runtime_error("AC power lost or unknown: benchmark sample invalid; timing stopped");
#else
    throw std::runtime_error("This benchmark's AC guard requires Windows");
#endif
}

using Clock=std::chrono::steady_clock;
std::uint32_t read_u32(std::istream& input) {
    unsigned char b[4];
    if (!input.read(reinterpret_cast<char*>(b),4)) throw std::runtime_error("Truncated request");
    return std::uint32_t(b[0])|(std::uint32_t(b[1])<<8)|(std::uint32_t(b[2])<<16)|(std::uint32_t(b[3])<<24);
}
template<class T> void array(std::ostream& output,const std::vector<T>& values) {
    output << '[';
    for (std::size_t i=0;i<values.size();++i) output << (i ? "," : "") << values[i];
    output << ']';
}
int main(int argc,char** argv) {
    try {
        if (argc==2 && std::string(argv[1])=="--runtime-info") {
            std::cout << "{\"runtime\":\"leaf-generation-benchmark\",\"benchmark_power_guard\":true}\n";
            return 0;
        }
        if (argc!=7) throw std::runtime_error("ARTIFACT REQUEST OUTPUT MAX_TOKENS QUALITY_0_OR_1 CACHED_PREFIX_TOKENS");
        const auto maximum=std::stoul(argv[4]);
        const bool quality=std::stoi(argv[5])!=0;
        const bool require_ac=!quality && std::getenv("LEAF_BENCHMARK_REQUIRE_AC");
        const auto cached=std::stoul(argv[6]);
        if (cached>1) throw std::runtime_error("This workload supports zero or one cached prefix token");
        if (!maximum || maximum>256) throw std::runtime_error("Invalid generation length");
        std::ifstream input(argv[2],std::ios::binary);
        if (!input) throw std::runtime_error("Cannot read request");
        const auto count=read_u32(input);
        if (!count || count>10000) throw std::runtime_error("Invalid request count");
        std::vector<std::vector<std::uint32_t>> requests;
        for (unsigned sample=0;sample<count;++sample) {
            const auto length=read_u32(input);
            if (length<2 || length>256) throw std::runtime_error("Invalid prefix length");
            std::vector<std::uint32_t> ids(length);
            for (auto& id:ids) id=read_u32(input);
            requests.push_back(std::move(ids));
        }
        if (input.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("Trailing request bytes");
        leaf::runtime::Decoder decoder(argv[1],1,false,32);
        std::ofstream output(argv[3]);
        if (!output) throw std::runtime_error("Cannot write results");
        output << std::setprecision(10);
        for (const auto& ids:requests) {
            check_benchmark_power(require_ac);
            decoder.reset();
            if (cached) decoder.forward({ids[0]}); // outside timer, matching the reported cache reuse
            const std::vector<std::uint32_t> suffix(ids.begin()+cached,ids.end());
            const auto gate_before=decoder.silu_gate_vector_calls();
            auto start=Clock::now();
            auto logits=decoder.forward(suffix);
            const double prefill=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            const auto gate_after_prefill=decoder.silu_gate_vector_calls();
            std::vector<std::uint32_t> generated;
            std::vector<double> logprobs;
            auto select=[&]() {
                const auto id=static_cast<std::uint32_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
                generated.push_back(id);
                if (quality) {
                    double sum=0;
                    const auto maximum=*std::max_element(logits.begin(),logits.end());
                    for (float v:logits) sum+=std::exp(double(v)-maximum);
                    logprobs.push_back(double(logits[id])-maximum-std::log(sum));
                }
            };
            select();
            start=Clock::now();
            while (generated.size()<maximum && !decoder.is_eos_token(generated.back())) {
                logits=decoder.forward({generated.back()});
                select();
            }
            const double decode=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            check_benchmark_power(require_ac);
            if (decoder.cache_tokens()!=ids.size()+generated.size()-1)
                throw std::runtime_error("Cache length differs from generated work");
            output << "{\"input_tokens\":" << ids.size() << ",\"cached_prompt_tokens\":" << cached << ",\"uncached_prompt_tokens\":"
                   << suffix.size() << ",\"cached_decode_forwards\":" << generated.size()-1
                   << ",\"prefill_ms\":" << prefill << ",\"generation_phase_ms\":" << decode
                   << ",\"quality_mode\":" << (quality ? "true" : "false")
                   << ",\"ac_power_checked\":" << (require_ac ? "true" : "false")
                   << ",\"prefill_vector_silu_calls\":" << gate_after_prefill-gate_before
                   << ",\"cached_decode_vector_silu_calls\":" << decoder.silu_gate_vector_calls()-gate_after_prefill
                   << ",\"generated_tokens\":";
            array(output,generated);
            if (quality) { output << ",\"selected_logprobs\":"; array(output,logprobs); }
            output << "}\n" << std::flush;
        }
        if (!output) throw std::runtime_error("Incomplete result file");
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
