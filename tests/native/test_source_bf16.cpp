#include "leaf/kernels/source_bf16.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

void require(bool passed, const char* message) {
    if (!passed) throw std::runtime_error(message);
}

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
__attribute__((target("avx2,fma"))) void check_vectors() {
    alignas(32) float values[8];
    std::uint16_t source[8];
    // Include every sign, exponent and payload pattern, without arithmetic.
    for (unsigned first = 0; first < 65536; first += 8) {
        for (unsigned i = 0; i < 8; ++i) source[i] = static_cast<std::uint16_t>(first+i);
        _mm256_store_ps(values,leaf::kernels::source_bf16_detail::load8(source));
        for (unsigned i = 0; i < 8; ++i) {
            std::uint32_t bits;
            std::memcpy(&bits,values+i,4);
            require(bits == std::uint32_t(source[i])<<16,"Vector expansion changed source bits");
        }
    }
    std::mt19937 random(4321);
    for (std::size_t columns : {8,16,24,32,40,48,56,64,72,896,960}) {
        std::vector<std::uint16_t> compact(2*columns+2,0x1234);
        std::vector<float> full(2*columns+2), input(columns+2);
        for (std::size_t i=0;i<2*columns;++i) {
            compact[i+1] = static_cast<std::uint16_t>(random());
            // Keep finite small weights, including signed zeros/subnormals.
            compact[i+1] &= 0x83ff;
            compact[i+1] |= static_cast<std::uint16_t>((random()%119)<<7);
            full[i+1] = leaf::kernels::source_bf16_value(compact[i+1]);
        }
        for (auto& value:input) value = (static_cast<int>(random()%257)-128)/128.0f;
        const auto saved=compact;
        float before[4]={1234,0,0,5678}, after[4]={1234,0,0,5678};
        leaf::kernels::gemv_pair_f32(full.data()+1,input.data()+1,columns,before+1);
        leaf::kernels::gemv_pair_source_bf16(compact.data()+1,input.data()+1,columns,after+1);
        require(std::memcmp(before,after,sizeof(before))==0,"BF16/F32 paired GEMV changed output bits or guards");
        require(compact==saved,"BF16 GEMV changed weight storage");
    }
}
#endif

int main() {
    try {
        for (unsigned value=0;value<65536;++value) {
            const auto expanded=leaf::kernels::source_bf16_value(static_cast<std::uint16_t>(value));
            std::uint32_t bits;
            std::memcpy(&bits,&expanded,4);
            require(bits==std::uint32_t(value)<<16,"Scalar expansion changed source bits");
        }
        std::uint16_t source[5]={0,0x8000,1,0x3f80,0xbf80};
        float expanded[7]={12,0,0,0,0,0,34};
        leaf::kernels::expand_source_bf16(source,expanded+1,5);
        require(expanded[0]==12 && expanded[6]==34,"Expansion overwrote guards");
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) check_vectors();
#endif
        std::cout << "Source BF16 correctness passed: exhaustive storage bits, vector expansion, paired GEMV tails and guards\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
