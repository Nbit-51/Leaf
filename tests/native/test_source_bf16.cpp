#include "leaf/kernels/source_bf16.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

void require(bool passed, const char* message) {
    if (!passed) throw std::runtime_error(message);
}

namespace {
static_assert(sizeof(float) == sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559,
              "Source BF16 storage requires IEEE binary32");
constexpr std::uint32_t expansion_guard = 0xffc0deadu;
std::size_t expansion_values_checked = 0, expansion_calls_checked = 0;

std::uint32_t raw_bits(const float& value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

void store_bits(float& value, std::uint32_t bits) {
    std::memcpy(&value, &bits, sizeof(bits));
}

// Only integer operations construct or compare expanded values. This includes
// signaling NaNs: no floating comparison or arithmetic may quiet their payloads.
void check_expansion(std::vector<std::uint16_t>& source, std::vector<float>& output,
                     std::size_t count, std::size_t source_offset, std::size_t output_offset) {
    require(source_offset + count <= source.size() && output_offset + count <= output.size(),
            "Expansion test extent is invalid");
    const auto saved = source;
    for (unsigned policy = 0; policy < 3; ++policy) {
        for (auto& value : output) store_bits(value, expansion_guard);
        const auto* input = source.data() + source_offset;
        auto* destination = output.data() + output_offset;
        if (policy == 0) leaf::kernels::expand_source_bf16(input, destination, count);
        else leaf::kernels::expand_source_bf16(input, destination, count, policy == 2);
        for (std::size_t i = 0; i < output.size(); ++i) {
            const auto expected = i >= output_offset && i - output_offset < count ?
                std::uint32_t(source[source_offset + i - output_offset]) << 16 : expansion_guard;
            require(raw_bits(output[i]) == expected,
                    "Expansion changed source bits, crossed a guard or modified an inactive output tail");
            ++expansion_values_checked;
        }
        require(source == saved, "Expansion modified caller source storage or source guards");
        ++expansion_calls_checked;
    }
}

void check_expansion_case(std::size_t count, std::size_t source_offset, std::size_t output_offset) {
    std::vector<std::uint16_t> source(source_offset + count + 8, 0xa55a);
    std::vector<float> output(std::max<std::size_t>(64, output_offset + count + 16));
    // Multiplication by an odd integer permutes all uint16 patterns for count
    // 65536, covering both signs, zeros, subnormals, infinities and NaN payloads.
    for (std::size_t i = 0; i < count; ++i)
        source[source_offset + i] = static_cast<std::uint16_t>(i * 40503u + 17u);
    check_expansion(source, output, count, source_offset, output_offset);
}

void check_dispatched_expansion() {
    // Zero values must not dereference either pointer or perform null arithmetic.
    leaf::kernels::expand_source_bf16(nullptr, nullptr, 0);
    leaf::kernels::expand_source_bf16(nullptr, nullptr, 0, false);
    leaf::kernels::expand_source_bf16(nullptr, nullptr, 0, true);
    expansion_calls_checked += 3;
    check_expansion_case(65536, 0, 0);
    check_expansion_case(65536, 1, 3);
    for (std::size_t count = 0; count <= 33; ++count)
        for (std::size_t source_offset = 0; source_offset < 8; ++source_offset)
            for (std::size_t output_offset = 0; output_offset < 8; ++output_offset)
                check_expansion_case(count, source_offset, output_offset);
    for (std::size_t count : {5376u, 5760u, 15360u, 29184u}) {
        check_expansion_case(count, 0, 0);
        check_expansion_case(count, 1, 3);
    }
    // Reuse the same constructed source and high-water destination through
    // growth/shrink sequences. Every byte outside the active range stays guarded.
    std::vector<std::uint16_t> source(65536 + 9);
    std::vector<float> output(65536 + 19);
    for (std::size_t i = 0; i < source.size(); ++i)
        source[i] = static_cast<std::uint16_t>(i * 40503u + 29u);
    for (std::size_t count : {65536u, 0u, 1u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u,
                             5376u, 5760u, 15360u, 29184u})
        check_expansion(source, output, count, 1, 3);
}
} // namespace

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
        check_dispatched_expansion();
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
        std::cout << "Source BF16 correctness passed: exhaustive storage bits, dispatched expansion, paired GEMV tails and guards; "
                  << expansion_values_checked << " expansion value/guard checks, "
                  << expansion_calls_checked << " expansion calls\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
