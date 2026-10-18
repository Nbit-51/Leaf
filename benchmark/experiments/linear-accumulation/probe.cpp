#include "blocked_sum.h"
#ifdef LEAF_ACCUMULATION_TREE_OUTER
#include "tree_sum.h"
#endif
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

using Gemm = decltype(&leaf::kernels::gemm_token_panels_f32);
using Clock = std::chrono::steady_clock;
volatile double checksum = 0;
#ifdef LEAF_ACCUMULATION_TREE_OUTER
constexpr Gemm candidate_gemm = &leaf::experiments::gemm_tree_sum;
constexpr auto candidate_name = "6x16_fp32_tree";
#else
constexpr Gemm candidate_gemm = &leaf::experiments::gemm_blocked_sum;
constexpr auto candidate_name = "6x16_k128_fp64_outer_sum";
#endif


struct Error {
    double max_abs = 0, square = 0, max_ratio = 0;
    std::size_t count = 0, violations = 0;
    void add(float value, double expected) {
        const double delta = std::abs(static_cast<double>(value) - expected);
        const double tolerance = 2e-4 + 2e-4 * std::abs(expected);
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite output");
        max_abs = std::max(max_abs, delta);
        square += delta * delta;
        max_ratio = std::max(max_ratio, delta / tolerance);
        violations += delta > tolerance;
        ++count;
    }
    void print() const {
        std::cout << "{\"max_abs_error\":" << max_abs << ",\"rmse\":" << std::sqrt(square/count)
                  << ",\"max_tolerance_ratio\":" << max_ratio << ",\"violations\":" << violations << '}';
    }
};

float random_value(std::uint32_t& seed) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return static_cast<float>(static_cast<int>(seed % 20001) - 10000) / 10000.0f;
}

void shape(std::size_t tokens, std::size_t columns, std::size_t rows, bool strided, bool with_bias, bool timing) {
    const auto xs = columns + (strided ? 5 : 0), ws = columns + (strided ? 3 : 0), ys = rows + (strided ? 7 : 0);
    constexpr float sentinel = 123456.0f;
    std::vector<float> input(tokens*xs, sentinel), weights(rows*ws, sentinel), bias(rows);
    std::vector<float> baseline(tokens*ys, sentinel), candidate(tokens*ys, sentinel);
    std::uint32_t seed = 0x672613af;
    for (std::size_t t=0;t<tokens;++t) for (std::size_t k=0;k<columns;++k) input[t*xs+k]=random_value(seed);
    for (std::size_t r=0;r<rows;++r) {
        bias[r]=random_value(seed)/8;
        for (std::size_t k=0;k<columns;++k) weights[r*ws+k]=random_value(seed)/32;
    }
    const auto saved_input = input, saved_weights = weights;
    leaf::kernels::TokenPanelsF32 panels;
    panels.pack(input.data(), tokens, columns, xs);
    const auto* b = with_bias ? bias.data() : nullptr;
    leaf::kernels::gemm_token_panels_f32_row_reuse(weights.data(),rows,columns,ws,panels,baseline.data(),ys,b);
    candidate_gemm(weights.data(),rows,columns,ws,panels,candidate.data(),ys,b,true);
#ifdef LEAF_ACCUMULATION_TREE_CACHE
    std::vector<float> tree_reference(tokens*ys, sentinel);
    leaf::experiments::tree_sum_detail::gemm(weights.data(), rows, columns, ws, panels,
                                            tree_reference.data(), ys, b);
    if (std::memcmp(tree_reference.data(), candidate.data(), candidate.size()*sizeof(float)))
        throw std::runtime_error("Cache traversal changed tree arithmetic");
#endif
    Error base_error, split_error;
    for (std::size_t t=0;t<tokens;++t) {
        for (std::size_t r=0;r<rows;++r) {
            double expected = with_bias ? bias[r] : 0;
            for (std::size_t k=0;k<columns;++k) expected += static_cast<double>(input[t*xs+k])*weights[r*ws+k];
            base_error.add(baseline[t*ys+r],expected);
            split_error.add(candidate[t*ys+r],expected);
        }
        for (std::size_t r=rows;r<ys;++r)
            if (candidate[t*ys+r]!=sentinel || baseline[t*ys+r]!=sentinel) throw std::runtime_error("output stride overwritten");
    }
    if (input != saved_input || weights != saved_weights) throw std::runtime_error("input mutated");
    if (split_error.violations || base_error.violations) throw std::runtime_error("precision check failed");
    std::vector<float> scalar_base(tokens*ys,sentinel), scalar_split(tokens*ys,sentinel);
    leaf::kernels::gemm_token_panels_f32_row_reuse(weights.data(),rows,columns,ws,panels,scalar_base.data(),ys,b,false);
    candidate_gemm(weights.data(),rows,columns,ws,panels,scalar_split.data(),ys,b,false);
    if (std::memcmp(scalar_base.data(),scalar_split.data(),scalar_base.size()*sizeof(float))) throw std::runtime_error("scalar fallback changed");
    std::cout << "{\"m\":"<<tokens<<",\"k\":"<<columns<<",\"n\":"<<rows
              <<",\"strided\":"<<(strided?"true":"false")<<",\"bias\":"<<(with_bias?"true":"false")
              <<",\"scalar_fallback_bit_exact\":true,\"row_reuse\":";
    base_error.print(); std::cout << ",\"blocked_sum\":"; split_error.print();
    std::cout << ",\"passes\":[";
    if (timing) for(unsigned pass=0;pass<4;++pass) {
        const bool split = pass==1 || pass==2;
        Gemm gemm = split ? candidate_gemm : &leaf::kernels::gemm_token_panels_f32_row_reuse;
        std::cout << (pass?",":"") << "{\"algorithm\":\""<<(split?"blocked_sum":"row_reuse")<<"\",\"ms\":[";
        for(unsigned run=0;run<9;++run) {
            const auto start=Clock::now();
            panels.pack(input.data(),tokens,columns,xs);
            gemm(weights.data(),rows,columns,ws,panels,candidate.data(),ys,b,true);
            const auto ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            checksum=candidate[(run*17)%candidate.size()];
            if(run>=2) std::cout<<(run>2?",":"")<<ms;
        }
        std::cout << "]}";
    }
    std::cout << "]}";
}

int main(int argc,char**argv) {
    try {
        float values[129]; values[0] = 8192.0f;
        std::fill(values + 1, values + 129, 1.0f);
        if (leaf::experiments::rms_square_sum(values, 129) != 67108992.0f)
            throw std::runtime_error("RMS lost small squares");
        const bool timing=argc>1 && std::string(argv[1])=="timing";
        std::cout<<std::setprecision(12)<<"{\"candidate\":\""<<candidate_name<<"\",\"packing_included\":true,\"synthetic\":true,\"model_latency_measured\":false,\"shapes\":[";
        bool comma=false;
        auto run=[&](std::size_t m,std::size_t k,std::size_t n,bool strides,bool bias) {
            if(comma) std::cout<<','; comma=true; shape(m,k,n,strides,bias,timing && !strides);
        };
        if(!timing) {
            run(16,127,6,true,true); run(16,128,6,true,false); run(16,129,6,true,true);
            run(17,255,10,true,true); run(33,257,11,true,false);
            run(1,1,1,true,false); run(7,9,2,true,true); run(8,16,3,true,false);
            run(9,17,4,true,true); run(17,65,7,true,true); run(31,961,11,true,false);
        }
        run(63,960,2560,false,true); run(63,896,4864,false,true);
        std::cout<<"]}\n";
        return 0;
    } catch(const std::exception&e) { std::cerr<<e.what()<<'\n'; return 1; }
}
