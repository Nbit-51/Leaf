#include "scheduled_tree.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using Gemm = decltype(&leaf::experiments::gemm_tree_sum);
using Clock = std::chrono::steady_clock;
volatile float consumed = 0;
void require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
void power() {
#ifdef _WIN32
    SYSTEM_POWER_STATUS state{};
    require(GetSystemPowerStatus(&state) && state.ACLineStatus==1,"Timing requires verified AC");
#else
    throw std::runtime_error("This timing probe requires Windows");
#endif
}
float random_value(std::uint32_t& state) {
    state^=state<<13; state^=state>>17; state^=state<<5;
    return (static_cast<int>(state%20001)-10000)/10000.0f;
}

void shape(std::size_t tokens,std::size_t columns,std::size_t rows,bool strides,bool timing) {
    const std::size_t xs=columns+(strides?5:0), ws=columns+(strides?3:0), ys=rows+(strides?7:0);
    constexpr float guard=123456.0f;
    std::vector<float> x(tokens*xs+4,guard),w(rows*ws+4,guard),bias(rows);
    std::vector<float> before(tokens*ys+4,guard),after(before);
    std::uint32_t seed=591;
    for (std::size_t t=0;t<tokens;++t) for (std::size_t k=0;k<columns;++k) x[2+t*xs+k]=random_value(seed);
    for (std::size_t r=0;r<rows;++r) {
        bias[r]=random_value(seed)/8;
        for (std::size_t k=0;k<columns;++k) w[2+r*ws+k]=random_value(seed)/32;
    }
    const auto saved_x=x,saved_w=w;
    leaf::kernels::TokenPanelsF32 packed;
    packed.pack(x.data()+2,tokens,columns,xs);
    const Gemm baseline=leaf::experiments::gemm_tree_sum;
    const Gemm candidate=leaf::experiments::gemm_scheduled_sum;
    for (const bool scalar : {false,true}) {
        baseline(w.data()+2,rows,columns,ws,packed,before.data()+2,ys,bias.data(),!scalar);
        candidate(w.data()+2,rows,columns,ws,packed,after.data()+2,ys,bias.data(),!scalar);
        require(std::memcmp(before.data(),after.data(),before.size()*sizeof(float))==0,"Scheduling changed output bits");
        for (std::size_t t=0;t<tokens;++t) {
            for (std::size_t r=0;r<rows;++r) require(std::isfinite(after[2+t*ys+r]),"Nonfinite result");
            for (std::size_t r=rows;r<ys;++r) require(after[2+t*ys+r]==guard,"Output stride guard changed");
        }
        require(after[0]==guard && after[1]==guard && after[after.size()-1]==guard && after[after.size()-2]==guard,"Output guard changed");
    }
    require(x==saved_x && w==saved_w,"Input or weight storage changed");
    std::cout<<"{\"tokens\":"<<tokens<<",\"columns\":"<<columns<<",\"rows\":"<<rows
             <<",\"strided\":"<<(strides?"true":"false")<<",\"bit_exact\":true,\"scalar_bit_exact\":true,\"passes\":[";
    if (timing) for (unsigned pass=0;pass<12;++pass) {
        const bool changed=(pass%4==1 || pass%4==2);
        const Gemm kernel=changed?candidate:baseline;
        std::cout<<(pass?",":"")<<"{\"stage\":\""<<(changed?"after":"before")<<"\",\"samples_ms\":[";
        for (unsigned run=0;run<41;++run) {
            power();
            const auto start=Clock::now();
            packed.pack(x.data()+2,tokens,columns,xs);
            kernel(w.data()+2,rows,columns,ws,packed,after.data()+2,ys,bias.data(),true);
            const double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            power(); consumed=after[2+(run*17)%(tokens*ys)];
            if (run>=10) std::cout<<(run>10?",":"")<<ms;
        }
        std::cout<<"]}";
    }
    std::cout<<"]}";
}

int main(int argc,char** argv) {
    try {
        require(leaf::kernels::token_panel_f32_avx2_available(),"AVX2/FMA required to validate this experiment");
        const bool timing=argc>1 && std::string(argv[1])=="timing";
#ifdef _WIN32
        if (timing) {
            require(SetPriorityClass(GetCurrentProcess(),ABOVE_NORMAL_PRIORITY_CLASS),"Cannot set probe priority");
            require(GetPriorityClass(GetCurrentProcess())==ABOVE_NORMAL_PRIORITY_CLASS,"Probe priority not verified");
        }
#endif
        std::cout<<std::setprecision(12)<<"{\"synthetic\":true,\"whole_model_speed_claim\":false,\"packing_included\":true,\"shapes\":[";
        bool comma=false;
        auto run=[&](std::size_t m,std::size_t k,std::size_t n,bool strided,bool timed) {
            if (comma) std::cout<<','; comma=true; shape(m,k,n,strided,timed);
        };
        if (!timing) for (auto m : {1,7,8,9,15,16,17,31,63,64})
            for (auto k : {1,15,255,256,257,896,960,2560,4864})
                for (auto n : {1,5,6,7,13}) run(m,k,n,true,false);
        run(63,960,2560,false,timing); run(63,2560,960,false,timing);
        run(63,896,4864,false,timing); run(63,4864,896,false,timing);
        std::cout<<"],\"passed\":true}\n";
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
