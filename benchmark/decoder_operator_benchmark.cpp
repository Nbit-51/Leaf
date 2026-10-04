#include "leaf/kernels/attention_vector.h"
#include "leaf/kernels/token_panel.h"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

volatile double observed_operator_checksum=0;
using Clock=std::chrono::steady_clock;

template<class F> void passes(F execute, unsigned runs, unsigned warmup) {
    std::cout<<"\"passes\":[";
    for(unsigned p=0;p<4;++p) {
        const bool candidate=p==1 || p==2;
        std::cout<<(p?",":"")<<"{\"stage\":\""<<(candidate?"optimized":"full_k")<<"\",\"samples_ms\":[";
        for(unsigned i=0;i<runs+warmup;++i) {
            const auto start=Clock::now();
            const auto* result=execute(candidate);
            const auto elapsed=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            observed_operator_checksum+=*result;
            if(i>=warmup) std::cout<<(i>warmup?",":"")<<elapsed;
        }
        std::cout<<"]}";
    }
    std::cout<<"]}";
}

void fill(std::vector<float>& x) {
    for(std::size_t i=0;i<x.size();++i) x[i]=static_cast<float>(static_cast<int>(i%127)-63)/73.0f;
}

void attention(std::size_t tokens, unsigned runs, unsigned warmup) {
    constexpr std::size_t heads=12, dim=64;
    std::vector<float> q(heads*tokens*dim),k(q.size()),v(q.size()),y(q.size()),mask(tokens*tokens);
    fill(q);fill(k);fill(v);
    for(std::size_t t=0;t<tokens;++t) for(std::size_t j=0;j<=t;++j) mask[t*tokens+j]=1;
    const std::size_t shape[4]={1,1,tokens,tokens};
    const float scale=1/std::sqrt(std::sqrt(static_cast<float>(dim)));
    auto execute=[&](bool candidate) {
        if(candidate) leaf::kernels::attention_f32_gqa_strided_vector(q.data(),k.data(),v.data(),mask.data(),y.data(),
            1,heads,heads,tokens,tokens,tokens,dim,shape,scale);
        else leaf::kernels::attention_f32_gqa_strided(q.data(),k.data(),v.data(),mask.data(),y.data(),
            1,heads,heads,tokens,tokens,tokens,dim,shape,scale);
        return y.data();
    };
    execute(false);const auto reference=y;execute(true);
    double error=0;
    for(std::size_t i=0;i<y.size();++i) {
        error=std::max(error,static_cast<double>(std::abs(y[i]-reference[i])));
        if(!std::isfinite(y[i]) || std::abs(y[i]-reference[i])>3e-5f+3e-5f*std::abs(reference[i]))
            throw std::runtime_error("attention mismatch");
    }
    std::cout<<"{\"name\":\"attention_"<<tokens<<"\",\"quality_passed\":true,\"max_abs_error\":"<<error<<',';
    passes(execute,runs,warmup);
}

void qkv(unsigned runs,unsigned warmup) {
    constexpr std::size_t tokens=63,h=768;
    std::vector<float> x(tokens*h),w(3*h*h),y(3*tokens*h);
    fill(x);fill(w);
    leaf::kernels::TokenPanelsF32 panels;
    auto execute=[&](bool candidate) {
        if(candidate) panels.pack(x.data(),tokens,h,h);
        for(std::size_t p=0;p<3;++p) {
            if(!candidate) panels.pack(x.data(),tokens,h,h);
            leaf::kernels::gemm_token_panels_f32(w.data()+p*h*h,h,h,h,panels,y.data()+p*tokens*h,h);
        }
        return y.data();
    };
    execute(false);const auto reference=y;execute(true);
    if(std::memcmp(y.data(),reference.data(),y.size()*sizeof(float))) throw std::runtime_error("QKV packing mismatch");
    std::cout<<"{\"name\":\"shared_qkv_63\",\"quality_passed\":true,\"bit_exact\":true,";
    passes(execute,runs,warmup);
}

int main(int argc,char** argv) {
    try {
        const unsigned runs=argc>1?std::stoul(argv[1]):31, warmup=argc>2?std::stoul(argv[2]):10;
        if(runs<5 || runs>10000 || warmup>10000) throw std::runtime_error("invalid iteration count");
        std::cout<<std::setprecision(10)<<"{\"cases\":[";
        attention(63,runs,warmup);std::cout<<',';attention(128,runs,warmup);std::cout<<',';qkv(runs,warmup);
        std::cout<<"]}\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
