#include "tree_sum.h"
#include "aligned_workers.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

using Clock=std::chrono::steady_clock;
volatile float consumed_queue=0;
void require_queue(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
void queue_power() {
    SYSTEM_POWER_STATUS power{};
    require_queue(GetSystemPowerStatus(&power) && power.ACLineStatus==1,"AC missing during queue timing");
}

void queue_shape(std::size_t tokens,std::size_t columns,std::size_t rows,bool timing) {
    constexpr float guard=123456;
    const std::size_t ws=columns+3,ys=rows+7;
    std::vector<float> input(tokens*columns),weights(rows*ws+4,guard),bias(rows);
    std::vector<float> baseline(tokens*ys+4,guard),candidate(baseline),reference(baseline);
    for(std::size_t i=0;i<input.size();++i) input[i]=(int(i*17%257)-128)/128.0f;
    for(std::size_t row=0;row<rows;++row) {
        bias[row]=float(row%7)/11;
        for(std::size_t col=0;col<columns;++col) weights[2+row*ws+col]=(int((row*columns+col)*13%251)-125)/1024.0f;
    }
    const auto saved_input=input,saved_weights=weights;
    leaf::kernels::TokenPanelsF32 packed; packed.pack(input.data(),tokens,columns,columns);
    leaf::runtime::detail::Workers before(4);
    leaf::experiments::AlignedWorkers after(4);
    auto compute=[&](auto& workers,std::vector<float>& result,bool avx) {
        workers.run(rows,[&](std::size_t first,std::size_t last) {
            leaf::experiments::gemm_tree_sum(weights.data()+2+first*ws,last-first,columns,ws,
                packed,result.data()+2+first,ys,bias.data()+first,avx);
        });
    };
    for(bool avx : {false,true}) {
        leaf::experiments::gemm_tree_sum(weights.data()+2,rows,columns,ws,packed,reference.data()+2,ys,bias.data(),avx);
        compute(before,baseline,avx); compute(after,candidate,avx);
        require_queue(std::memcmp(reference.data(),candidate.data(),reference.size()*4)==0,"Aligned queue changed output bits");
        require_queue(std::memcmp(reference.data(),baseline.data(),reference.size()*4)==0,"Baseline queue changed output bits");
    }
    for(std::size_t token=0;token<tokens;++token) {
        for(std::size_t row=0;row<rows;++row) require_queue(std::isfinite(candidate[2+token*ys+row]),"Nonfinite output");
        for(std::size_t row=rows;row<ys;++row) require_queue(candidate[2+token*ys+row]==guard,"Output padding overwritten");
    }
    require_queue(candidate[0]==guard && candidate[1]==guard && candidate[candidate.size()-1]==guard && candidate[candidate.size()-2]==guard,"Output guard overwritten");
    require_queue(input==saved_input && weights==saved_weights,"Inputs modified");
    std::cout<<"{\"tokens\":"<<tokens<<",\"columns\":"<<columns<<",\"rows\":"<<rows
             <<",\"bit_exact\":true,\"scalar_bit_exact\":true,\"passes\":[";
    if(timing) for(unsigned pass=0;pass<12;++pass) {
        const bool changed=pass%4==1 || pass%4==2;
        std::cout<<(pass?",":"")<<"{\"stage\":\""<<(changed?"after":"before")<<"\",\"samples_ms\":[";
        for(unsigned run=0;run<41;++run) {
            queue_power(); const auto start=Clock::now();
            packed.pack(input.data(),tokens,columns,columns);
            if(changed) compute(after,candidate,true); else compute(before,candidate,true);
            const auto ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            queue_power(); consumed_queue=candidate[2+run%rows];
            if(run>=10) std::cout<<(run>10?",":"")<<ms;
        }
        std::cout<<"]}";
    }
    std::cout<<"]}";
}

int main(int argc,char** argv) {
    try {
        require_queue(leaf::kernels::token_panel_f32_avx2_available(),"AVX2/FMA required");
        const bool timing=argc>1 && std::string(argv[1])=="timing";
        if(timing) {
            require_queue(SetProcessAffinityMask(GetCurrentProcess(),(1ull<<2)|(1ull<<4)|(1ull<<6)|(1ull<<8)),"Cannot apply four-core affinity");
            require_queue(SetPriorityClass(GetCurrentProcess(),ABOVE_NORMAL_PRIORITY_CLASS),"Cannot set priority");
        }
        std::cout<<std::setprecision(12)<<"{\"synthetic\":true,\"whole_model_speed_claim\":false,\"packing_included\":true,\"threads\":4,\"shapes\":[";
        bool comma=false;
        auto run=[&](std::size_t m,std::size_t k,std::size_t n,bool timed) {
            if(comma) std::cout<<','; comma=true; queue_shape(m,k,n,timed);
        };
        if(!timing) for(auto m : {1,7,16,17,63}) for(auto k : {15,255,256,257})
            for(auto n : {1,31,32,33,95,96,97,383,384,385,767,768,769,1535,1536,1537}) run(m,k,n,false);
        run(63,960,2560,timing); run(63,2560,960,timing);
        run(63,896,4864,timing); run(63,4864,896,timing);
        std::cout<<"],\"passed\":true}\n";
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
