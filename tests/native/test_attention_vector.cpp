#include "leaf/kernels/attention_vector.h"
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

int main() {
    try {
        std::mt19937 random(425);
        std::uniform_real_distribution<float> draw(-3.0f, 3.0f);
        for (std::size_t dim : {1, 7, 8, 15, 16, 63, 64, 65, 128}) {
            for (std::size_t queries : {1, 5, 17}) {
                constexpr std::size_t batch=2, heads=4, kv=2, keys=19, stride=32;
                std::vector<float> q(batch*heads*queries*dim), k(batch*kv*stride*dim), v(k.size());
                for (auto* x : {&q, &k, &v}) for (auto& f : *x) f=draw(random);
                for (unsigned broadcast=0; broadcast<16; ++broadcast) {
                    const std::size_t shape[4]={broadcast&1 ? 1:batch, broadcast&2 ? 1:heads,
                                               broadcast&4 ? 1:queries, broadcast&8 ? 1:keys};
                    std::vector<float> mask(shape[0]*shape[1]*shape[2]*shape[3]);
                    for (std::size_t i=0;i<mask.size();++i) mask[i]=i%3 ? 1:0;
                    for (bool polarity : {false,true}) {
                        std::vector<float> old(q.size()), scalar(q.size()), actual(q.size()+2,9876.0f);
                        leaf::kernels::attention_f32_gqa_strided(q.data(),k.data(),v.data(),mask.data(),old.data(),
                            batch,heads,kv,queries,keys,stride,dim,shape,.5f,polarity);
                        leaf::kernels::attention_f32_gqa_strided_vector(q.data(),k.data(),v.data(),mask.data(),scalar.data(),
                            batch,heads,kv,queries,keys,stride,dim,shape,.5f,polarity,false);
                        leaf::kernels::attention_f32_gqa_strided_vector(q.data(),k.data(),v.data(),mask.data(),actual.data()+1,
                            batch,heads,kv,queries,keys,stride,dim,shape,.5f,polarity,true);
                        if (actual.front()!=9876 || actual.back()!=9876) throw std::runtime_error("attention guard overwritten");
                        if (std::memcmp(old.data(),scalar.data(),old.size()*sizeof(float))) throw std::runtime_error("scalar fallback changed");
                        for (std::size_t i=0;i<old.size();++i)
                            if (!std::isfinite(actual[i+1]) || std::abs(actual[i+1]-old[i])>3e-5f+3e-5f*std::abs(old[i]))
                                throw std::runtime_error("vector attention exceeds tolerance");
                    }
                }
            }
        }
        // Fully masked rows return zero; masked nonfinite V retains the old
        // 0*NaN behavior when another key is valid. No masked-key pruning.
        float q[8]={}, k[16]={}, v[16]={}, mask[2]={0,0}, y[8];
        const std::size_t shape[4]={1,1,1,2};
        leaf::kernels::attention_f32_gqa_strided_vector(q,k,v,mask,y,1,1,1,1,2,2,8,shape,1);
        for(float f:y) if(f!=0) throw std::runtime_error("fully masked row changed");
        mask[0]=1; v[8]=std::numeric_limits<float>::quiet_NaN();
        leaf::kernels::attention_f32_gqa_strided_vector(q,k,v,mask,y,1,1,1,1,2,2,8,shape,1);
        if(!std::isnan(y[0])) throw std::runtime_error("masked nonfinite value behavior changed");
        std::cout << "Attention vector correctness passed: GQA, broadcast masks, polarity, tails, guards and scalar fallback\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
