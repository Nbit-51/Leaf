#include "leaf/kernels/attention_vector.h"
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <cstdint>

#if defined(__x86_64__) && defined(__GNUC__)
// The previously shipped vector kernel. Keep an independent libm reference;
// the vector exponential is checked separately to at most one FP32 ULP.
__attribute__((target("avx2,no-fma"))) void previous_attend(
        const float* query, const float* key, const float* value, const float* mask, float* output,
        std::size_t batch, std::size_t heads, std::size_t kv_heads, std::size_t queries,
        std::size_t keys, std::size_t stride, std::size_t dim, const std::size_t* shape,
        float scale, bool valid_nonzero) {
    std::vector<float> scores(keys);
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t h = 0; h < heads; ++h) {
            const auto kv = h / (heads / kv_heads);
            const auto* key_head = key + (b * kv_heads + kv) * stride * dim;
            const auto* value_head = value + (b * kv_heads + kv) * stride * dim;
            for (std::size_t q = 0; q < queries; ++q) {
                const auto* x = query + ((b * heads + h) * queries + q) * dim;
                auto* y = output + ((b * queries + q) * heads + h) * dim;
                std::fill_n(y, dim, 0.0f);
                const auto mb = shape[0] == 1 ? 0 : b, mh = shape[1] == 1 ? 0 : h;
                const auto mq = shape[2] == 1 ? 0 : q;
                const auto* mask_row = mask + ((mb * shape[1] + mh) * shape[2] + mq) * shape[3];
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t k = 0; k < keys; ++k) {
                    if ((mask_row[shape[3] == 1 ? 0 : k] != 0.0f) != valid_nonzero) {
                        scores[k] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    const auto* w = key_head + k * dim;
                    auto sum = _mm256_setzero_ps();
                    std::size_t d = 0;
                    for (; dim - d >= 8; d += 8)
                        sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(x + d), _mm256_loadu_ps(w + d)));
                    alignas(32) float lanes[8];
                    _mm256_store_ps(lanes, sum);
                    float dot = 0;
                    for (float lane : lanes) dot += lane;
                    for (; d < dim; ++d) dot += x[d] * w[d];
                    scores[k] = dot * scale * scale;
                    maximum = std::max(maximum, scores[k]);
                }
                if (!std::isfinite(maximum)) continue;
                float denominator = 0;
                for (std::size_t k = 0; k < keys; ++k) {
                    scores[k] = scores[k] == -std::numeric_limits<float>::infinity()
                        ? 0.0f : std::exp(scores[k] - maximum);
                    denominator += scores[k];
                }
                for (std::size_t k = 0; k < keys; ++k) {
                    const float probability = scores[k] / denominator;
                    const auto p = _mm256_set1_ps(probability);
                    const auto* v = value_head + k * dim;
                    std::size_t d = 0;
                    for (; dim - d >= 8; d += 8)
                        _mm256_storeu_ps(y + d, _mm256_add_ps(_mm256_loadu_ps(y + d),
                            _mm256_mul_ps(p, _mm256_loadu_ps(v + d))));
                    for (; d < dim; ++d) y[d] += probability * v[d];
                }
            }
        }
}
#endif

int main() {
    try {
        std::mt19937 random(425);
        std::uniform_real_distribution<float> draw(-3.0f, 3.0f);
#if defined(__x86_64__) && defined(__GNUC__)
        if (__builtin_cpu_supports("avx2")) {
            std::uniform_real_distribution<float> arguments(-105.0f, 0.0f);
            std::uint32_t max_ulp = 0;
            for (unsigned batch = 0; batch < 4096; ++batch) {
                float values[257], expected[257];
                for (unsigned i = 0; i < 257; ++i) {
                    values[i] = arguments(random);
                    expected[i] = std::exp(values[i]);
                }
                leaf::kernels::attention_vector_detail::exp_scores(values, 257, 0.0f);
                for (unsigned i = 0; i < 257; ++i) {
                    std::uint32_t a, b;
                    std::memcpy(&a, values + i, sizeof(a));
                    std::memcpy(&b, expected + i, sizeof(b));
                    const auto difference = a > b ? a - b : b - a;
                    max_ulp = std::max(max_ulp, difference);
                    if (difference > 1) throw std::runtime_error("softmax exp exceeds one FP32 ULP");
                }
            }
            for (float argument : {-INFINITY, INFINITY, NAN, -104.0f, -80.0f, -0.0f, 0.0f, 1.0f}) {
                float values[4] = {argument, argument, argument, argument};
                leaf::kernels::attention_vector_detail::exp_scores(values, 4, 0.0f);
                const float expected = std::exp(argument);
                for (float value : values)
                    if (std::isnan(expected) ? !std::isnan(value) : value != expected)
                        throw std::runtime_error("softmax exp special-value behavior changed");
            }
            std::cout << "Softmax exp maximum FP32 ULP error: " << max_ulp << '\n';
        }
#endif
        for (std::size_t dim : {1, 7, 8, 15, 16, 24, 32, 40, 48, 56, 63, 64, 65, 72, 128, 136}) {
          for (std::size_t keys : {1, 3, 4, 5, 19, 32}) {
            for (std::size_t queries : {1, 5, 17}) {
                constexpr std::size_t batch=2, heads=4, kv=2, stride=32;
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
#if defined(__x86_64__) && defined(__GNUC__)
                        if (__builtin_cpu_supports("avx2")) {
                            std::vector<float> previous(q.size());
                            previous_attend(q.data(),k.data(),v.data(),mask.data(),previous.data(),
                                batch,heads,kv,queries,keys,stride,dim,shape,.5f,polarity);
                            for (std::size_t i = 0; i < previous.size(); ++i)
                                if (std::abs(previous[i] - actual[i + 1]) > 1e-6f + 1e-6f * std::abs(previous[i]))
                                    throw std::runtime_error("vector attention differs from the previous kernel beyond tolerance");
                        }
#endif
                    }
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
