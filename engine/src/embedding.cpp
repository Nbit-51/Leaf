#include "leaf/runtime/embedding.h"
#include "leaf/runtime/host_support.h"
#include "leaf/kernels/row_reuse.h"
#include "leaf/kernels/attention_vector.h"
#include <cmath>
#include <unordered_map>

namespace leaf::runtime {
namespace {
using detail::check;
struct Matrix {
    std::size_t rows, cols;
    const float *data;
};

// Gemma's zero-centred RMSNorm multiplies by (1 + weight), including Q/K.
void norm(const float *x, float *y, std::size_t rows, std::size_t width, const float *weights,
          float epsilon) {
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0;
        for (std::size_t j = 0; j < width; ++j)
            sum += x[row * width + j] * x[row * width + j];
        const float factor = 1.0f / std::sqrt(sum / static_cast<float>(width) + epsilon);
        for (std::size_t j = 0; j < width; ++j)
            y[row * width + j] = (x[row * width + j] * factor) * (1.0f + weights[j]);
    }
}
} // namespace

struct EmbeddingEncoder::Impl {
    detail::Mapping mapping;
    detail::Workers workers;
    std::unordered_map<std::string, Matrix> weights;
    std::size_t h, f, heads, kv, layers, vocab, maximum, dim, radius, dense, output;
    float global_theta, local_theta, epsilon, embedding_scale, attention_scale;
    bool avx;
    std::vector<unsigned> local;
    kernels::TokenPanelsF32 panels;
    std::vector<float> hidden, normalized, q, k, v, qh, kh, vh, attention, projected, gate, up;
    std::vector<float> local_mask, global_mask, cosine, sine;

    Impl(const std::string &path, unsigned threads, bool scalar)
        : mapping(path), workers(threads), avx(!scalar && kernels::token_panel_f32_avx2_available()) {
        detail::Reader r{mapping};
        r.require(8);
        check(std::memcmp(mapping.data, "LEAFEM01", 8) == 0, "unsupported embedding artifact");
        r.position = 8;
        check(r.u32() == 1, "unsupported embedding version");
        auto count = r.u32();
        auto base = r.u64();
        h = r.u32();
        f = r.u32();
        heads = r.u32();
        kv = r.u32();
        layers = r.u32();
        vocab = r.u32();
        maximum = r.u32();
        dim = r.u32();
        radius = r.u32();
        dense = r.u32();
        output = r.u32();
        global_theta = r.f32();
        local_theta = r.f32();
        epsilon = r.f32();
        embedding_scale = r.f32();
        attention_scale = r.f32();
        check(h && f && heads && kv && heads % kv == 0 && layers && vocab && maximum && dim && dim % 2 == 0 &&
                  dense && output && radius < maximum,
              "invalid embedding geometry");
        for (auto value : {global_theta, local_theta, epsilon, embedding_scale, attention_scale})
            check(std::isfinite(value) && value > 0, "invalid embedding numerical constant");
        const auto limit = std::numeric_limits<std::size_t>::max() / sizeof(float);
        check(heads <= limit / dim &&
                  maximum <= limit / std::max({h, f, heads * dim, maximum, dense, output}),
              "embedding workspace overflow");
        check(base % 64 == 0 && base <= mapping.size && count <= mapping.size / 24 && layers <= count / 13 &&
                  count == layers * 13 + 4,
              "invalid embedding tensor table");
        for (std::size_t i = 0; i < layers; ++i) {
            auto flag = r.u32();
            check(flag <= 1, "invalid layer attention kind");
            local.push_back(flag);
        }
        for (std::size_t i = 0; i < count; ++i) {
            auto name = r.string();
            auto rows = r.u32();
            auto cols = r.u32();
            auto offset = r.u64();
            auto bytes = r.u64();
            const std::uint64_t elements = std::uint64_t(rows) * cols;
            check(rows && cols && elements <= limit && bytes == elements * 4 && offset % 64 == 0 &&
                      offset <= mapping.size - base && bytes <= mapping.size - base - offset,
                  "invalid embedding tensor range");
            check(weights
                      .emplace(name, Matrix{rows, cols,
                                            reinterpret_cast<const float *>(mapping.data + base + offset)})
                      .second,
                  "duplicate embedding tensor");
        }
        check(r.position <= base, "embedding metadata overlaps weights");
        require("embed_tokens.weight", vocab, h);
        require("norm.weight", 1, h);
        require("pool.up.weight", dense, h);
        require("pool.down.weight", output, dense);
        for (std::size_t i = 0; i < layers; ++i) {
            auto p = "layers." + std::to_string(i) + ".";
            for (auto n : {"input_layernorm.weight", "post_attention_layernorm.weight",
                           "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"})
                require(p + n, 1, h);
            require(p + "self_attn.q_norm.weight", 1, dim);
            require(p + "self_attn.k_norm.weight", 1, dim);
            require(p + "self_attn.q_proj.weight", heads * dim, h);
            require(p + "self_attn.k_proj.weight", kv * dim, h);
            require(p + "self_attn.v_proj.weight", kv * dim, h);
            require(p + "self_attn.o_proj.weight", h, heads * dim);
            require(p + "mlp.gate_proj.weight", f, h);
            require(p + "mlp.up_proj.weight", f, h);
            require(p + "mlp.down_proj.weight", h, f);
        }
    }
    const Matrix &require(const std::string &name, std::size_t rows, std::size_t cols) const {
        auto found = weights.find(name);
        check(found != weights.end() && found->second.rows == rows && found->second.cols == cols,
              "missing/wrong embedding tensor " + name);
        return found->second;
    }
    void linear(const std::vector<float> &x, std::size_t tokens, const std::string &name,
                std::vector<float> &y) {
        const auto &w = weights.at(name);
        panels.pack(x.data(), tokens, w.cols, w.cols);
        y.resize(tokens * w.rows);
        workers.run(w.rows, [&](std::size_t first, std::size_t last) {
            kernels::gemm_token_panels_f32_row_reuse(w.data + first * w.cols, last - first, w.cols, w.cols,
                                                     panels, y.data() + first, w.rows, nullptr, avx);
        });
    }
    void normalize(const std::vector<float> &x, std::vector<float> &y, std::size_t tokens, std::size_t width,
                   const std::string &name) {
        y.resize(x.size());
        norm(x.data(), y.data(), tokens, width, weights.at(name).data, epsilon);
    }
    void reorder(const std::vector<float> &x, std::vector<float> &y, std::size_t tokens, std::size_t nheads,
                 bool rotate) {
        y.resize(x.size());
        for (std::size_t t = 0; t < tokens; ++t)
            for (std::size_t head = 0; head < nheads; ++head) {
                const auto from = (t * nheads + head) * dim, to = (head * tokens + t) * dim;
                if (!rotate) {
                    std::copy_n(x.data() + from, dim, y.data() + to);
                    continue;
                }
                for (std::size_t j = 0; j < dim / 2; ++j) {
                    auto c = cosine[t * (dim / 2) + j], s = sine[t * (dim / 2) + j];
                    auto a = x[from + j], b = x[from + j + dim / 2];
                    y[to + j] = a * c - b * s;
                    y[to + j + dim / 2] = b * c + a * s;
                }
            }
    }
    std::vector<float> encode(const std::vector<std::uint32_t> &ids) {
        const auto tokens = ids.size();
        check(tokens && tokens <= maximum, "invalid embedding sequence length");
        hidden.resize(tokens * h);
        const auto *embeddings = weights.at("embed_tokens.weight").data;
        for (std::size_t t = 0; t < tokens; ++t) {
            check(ids[t] < vocab, "embedding token outside vocabulary");
            for (std::size_t j = 0; j < h; ++j)
                hidden[t * h + j] = embeddings[std::size_t(ids[t]) * h + j] * embedding_scale;
        }
        local_mask.resize(tokens * tokens);
        global_mask.assign(tokens * tokens, 1.0f);
        for (std::size_t t = 0; t < tokens; ++t)
            for (std::size_t s = 0; s < tokens; ++s)
                local_mask[t * tokens + s] = (t > s ? t - s : s - t) <= radius ? 1.0f : 0.0f;
        cosine.resize(tokens * dim / 2);
        sine.resize(tokens * dim / 2);
        for (std::size_t i = 0; i < layers; ++i) {
            const auto p = "layers." + std::to_string(i) + ".";
            normalize(hidden, normalized, tokens, h, p + "input_layernorm.weight");
            linear(normalized, tokens, p + "self_attn.q_proj.weight", q);
            linear(normalized, tokens, p + "self_attn.k_proj.weight", k);
            linear(normalized, tokens, p + "self_attn.v_proj.weight", v);
            norm(q.data(), q.data(), tokens * heads, dim, weights.at(p + "self_attn.q_norm.weight").data,
                 epsilon);
            norm(k.data(), k.data(), tokens * kv, dim, weights.at(p + "self_attn.k_norm.weight").data,
                 epsilon);
            const float theta = local[i] ? local_theta : global_theta;
            for (std::size_t j = 0; j < dim / 2; ++j) {
                const float frequency =
                    1.0f / std::pow(theta, static_cast<float>(2 * j) / static_cast<float>(dim));
                for (std::size_t t = 0; t < tokens; ++t) {
                    float angle = static_cast<float>(t) * frequency;
                    cosine[t * (dim / 2) + j] = std::cos(angle);
                    sine[t * (dim / 2) + j] = std::sin(angle);
                }
            }
            reorder(q, qh, tokens, heads, true);
            reorder(k, kh, tokens, kv, true);
            reorder(v, vh, tokens, kv, false);
            attention.resize(tokens * heads * dim);
            const std::size_t mask_shape[] = {1, 1, tokens, tokens};
            // Existing attention API squares its scale; preserve model score scaling.
            kernels::attention_f32_gqa_strided_vector(qh.data(), kh.data(), vh.data(),
                                                      (local[i] ? local_mask : global_mask).data(),
                                                      attention.data(), 1, heads, kv, tokens, tokens, tokens,
                                                      dim, mask_shape, std::sqrt(attention_scale), true, avx);
            linear(attention, tokens, p + "self_attn.o_proj.weight", projected);
            normalize(projected, projected, tokens, h, p + "post_attention_layernorm.weight");
            for (std::size_t j = 0; j < hidden.size(); ++j)
                hidden[j] += projected[j];
            normalize(hidden, normalized, tokens, h, p + "pre_feedforward_layernorm.weight");
            linear(normalized, tokens, p + "mlp.gate_proj.weight", gate);
            linear(normalized, tokens, p + "mlp.up_proj.weight", up);
            for (std::size_t j = 0; j < gate.size(); ++j) {
                const float x = gate[j];
                gate[j] = (0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)))) *
                          up[j];
            }
            linear(gate, tokens, p + "mlp.down_proj.weight", projected);
            normalize(projected, projected, tokens, h, p + "post_feedforward_layernorm.weight");
            for (std::size_t j = 0; j < hidden.size(); ++j)
                hidden[j] += projected[j];
        }
        normalize(hidden, normalized, tokens, h, "norm.weight");
        std::vector<float> pooled(h, 0.0f), expanded, result;
        for (std::size_t t = 0; t < tokens; ++t)
            for (std::size_t j = 0; j < h; ++j)
                pooled[j] += normalized[t * h + j];
        for (auto &value : pooled)
            value /= static_cast<float>(tokens);
        linear(pooled, 1, "pool.up.weight", expanded);
        linear(expanded, 1, "pool.down.weight", result);
        double sum = 0;
        for (float value : result) {
            check(std::isfinite(value), "nonfinite embedding output");
            sum += double(value) * value;
        }
        check(sum > 0, "zero embedding output");
        const float length = static_cast<float>(std::sqrt(sum));
        for (auto &value : result)
            value /= length;
        return result;
    }
};
EmbeddingEncoder::EmbeddingEncoder(const std::string &p, unsigned threads, bool scalar)
    : impl_(std::make_unique<Impl>(p, threads, scalar)) {}
EmbeddingEncoder::~EmbeddingEncoder() = default;
std::vector<float> EmbeddingEncoder::encode(const std::vector<std::uint32_t> &ids) {
    return impl_->encode(ids);
}
std::size_t EmbeddingEncoder::dimensions() const { return impl_->output; }
std::size_t EmbeddingEncoder::maximum_tokens() const { return impl_->maximum; }
bool EmbeddingEncoder::uses_avx2() const { return impl_->avx; }
} // namespace leaf::runtime
