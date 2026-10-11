// Matched token-ID -> normalized embedding timing. Build once per backend.
// Model loading, tokenization and output-file I/O are deliberately outside timing.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef LEAF_COMPARE_LLAMA
#include "llama.h"
#include "ggml-backend.h"
#else
#include "leaf/runtime/embedding.h"
#endif

static std::uint32_t u32(std::istream& stream) {
    unsigned char b[4];
    if (!stream.read(reinterpret_cast<char*>(b), 4)) throw std::runtime_error("Truncated request");
    return std::uint32_t(b[0]) | (std::uint32_t(b[1]) << 8) |
           (std::uint32_t(b[2]) << 16) | (std::uint32_t(b[3]) << 24);
}

int main(int argc, char** argv) {
    try {
        if (argc != 8) throw std::runtime_error("MODEL REQUEST OUTPUT THREADS WARMUPS RUNS BACKEND_DIR");
        const int threads = std::stoi(argv[4]), warmups = std::stoi(argv[5]), runs = std::stoi(argv[6]);
        if (threads < 1 || warmups < 0 || runs < 1) throw std::runtime_error("Invalid counts");
        std::ifstream input(argv[2], std::ios::binary);
        char magic[8];
        if (!input.read(magic, 8) || std::string(magic, 8) != "LEAFER01") throw std::runtime_error("Bad request");
        const auto count = u32(input);
        if (!count || count > 10000) throw std::runtime_error("Bad count");
        std::vector<std::vector<std::uint32_t>> sequences(count);
        for (auto& sequence : sequences) {
            const auto n = u32(input);
            if (!n || n > 2048) throw std::runtime_error("Bad length");
            sequence.resize(n);
            for (auto& id : sequence) id = u32(input);
        }
#ifdef LEAF_COMPARE_LLAMA
        ggml_backend_load_all_from_path(argv[7]);
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        auto* model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("llama model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = cp.n_batch = cp.n_ubatch = 2048;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.embeddings = true;
        cp.offload_kqv = cp.op_offload = false;
        cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.pooling_type = LLAMA_POOLING_TYPE_MEAN;
        cp.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
        auto* ctx = llama_init_from_model(model, cp);
        if (!ctx || llama_model_n_embd_out(model) != 768) throw std::runtime_error("llama context/dimension mismatch");
        auto batch = llama_batch_init(2048, 0, 1);
        auto encode = [&](const std::vector<std::uint32_t>& ids) {
            if (auto memory = llama_get_memory(ctx)) llama_memory_clear(memory, true);
            batch.n_tokens = static_cast<int>(ids.size());
            for (int i = 0; i < batch.n_tokens; ++i) {
                batch.token[i] = ids[i]; batch.pos[i] = i;
                batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = 1;
            }
            if (llama_decode(ctx, batch) != 0) throw std::runtime_error("llama execution failed");
            const auto* ptr = llama_get_embeddings_seq(ctx, 0);
            if (!ptr) throw std::runtime_error("Missing pooled embedding");
            std::vector<float> result(ptr, ptr + 768);
            double square = 0;
            for (auto value : result) square += double(value) * value;
            if (!(square > 0) || !std::isfinite(square)) throw std::runtime_error("Invalid embedding norm");
            const float scale = 1.0 / std::sqrt(square);
            for (auto& value : result) value *= scale;
            return result;
        };
#else
        leaf::runtime::EmbeddingEncoder model(argv[1], threads, false);
        auto encode = [&](const std::vector<std::uint32_t>& ids) { return model.encode(ids); };
#endif
        std::ofstream output(argv[3], std::ios::binary);
        if (!output) throw std::runtime_error("Cannot open output");
        std::cout << "{\"samples_ms\":[";
        for (std::size_t index = 0; index < sequences.size(); ++index) {
            if (index) std::cout << ',';
            std::cout << '[';
            std::vector<float> result;
            for (int i = -warmups; i < runs; ++i) {
                const auto begin = std::chrono::steady_clock::now();
                result = encode(sequences[index]);
                const auto end = std::chrono::steady_clock::now();
                if (i >= 0) {
                    if (i) std::cout << ',';
                    std::cout << std::chrono::duration<double, std::milli>(end - begin).count();
                }
            }
            std::cout << ']' << std::flush;
            output.write(reinterpret_cast<const char*>(result.data()), result.size() * sizeof(float));
        }
        output.close();
        if (!output) throw std::runtime_error("Embedding write failed");
        std::cout << "]}" << std::endl;
#ifdef LEAF_COMPARE_LLAMA
        llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
