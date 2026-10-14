// Adapter for a pinned llama.cpp release. Same request protocol/work as Leaf's
// main_decoder bench: prefix -> last logits, then one teacher-forced KV token.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::uint32_t u32(std::istream& input) {
    unsigned char b[4];
    if (!input.read(reinterpret_cast<char*>(b), 4)) throw std::runtime_error("Truncated tokens");
    return std::uint32_t(b[0]) | (std::uint32_t(b[1]) << 8) |
           (std::uint32_t(b[2]) << 16) | (std::uint32_t(b[3]) << 24);
}
static void array(std::ostream& out, const std::vector<double>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
    out << ']';
}
static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto n = values.size();
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}
int main(int argc, char** argv) {
    try {
        if (argc != 10) throw std::runtime_error("MODEL REQUEST LOGITS METRICS MODE THREADS RUNS WARMUP BACKEND_DIR");
        const std::string mode = argv[5];
        const int threads = std::stoi(argv[6]), runs = std::stoi(argv[7]), warmup = std::stoi(argv[8]);
        if ((mode != "bench" && mode != "verify") || threads < 1 || runs < 1 || warmup < 0)
            throw std::runtime_error("Invalid policy");
        std::ifstream input(argv[2], std::ios::binary);
        const auto count = u32(input);
        if (!count || count > 1000) throw std::runtime_error("Invalid count");
        std::vector<std::vector<std::uint32_t>> sequences(count);
        std::size_t maximum = 0;
        for (auto& row : sequences) {
            const auto n = u32(input);
            if (n < 2 || n > 2048) throw std::runtime_error("Invalid length");
            row.resize(n);
            for (auto& id : row) id = u32(input);
            maximum = std::max(maximum, row.size());
        }
        if (input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Trailing input");
        ggml_backend_load_all_from_path(argv[9]);
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        auto* model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("llama model load failed");
        const auto vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        for (const auto& row : sequences) for (auto id : row)
            if (id >= static_cast<unsigned>(vocab)) throw std::runtime_error("Token outside vocabulary");
        auto cp = llama_context_default_params();
        cp.n_ctx = cp.n_batch = cp.n_ubatch = static_cast<unsigned>(maximum);
        cp.n_threads = cp.n_threads_batch = threads;
        cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.offload_kqv = cp.op_offload = false;
        cp.embeddings = false;
        auto* ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("llama context failed");
        auto batch = llama_batch_init(static_cast<int>(maximum), 0, 1);
        auto prepare = [&](const std::vector<std::uint32_t>& ids, int first, int end, bool all) {
            batch.n_tokens = end - first;
            for (int i = first; i < end; ++i) {
                const int j = i - first;
                batch.token[j] = ids[i]; batch.pos[j] = i;
                batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0;
                batch.logits[j] = all || i + 1 == end;
            }
        };
        std::vector<float> copied(vocab);
        auto execute = [&]() {
            if (llama_decode(ctx, batch)) throw std::runtime_error("llama decode failed");
            llama_synchronize(ctx);
            const auto* logits = llama_get_logits_ith(ctx, -1);
            if (!logits) throw std::runtime_error("Missing logits");
            std::copy_n(logits, vocab, copied.data()); // Both runtimes return host logits.
        };
        std::ofstream output(argv[3], std::ios::binary), metrics(argv[4]);
        if (!output || !metrics) throw std::runtime_error("Cannot open output");
        std::vector<double> prefills, decodes;
        using Clock = std::chrono::steady_clock;
        for (const auto& row : sequences) {
            if (mode == "verify") {
                llama_memory_clear(llama_get_memory(ctx), true);
                prepare(row, 0, static_cast<int>(row.size()), true);
                execute();
                for (int i = 0; i < batch.n_tokens; ++i) {
                    const auto* logits = llama_get_logits_ith(ctx, i);
                    if (!logits) throw std::runtime_error("Missing verification row");
                    output.write(reinterpret_cast<const char*>(logits), vocab * sizeof(float));
                }
                continue;
            }
            for (int iteration = -warmup; iteration < runs; ++iteration) {
                llama_memory_clear(llama_get_memory(ctx), true); // Outside timing on both sides.
                prepare(row, 0, static_cast<int>(row.size()) - 1, false);
                auto start = Clock::now(); execute();
                const auto prefill = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                prepare(row, static_cast<int>(row.size()) - 1, static_cast<int>(row.size()), false);
                start = Clock::now(); execute();
                const auto decode = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                if (iteration >= 0) { prefills.push_back(prefill); decodes.push_back(decode); }
            }
        }
        metrics << std::setprecision(10) << "{\"threads\":" << threads
                << ",\"gpu_layers\":0,\"kv_precision\":\"FP32\",\"vocab\":" << vocab;
        if (mode == "bench") {
            metrics << ",\"prefill_p50_ms\":" << median(prefills) << ",\"decode_p50_ms\":" << median(decodes)
                    << ",\"prefill_samples_ms\":"; array(metrics, prefills);
            metrics << ",\"decode_samples_ms\":"; array(metrics, decodes);
        }
        metrics << '}';
        if (!output || !metrics) throw std::runtime_error("Output failed");
        llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
