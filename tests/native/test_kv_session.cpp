#include "leaf/graph_parser.h"
#include "leaf/runtime/executor.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

leaf::Tensor input(const std::string& path, std::vector<size_t> shape) {
    leaf::Tensor tensor;
    tensor.shape = std::move(shape);
    tensor.values.resize(tensor.element_count());
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open input " + path);
    const size_t bytes = tensor.values.size() * sizeof(float);
    stream.read(reinterpret_cast<char*>(tensor.values.data()),
                static_cast<std::streamsize>(bytes));
    if (stream.gcount() != static_cast<std::streamsize>(bytes)) {
        throw std::runtime_error("input byte count mismatch: " + path);
    }
    return tensor;
}

void write(const std::string& path, const leaf::Tensor& tensor) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot write output " + path);
    stream.write(reinterpret_cast<const char*>(tensor.values.data()),
                 static_cast<std::streamsize>(tensor.values.size() * sizeof(float)));
    if (!stream) throw std::runtime_error("failed writing output " + path);
}

double median(std::vector<double>& values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 0
        ? (values[middle - 1] + values[middle]) * 0.5 : values[middle];
}

}  // namespace

int main(int argc, char** argv) {
    // Test harness for a graph with inputs q,k,v,mask; one cached Attention
    // node (cache_id attribute); and output y. Mask files are float32.
    if (argc != 16) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.leaf> <prefix.q> <prefix.k> <prefix.v> <prefix.mask>"
                  << " <step.q> <step.k> <step.v> <step.mask>"
                  << " <query_heads> <kv_heads> <head_dim> <prefix_tokens>"
                  << " <runs> <output_prefix>\n";
        return 1;
    }
    try {
        const size_t query_heads = std::stoull(argv[10]);
        const size_t kv_heads = std::stoull(argv[11]);
        const size_t head_dim = std::stoull(argv[12]);
        const size_t prefix_tokens = std::stoull(argv[13]);
        const size_t runs = std::stoull(argv[14]);
        if (query_heads == 0 || kv_heads == 0 || head_dim == 0 ||
            prefix_tokens == 0 || runs == 0) {
            throw std::runtime_error("all dimensions and runs must be positive");
        }
        const leaf::Graph graph = leaf::Graph::load(argv[1]);
        const std::unordered_map<std::string, leaf::Tensor> prefix{
            {"q", input(argv[2], {1, query_heads, prefix_tokens, head_dim})},
            {"k", input(argv[3], {1, kv_heads, prefix_tokens, head_dim})},
            {"v", input(argv[4], {1, kv_heads, prefix_tokens, head_dim})},
            {"mask", input(argv[5], {1, 1, prefix_tokens, prefix_tokens})},
        };
        const std::unordered_map<std::string, leaf::Tensor> step{
            {"q", input(argv[6], {1, query_heads, 1, head_dim})},
            {"k", input(argv[7], {1, kv_heads, 1, head_dim})},
            {"v", input(argv[8], {1, kv_heads, 1, head_dim})},
            {"mask", input(argv[9], {1, 1, 1, 1})},
        };
        leaf::Executor executor;
        std::vector<double> prefill_ms, decode_ms;
        prefill_ms.reserve(runs);
        decode_ms.reserve(runs);
        for (size_t iteration = 0; iteration < runs + 1; ++iteration) {
            std::unordered_map<std::string, leaf::runtime::KVCache> caches;
            const auto before_prefill = std::chrono::steady_clock::now();
            auto prefill = executor.run_outputs_cached(graph, prefix, caches);
            const auto before_decode = std::chrono::steady_clock::now();
            auto decode = executor.run_outputs_cached(graph, step, caches);
            const auto after_decode = std::chrono::steady_clock::now();
            if (caches.at("layer_0").length() != prefix_tokens + 1) {
                throw std::runtime_error("cached attention did not append the decode token");
            }
            if (iteration == 0) {
                write(std::string(argv[15]) + ".prefill.bin", prefill.at("y"));
                write(std::string(argv[15]) + ".decode.bin", decode.at("y"));
            } else {
                prefill_ms.push_back(std::chrono::duration<double, std::milli>(
                    before_decode - before_prefill).count());
                decode_ms.push_back(std::chrono::duration<double, std::milli>(
                    after_decode - before_decode).count());
            }
        }
        std::cout << "prefill_p50_ms=" << median(prefill_ms)
                  << " decode_p50_ms=" << median(decode_ms) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
