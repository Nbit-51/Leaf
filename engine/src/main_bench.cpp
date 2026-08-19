#include "leaf/graph_parser.h"
#include "leaf/runtime/executor.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<size_t> parse_shape(const std::string& text) {
    std::vector<size_t> shape;
    size_t start = 0;
    while (start < text.size()) {
        const size_t end = text.find(',', start);
        const std::string token = text.substr(start, end == std::string::npos ? end : end - start);
        if (token.empty()) {
            throw std::runtime_error("empty dimension in input shape");
        }
        const unsigned long long dimension = std::stoull(token);
        if (dimension == 0) {
            throw std::runtime_error("input dimensions must be positive");
        }
        shape.push_back(static_cast<size_t>(dimension));
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return shape;
}

std::vector<float> read_floats(const std::string& path, size_t count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open input: " + path);
    }
    std::vector<float> values(count);
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (file.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
        throw std::runtime_error("input byte count does not match the supplied shape");
    }
    return values;
}

size_t parse_count(const char* text, const char* name) {
    const unsigned long long value = std::stoull(text);
    if (value == 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
    return static_cast<size_t>(value);
}

double percentile(const std::vector<double>& sorted_values, double fraction) {
    const size_t index = static_cast<size_t>(fraction * static_cast<double>(sorted_values.size() - 1));
    return sorted_values[index];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.leaf> <input.bin> <N,C,H,W> [warmup=1] [runs=5]\n";
        return 1;
    }

    try {
        leaf::Tensor input;
        input.shape = parse_shape(argv[3]);
        input.values = read_floats(argv[2], input.element_count());
        const size_t warmup = argc >= 5 ? parse_count(argv[4], "warmup") : 1;
        const size_t runs = argc >= 6 ? parse_count(argv[5], "runs") : 5;

        const leaf::Graph graph = leaf::Graph::load(argv[1]);
        const leaf::Executor executor;
        for (size_t i = 0; i < warmup; ++i) {
            static_cast<void>(executor.run(graph, input));
        }

        std::vector<double> timings_ms;
        timings_ms.reserve(runs);
        leaf::Tensor output;
        for (size_t i = 0; i < runs; ++i) {
            const auto start = std::chrono::steady_clock::now();
            output = executor.run(graph, input);
            const auto end = std::chrono::steady_clock::now();
            timings_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        const double mean = std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0) /
                            static_cast<double>(timings_ms.size());
        std::sort(timings_ms.begin(), timings_ms.end());
        const double checksum = std::accumulate(output.values.begin(), output.values.end(), 0.0);
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Leaf FP32 inference benchmark\n"
                  << "  warmup runs: " << warmup << "\n"
                  << "  measured runs: " << runs << "\n"
                  << "  latency ms: min=" << timings_ms.front()
                  << ", p50=" << percentile(timings_ms, 0.50)
                  << ", p95=" << percentile(timings_ms, 0.95)
                  << ", mean=" << mean << "\n"
                  << "  throughput: " << (1000.0 / mean) << " inferences/s\n"
                  << "  output checksum: " << checksum << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
