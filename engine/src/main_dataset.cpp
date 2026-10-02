#include "leaf/graph_parser.h"
#include "leaf/runtime/executor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
template <class T> std::vector<T> read(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open " + path);
    const auto length = input.tellg();
    if (length < 0 || static_cast<std::size_t>(length) % sizeof(T)) throw std::runtime_error("invalid array file");
    std::vector<T> values(static_cast<std::size_t>(length) / sizeof(T));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(values.data()), length)) throw std::runtime_error("truncated array");
    return values;
}
double percentile(const std::vector<double>& sorted, double fraction) {
    const double position = fraction * (sorted.size() - 1);
    const auto low = static_cast<std::size_t>(position);
    const auto high = std::min(low + 1, sorted.size() - 1);
    return sorted[low] + (position - low) * (sorted[high] - sorted[low]);
}
}

int main(int argc, char** argv) {
    if (argc < 8 || argc > 11) {
        std::cerr << "Usage: leaf_dataset artifact.leaf inputs.bin expected.bin labels.u32 shape count metrics.json"
                     " [strict=1] [atol=0.001] [rtol=0.001]\n";
        return 1;
    }
    try {
        leaf::Tensor input;
        std::istringstream dimensions(argv[5]); std::string item;
        while (std::getline(dimensions, item, ',')) {
            const auto dimension = std::stoull(item);
            if (!dimension) throw std::runtime_error("zero input dimension");
            input.shape.push_back(dimension);
        }
        const auto count = std::stoull(argv[6]);
        if (!count || input.shape.empty()) throw std::runtime_error("empty dataset");
        const auto elements = input.element_count();
        const auto inputs = read<float>(argv[2]), expected = read<float>(argv[3]);
        const auto labels = read<std::uint32_t>(argv[4]);
        if (inputs.size() / count != elements || inputs.size() % count ||
            labels.size() != count || expected.size() % count || expected.empty())
            throw std::runtime_error("dataset array sizes do not match");
        const auto classes = expected.size() / count;
        const bool strict = argc > 8 ? std::stoi(argv[8]) != 0 : true;
        const double atol = argc > 9 ? std::stod(argv[9]) : 1e-3;
        const double rtol = argc > 10 ? std::stod(argv[10]) : 1e-3;
        if (atol < 0 || rtol < 0 || !std::isfinite(atol) || !std::isfinite(rtol))
            throw std::runtime_error("invalid comparison tolerance");
        const auto graph = leaf::Graph::load(argv[1]);
        leaf::Executor executor;
        input.values.assign(inputs.begin(), inputs.begin() + elements);
        for (unsigned warmup = 0; warmup < 20; ++warmup) executor.run(graph, input);
        std::vector<double> times;
        times.reserve(count);
        double maximum_error = 0;
        std::size_t correct = 0, agreement = 0, parity_failures = 0;
        for (std::size_t sample = 0; sample < count; ++sample) {
            std::copy_n(inputs.data() + sample * elements, elements, input.values.data());
            const auto start = std::chrono::steady_clock::now();
            const auto output = executor.run(graph, input);
            times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            if (output.values.size() != classes || labels[sample] >= classes)
                throw std::runtime_error("classification output/label shape mismatch");
            const auto* reference = expected.data() + sample * classes;
            bool parity = true;
            for (std::size_t i = 0; i < classes; ++i) {
                const double error = std::abs(static_cast<double>(output.values[i]) - reference[i]);
                if (!std::isfinite(output.values[i]) || !std::isfinite(reference[i]))
                    throw std::runtime_error("non-finite model output");
                maximum_error = std::max(maximum_error, error);
                parity = parity && error <= atol + rtol * std::abs(reference[i]);
            }
            parity_failures += !parity;
            const auto prediction = std::max_element(output.values.begin(), output.values.end()) - output.values.begin();
            const auto expected_prediction = std::max_element(reference, reference + classes) - reference;
            correct += prediction == labels[sample]; agreement += prediction == expected_prediction;
            if ((sample + 1) % 1000 == 0) std::cout << "Evaluated " << sample + 1 << '/' << count << '\n' << std::flush;
        }
        const double mean = std::accumulate(times.begin(), times.end(), 0.0) / count;
        const auto ordered_times = times;
        std::sort(times.begin(), times.end());
        std::ofstream metrics(argv[7]);
        if (!metrics) throw std::runtime_error("cannot write dataset metrics");
        metrics.precision(12);
        metrics << "{\"samples\":" << count << ",\"correct\":" << correct
                << ",\"accuracy\":" << static_cast<double>(correct) / count
                << ",\"prediction_agreement\":" << static_cast<double>(agreement) / count
                << ",\"max_abs_error\":" << maximum_error << ",\"parity_failures\":" << parity_failures
                << ",\"p50_ms\":" << percentile(times, 0.5) << ",\"p95_ms\":" << percentile(times, 0.95)
                << ",\"mean_ms\":" << mean << ",\"warmup\":20,\"latency_samples_ms\":[";
        for (std::size_t i = 0; i < ordered_times.size(); ++i) {
            if (i) metrics << ',';
            metrics << ordered_times[i];
        }
        metrics << "]}\n";
        if (strict && parity_failures) throw std::runtime_error("dataset FP32 parity gate failed");
        return 0;
    } catch (const std::exception& error) { std::cerr << "Leaf dataset: " << error.what() << '\n'; return 1; }
}
