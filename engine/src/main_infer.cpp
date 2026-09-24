#include "leaf/graph_parser.h"
#include "leaf/runtime/executor.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

void write_floats(const std::string& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot create output: " + path);
    }
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!file) {
        throw std::runtime_error("failed while writing output: " + path);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool single = argc == 5;
    const bool named = argc >= 6 && (argc - 3) % 3 == 0;
    if (!single && !named) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.leaf> <input.bin> <shape> <output.bin>\n"
                  << "   or: " << argv[0]
                  << " <model.leaf> <output-prefix> <name> <shape> <input.bin> [more input triples]\n";
        return 1;
    }
    try {
        const leaf::Graph graph = leaf::Graph::load(argv[1]);
        if (single) {
            leaf::Tensor input;
            input.shape = parse_shape(argv[3]);
            input.values = read_floats(argv[2], input.element_count());
            const leaf::Tensor output = leaf::Executor().run(graph, input);
            write_floats(argv[4], output.values);
            std::cout << "Output shape: [";
            for (size_t i = 0; i < output.shape.size(); ++i) {
                std::cout << output.shape[i] << (i + 1 == output.shape.size() ? "" : ", ");
            }
            std::cout << "]\nWrote " << output.values.size() << " float32 values to " << argv[4] << "\n";
        } else {
            std::unordered_map<std::string, leaf::Tensor> inputs;
            for (int index = 3; index < argc; index += 3) {
                leaf::Tensor tensor;
                tensor.shape = parse_shape(argv[index + 1]);
                tensor.values = read_floats(argv[index + 2], tensor.element_count());
                if (!inputs.emplace(argv[index], std::move(tensor)).second) {
                    throw std::runtime_error("duplicate named input");
                }
            }
            auto outputs = leaf::Executor().run_outputs(graph, inputs);
            for (size_t index = 0; index < graph.outputs().size(); ++index) {
                const std::string& name = graph.outputs()[index];
                const leaf::Tensor& tensor = outputs.at(name);
                const std::string path = std::string(argv[2]) + "." + std::to_string(index) + ".bin";
                write_floats(path, tensor.values);
                std::cout << name << "\t" << path << "\t";
                for (size_t dimension = 0; dimension < tensor.shape.size(); ++dimension) {
                    if (dimension != 0) std::cout << ',';
                    std::cout << tensor.shape[dimension];
                }
                std::cout << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
