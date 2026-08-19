#include "leaf/graph_parser.h"
#include "leaf/runtime/executor.h"

#include <fstream>
#include <iostream>
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
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.leaf> <input.bin> <N,C,H,W> <output.bin>\n";
        return 1;
    }
    try {
        leaf::Tensor input;
        input.shape = parse_shape(argv[3]);
        input.values = read_floats(argv[2], input.element_count());

        const leaf::Graph graph = leaf::Graph::load(argv[1]);
        const leaf::Tensor output = leaf::Executor().run(graph, input);
        write_floats(argv[4], output.values);

        std::cout << "Output shape: [";
        for (size_t i = 0; i < output.shape.size(); ++i) {
            std::cout << output.shape[i] << (i + 1 == output.shape.size() ? "" : ", ");
        }
        std::cout << "]\nWrote " << output.values.size() << " float32 values to " << argv[4] << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
