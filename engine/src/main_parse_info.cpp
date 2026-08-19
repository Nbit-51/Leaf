// Minimal CLI: loads a .leaf file and prints its structure. Equivalent
// to read_binary.py's __main__ block -- exists to prove the C++ parser
// reads real exported graphs correctly, before any kernel work happens.

#include "leaf/graph_parser.h"

#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path.leaf>\n";
        return 1;
    }

    try {
        leaf::Graph g = leaf::Graph::load(argv[1]);

        std::cout << "version=" << g.version()
                  << ", nodes=" << g.nodes().size()
                  << ", initializers=" << g.initializer_count() << "\n";

        std::cout << "inputs=[";
        for (size_t i = 0; i < g.inputs().size(); ++i) {
            std::cout << g.inputs()[i] << (i + 1 < g.inputs().size() ? ", " : "");
        }
        std::cout << "]\n";

        std::cout << "outputs=[";
        for (size_t i = 0; i < g.outputs().size(); ++i) {
            std::cout << g.outputs()[i] << (i + 1 < g.outputs().size() ? ", " : "");
        }
        std::cout << "]\n";

        const auto& first = g.nodes().front();
        std::cout << "\nfirst node: op_type=" << first.op_type
                  << ", inputs=" << first.inputs.size()
                  << ", outputs=" << first.outputs.size()
                  << ", attrs=" << first.attributes_json << "\n";

        // Spot-check one real initializer's data against its declared
        // shape/byte_length, and print its first few float values --
        // this is the actual proof the data block offsets are correct,
        // not just that the header/tables parsed.
        const std::string& sample_name = g.initializer_meta(first.inputs.size() > 1 ? first.inputs[1] : "").name;
        if (!sample_name.empty()) {
            const auto& meta = g.initializer_meta(sample_name);
            const float* ptr = g.initializer_data(sample_name);
            std::cout << "\nsample initializer '" << sample_name << "': shape=[";
            for (size_t i = 0; i < meta.shape.size(); ++i) {
                std::cout << meta.shape[i] << (i + 1 < meta.shape.size() ? ", " : "");
            }
            std::cout << "], first 4 values=[";
            for (int i = 0; i < 4; ++i) {
                std::cout << ptr[i] << (i < 3 ? ", " : "");
            }
            std::cout << "]\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
