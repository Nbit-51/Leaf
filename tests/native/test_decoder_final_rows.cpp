// Emit actual logits for paired before/after testing of last-row selection.
// Exercise full/last logits, reset, single-token and multi-token cache appends.
#include "leaf/runtime/decoder.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 9) throw std::runtime_error("MODEL OUTPUT THREADS SCALAR ACTIVATION_BITS CONTEXT LENGTHS LAST_ONLY");
        leaf::runtime::Decoder decoder(argv[1], std::stoul(argv[3]), std::stoi(argv[4]) != 0, std::stoul(argv[5]));
        const auto context = std::stoul(argv[6]);
        const bool last_only = std::stoi(argv[8]) != 0;
        std::ofstream output(argv[2], std::ios::binary);
        if (!output) throw std::runtime_error("Cannot open output");
        const auto vocab = decoder.vocabulary_size();
        std::vector<std::size_t> lengths;
        std::string list = argv[7];
        for (std::size_t first = 0; first < list.size();) {
            const auto end = list.find(',', first);
            lengths.push_back(std::stoul(list.substr(first, end - first)));
            if (end == std::string::npos) break;
            first = end + 1;
        }
        std::size_t calls = 0;
        auto execute = [&](const std::vector<std::uint32_t>& ids, bool all) {
            const auto previous = decoder.cache_tokens();
            const auto values = decoder.forward(ids, all);
            if (decoder.cache_tokens() != previous + ids.size() || values.size() != (all ? ids.size() : 1) * vocab)
                throw std::runtime_error("Cache length or output shape changed");
            output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
            ++calls;
        };
        for (auto length : lengths) {
            if (!length || length + 1 > context) throw std::runtime_error("Invalid test length");
            std::vector<std::uint32_t> prefix(length);
            for (std::size_t i = 0; i < length; ++i) prefix[i] = (i * 17 + 4) % vocab;
            decoder.reset();
            if (decoder.cache_tokens()) throw std::runtime_error("Reset failed");
            execute(prefix, false);
            execute({static_cast<std::uint32_t>(7 % vocab)}, false);
            // A second multi-token last-logit call must retain the first call's
            // complete K/V, including tokens whose final MLP rows were skipped.
            if (decoder.cache_tokens() + 36 <= context) {
                std::vector<std::uint32_t> chunk(32);
                for (std::size_t i = 0; i < chunk.size(); ++i) chunk[i] = (i * 11 + 3) % vocab;
                execute(chunk, false);
                if (!last_only) execute({3, 4, 5}, true);
            }
            if (!last_only) { decoder.reset(); execute(prefix, true); }
        }
        if (!output) throw std::runtime_error("Output write failed");
        std::cout << "{\"calls\":" << calls << ",\"cache_shapes_passed\":true}\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
