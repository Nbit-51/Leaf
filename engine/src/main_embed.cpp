#include "leaf/runtime/embedding.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>

namespace {
std::uint32_t read_u32(std::istream &input) {
    unsigned char bytes[4];
    if (!input.read(reinterpret_cast<char *>(bytes), 4))
        throw std::runtime_error("Truncated embedding request");
    return std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8) | (std::uint32_t(bytes[2]) << 16) |
           (std::uint32_t(bytes[3]) << 24);
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc < 4 || argc > 6)
            throw std::runtime_error("Usage: leaf_embed ARTIFACT REQUEST OUTPUT [THREADS] [--scalar]");
        unsigned threads = argc >= 5 ? static_cast<unsigned>(std::stoul(argv[4])) : 1;
        bool scalar = argc == 6 && std::string(argv[5]) == "--scalar";
        if (argc == 6 && !scalar)
            throw std::runtime_error("Unknown embedding option");
        leaf::runtime::EmbeddingEncoder encoder(argv[1], threads, scalar);
        std::ifstream input(argv[2], std::ios::binary);
        char magic[8];
        if (!input.read(magic, 8) || std::memcmp(magic, "LEAFER01", 8))
            throw std::runtime_error("Invalid embedding request");
        auto count = read_u32(input);
        if (!count || count > 1000000)
            throw std::runtime_error("Invalid request count");
        std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("Cannot open embedding output");
        for (std::uint32_t i = 0; i < count; ++i) {
            auto length = read_u32(input);
            if (!length || length > encoder.maximum_tokens())
                throw std::runtime_error("Invalid request length");
            std::vector<std::uint32_t> ids(length);
            for (auto &id : ids)
                id = read_u32(input);
            auto values = encoder.encode(ids);
            output.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
            if (!output)
                throw std::runtime_error("Cannot write embeddings");
        }
        if (input.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("Trailing embedding request bytes");
        std::cout << "{\"sequences\":" << count << ",\"dimensions\":" << encoder.dimensions()
                  << ",\"avx2\":" << (encoder.uses_avx2() ? "true" : "false") << ",\"threads\":" << threads
                  << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
