#pragma once
// Diagnostic builds only: production builds contain no trace I/O or branches.
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct DecoderTrace {
    std::filesystem::path directory;
    DecoderTrace() {
        const auto* value = std::getenv("LEAF_TRACE_DIRECTORY");
        if (!value || !*value) throw std::runtime_error("Trace build requires LEAF_TRACE_DIRECTORY");
        directory = value;
    }
    void write(const std::string& name, const std::vector<float>& values) const {
        // The diagnostic harness runs exactly one forward call per process.
        const auto path = directory / (name + ".f32");
        if (std::filesystem::exists(path)) throw std::runtime_error("Trace refuses to overwrite a forward tensor");
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
        if (!stream) throw std::runtime_error("Cannot write decoder trace");
    }
};
