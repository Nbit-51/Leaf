// Artifact-backed API test. Pass an unquantized decoder artifact with at least
// ten context positions; no downloaded model or private runtime hook is needed.
#include "leaf/runtime/decoder.h"
#include "leaf/runtime/worker_affinity.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using leaf::runtime::Decoder;
using leaf::runtime::FP32Math;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void exact(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "Output size changed or stale rows remain");
    require(std::all_of(actual.begin(), actual.end(), [](float v) { return std::isfinite(v); }),
            "Output contains nonfinite values");
    require(std::all_of(expected.begin(), expected.end(), [](float v) { return std::isfinite(v); }),
            "Reference output contains nonfinite values");
    require(actual.empty() || std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
            "Owned-return and caller-buffer logits differ");
}

unsigned number(const std::string& value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected an unsigned integer");
    const auto result = std::stoull(value);
    require(result <= std::numeric_limits<unsigned>::max(), "Integer argument is too large");
    return static_cast<unsigned>(result);
}

std::vector<unsigned> cpu_list(const std::string& text) {
    std::vector<unsigned> cpus;
    std::size_t first = 0;
    while (true) {
        const auto end = text.find(',', first);
        cpus.push_back(number(text.substr(first, end == std::string::npos ? end : end - first)));
        if (end == std::string::npos) return cpus;
        first = end + 1;
    }
}

std::size_t exercise(const std::string& artifact, unsigned threads, bool scalar,
                     FP32Math math, const std::vector<unsigned>& cpus) {
    const auto affinity = leaf::runtime::detail::worker_affinity_supported()
        ? leaf::runtime::detail::allowed_worker_cpus() : std::vector<unsigned>{};
    const auto check_affinity = [&] {
        if (leaf::runtime::detail::worker_affinity_supported())
            require(leaf::runtime::detail::allowed_worker_cpus() == affinity,
                    "Forward operation leaked caller affinity");
    };
    std::size_t calls = 0;
    {
        Decoder reference(artifact, threads, scalar, 32, cpus, math);
        Decoder candidate(artifact, threads, scalar, 32, cpus, math);
        check_affinity();
        const auto vocab = std::size_t(candidate.vocabulary_size());
        require(vocab > 0, "Fixture has no vocabulary");
        const auto ids = [&](std::size_t count, unsigned seed = 5) {
            std::vector<std::uint32_t> result(count);
            for (std::size_t i = 0; i < count; ++i)
                result[i] = static_cast<std::uint32_t>((seed + i * 17) % vocab);
            return result;
        };
        std::vector<float> output;
        const auto reset = [&] {
            reference.reset(); candidate.reset();
            require(reference.cache_tokens() == 0 && candidate.cache_tokens() == 0,
                    "Reset left cached tokens");
            check_affinity();
        };
        const auto execute = [&](const std::vector<std::uint32_t>& tokens, bool all) {
            const auto previous_tokens = reference.cache_tokens();
            require(previous_tokens == candidate.cache_tokens(), "Sessions started with different caches");
            const auto capacity = output.capacity();
            const auto* pointer = output.data();
            // A partial write cannot accidentally pass by retaining old logits.
            std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());
            const auto expected = reference.forward(tokens, all);
            candidate.forward_into(tokens, output, all);
            require(output.size() == (all ? tokens.size() : 1) * vocab, "Output shape is incorrect");
            exact(output, expected);
            require(reference.cache_tokens() == previous_tokens + tokens.size() &&
                    candidate.cache_tokens() == reference.cache_tokens(), "Cache advancement changed");
            if (output.size() <= capacity)
                require(output.capacity() == capacity && output.data() == pointer,
                        "Caller buffer reallocated despite sufficient capacity");
            else
                require(output.capacity() > capacity, "Caller buffer failed to grow");
            check_affinity();
            ++calls;
        };

        execute(ids(5), false);
        execute(ids(1, 3), false); // Repeated generation shape must reuse storage.
        reset();
        execute(ids(5), true);    // Grow to complete logits.
        execute(ids(2, 7), true); // Shrink without exposing stale tail rows.
        execute(ids(1, 9), false);
        reset();
        execute(ids(9), true);    // Cross the eight-token packed-GEMM boundary.
        execute(ids(1, 11), true);
        reset();
        execute(ids(3), false);
        execute(ids(2, 13), false); // Chunked cache append with last-token output.

        const auto saved_output = output;
        const auto saved_capacity = output.capacity();
        const auto* saved_pointer = output.data();
        const auto cached = candidate.cache_tokens();
        bool rejected = false;
        try { candidate.forward_into({}, output); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, "Empty token input was accepted");
        require(candidate.cache_tokens() == cached, "Rejected input changed cache length");
        exact(output, saved_output);
        require(output.capacity() == saved_capacity && output.data() == saved_pointer,
                "Rejected empty input changed caller storage");
        check_affinity();
        execute(ids(1, 15), false); // Reuse the session after the exception.

        reset();
        const auto expected_owned = reference.forward(ids(5), false);
        const auto owned = candidate.forward(ids(5), false);
        const auto saved_owned = owned;
        exact(owned, expected_owned);
        check_affinity();
        ++calls;
        execute(ids(1, 19), false);
        exact(owned, saved_owned); // A later into-call cannot alias an old return.

        // Switching caller-owned destinations must not replace the first buffer.
        const auto first_output = output;
        const auto first_capacity = output.capacity();
        const auto* first_pointer = output.data();
        std::vector<float> other(2 * vocab, std::numeric_limits<float>::quiet_NaN());
        const auto expected_other = reference.forward(ids(1, 23), false);
        const auto other_capacity = other.capacity();
        const auto* other_pointer = other.data();
        candidate.forward_into(ids(1, 23), other);
        exact(other, expected_other);
        exact(output, first_output);
        require(output.capacity() == first_capacity && output.data() == first_pointer &&
                other.capacity() == other_capacity && other.data() == other_pointer,
                "Switching output buffers replaced caller storage");
        require(candidate.cache_tokens() == reference.cache_tokens(), "Destination switch changed cache length");
        check_affinity();
        ++calls;
    }
    check_affinity();
    std::cout << "PASS output buffer: math=" << (math == FP32Math::Fast ? "fast" : "accurate")
              << " threads=" << threads << " scalar=" << scalar << " comparisons=" << calls << '\n';
    return calls;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2)
            throw std::runtime_error("Usage: test_decoder_output_buffer ARTIFACT [--threads=N] [--scalar=0|1] [--math=fast|accurate] [--worker-cpus=CSV]");
        std::vector<unsigned> thread_counts{1, 2}, scalar_modes{0, 1}, cpus;
        std::vector<FP32Math> math_modes{FP32Math::Fast, FP32Math::Accurate};
        bool threads_seen = false, scalar_seen = false, math_seen = false, cpus_seen = false;
        for (int i = 2; i < argc; ++i) {
            const std::string value = argv[i];
            if (value.rfind("--threads=", 0) == 0 && !threads_seen) {
                const auto count = number(value.substr(10));
                require(count > 0, "Thread count must be positive");
                thread_counts = {count}; threads_seen = true;
            } else if (value.rfind("--scalar=", 0) == 0 && !scalar_seen) {
                const auto scalar = number(value.substr(9));
                require(scalar <= 1, "Scalar must be zero or one");
                scalar_modes = {scalar}; scalar_seen = true;
            } else if (value.rfind("--math=", 0) == 0 && !math_seen) {
                const auto math = value.substr(7);
                require(math == "fast" || math == "accurate", "Unknown FP32 math policy");
                math_modes = {math == "accurate" ? FP32Math::Accurate : FP32Math::Fast}; math_seen = true;
            } else if (value.rfind("--worker-cpus=", 0) == 0 && !cpus_seen) {
                cpus = cpu_list(value.substr(14)); cpus_seen = true;
            } else throw std::runtime_error("Unknown or duplicate test option: " + value);
        }
        if (cpus_seen && !threads_seen) thread_counts = {static_cast<unsigned>(cpus.size())};
        if (cpus_seen)
            require(cpus.size() == thread_counts.front(), "Provide exactly one worker CPU per thread");
        std::size_t cases = 0, calls = 0;
        for (const auto threads : thread_counts)
            for (const auto scalar : scalar_modes)
                for (const auto math : math_modes) {
                    calls += exercise(argv[1], threads, scalar != 0, math, cpus);
                    ++cases;
                }
        std::cout << "{\"passed\":true,\"cases\":" << cases << ",\"comparisons\":" << calls << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Output-buffer test: " << error.what() << '\n';
        return 1;
    }
}
