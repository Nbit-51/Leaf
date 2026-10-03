#include "leaf/runtime/decoder.h"
#include "leaf/runtime/kv_cache.h"
#include "leaf/kernels/token_panel.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#include <immintrin.h>
#define LEAF_DECODER_AVX 1
#define LEAF_AVX_TARGET __attribute__((target("avx2,fma")))
#else
#define LEAF_DECODER_AVX 0
#endif

#if LEAF_DECODER_AVX && ((!defined(__clang__) && __GNUC__ >= 11) || (defined(__clang__) && __clang_major__ >= 15))
#define LEAF_DECODER_VNNI 1
#define LEAF_VNNI_TARGET __attribute__((target("avx2,avxvnni")))
#else
#define LEAF_DECODER_VNNI 0
#endif

namespace leaf::runtime {
namespace {
void check(bool valid, const std::string& message) {
    if (!valid) throw std::runtime_error("Leaf decoder: " + message);
}

// Optional diagnostics only: fixed operation classes keep profiling independent
// of import-adapter names. A disabled scope never reads the clock or aggregates.
class DecoderProfile {
public:
    enum class Phase : std::size_t {
        Forward, Embedding, RMSNorm, LayerNorm, LinearFP32, LinearW8A32,
        LinearW4A32, LinearW8A8, LinearW4A8, RoPE, PositionReorder,
        Attention, Activation, KVAppend, CausalMask, Residual, Count
    };
    DecoderProfile() {
        const char* value = std::getenv("LEAF_DECODER_PROFILE");
        enabled_ = value && value[0];
    }
    void forward_tokens(std::size_t tokens) noexcept { bucket_ = tokens == 1 ? 1 : 0; }
    class Scope {
    public:
        Scope(DecoderProfile& profile, Phase phase, std::size_t tokens)
            : profile_(profile.enabled_ ? &profile : nullptr), phase_(phase), tokens_(tokens) {
            if (profile_) start_ = Clock::now();
        }
        ~Scope() {
            if (profile_) {
                auto& total = profile_->totals_[profile_->bucket_][static_cast<std::size_t>(phase_)];
                ++total.calls; total.tokens += tokens_;
                total.milliseconds += std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
            }
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
    private:
        using Clock = std::chrono::steady_clock;
        DecoderProfile* profile_;
        Phase phase_;
        std::size_t tokens_;
        Clock::time_point start_{};
    };
    void emit() const noexcept {
        if (!enabled_) return;
        // Destruction must not turn a successful inference into a failure if
        // diagnostics cannot be allocated or stderr has been closed.
        try {
            static const char* names[] = {
                "forward", "embedding", "rms_norm", "layer_norm", "linear_fp32", "linear_w8a32",
                "linear_w4a32", "linear_w8a8", "linear_w4a8", "rope", "position_reorder",
                "attention", "activation", "kv_append", "causal_mask", "residual"
            };
            std::ostringstream output; output.precision(10);
            output << "{\"leaf_decoder_profile\":1,\"clock\":\"steady_clock\","
                      "\"bucket_by\":\"forward_input_tokens\",\"timings\":{";
            for (std::size_t bucket = 0; bucket < totals_.size(); ++bucket) {
                output << (bucket ? ",\"decode\":{" : "\"prefill\":{");
                bool first = true;
                for (std::size_t phase = 0; phase < names_count; ++phase) {
                    const auto& total = totals_[bucket][phase];
                    if (!total.calls) continue;
                    output << (first ? "" : ",") << '"' << names[phase]
                           << "\":{\"calls\":" << total.calls << ",\"tokens\":" << total.tokens
                           << ",\"milliseconds\":" << total.milliseconds << '}';
                    first = false;
                }
                output << '}';
            }
            output << "}}\n"; std::cerr << output.str();
        } catch (...) {}
    }
private:
    struct Total { std::uint64_t calls = 0, tokens = 0; double milliseconds = 0; };
    static constexpr auto names_count = static_cast<std::size_t>(Phase::Count);
    std::array<std::array<Total, names_count>, 2> totals_{};
    bool enabled_ = false;
    std::size_t bucket_ = 0;
};

class Mapping {
public:
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
    explicit Mapping(const std::string& path) {
#ifdef _WIN32
        file_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        check(file_ != INVALID_HANDLE_VALUE, "cannot open artifact " + path);
        LARGE_INTEGER length;
        if (!GetFileSizeEx(file_, &length) || length.QuadPart <= 0 ||
            static_cast<std::uint64_t>(length.QuadPart) > std::numeric_limits<std::size_t>::max()) {
            CloseHandle(file_); throw std::runtime_error("invalid decoder artifact size");
        }
        size = static_cast<std::size_t>(length.QuadPart);
        mapping_ = CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_) data = static_cast<const std::uint8_t*>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        if (!data) {
            if (mapping_) CloseHandle(mapping_);
            CloseHandle(file_); throw std::runtime_error("cannot map decoder artifact");
        }
#else
        file_ = open(path.c_str(), O_RDONLY);
        check(file_ >= 0, "cannot open artifact " + path);
        struct stat info;
        if (fstat(file_, &info) || info.st_size <= 0) {
            close(file_); throw std::runtime_error("invalid decoder artifact size");
        }
        size = static_cast<std::size_t>(info.st_size);
        void* mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file_, 0);
        if (mapped == MAP_FAILED) {
            close(file_); throw std::runtime_error("cannot map decoder artifact");
        }
        data = static_cast<const std::uint8_t*>(mapped);
#endif
    }
    ~Mapping() {
#ifdef _WIN32
        if (data) UnmapViewOfFile(data);
        if (mapping_) CloseHandle(mapping_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
#else
        if (data) munmap(const_cast<std::uint8_t*>(data), size);
        if (file_ >= 0) close(file_);
#endif
    }
private:
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int file_ = -1;
#endif
};

struct Reader {
    const Mapping& mapping;
    std::size_t position = 0;
    void require(std::size_t length) const {
        check(position <= mapping.size && length <= mapping.size - position, "truncated artifact metadata");
    }
    std::uint32_t u32() {
        require(4); std::uint32_t result = 0;
        for (unsigned i = 0; i < 4; ++i) result |= std::uint32_t(mapping.data[position++]) << (8 * i);
        return result;
    }
    std::uint64_t u64() {
        const std::uint64_t low = u32(); return low | (std::uint64_t(u32()) << 32);
    }
    float f32() { const auto bits = u32(); float result; std::memcpy(&result, &bits, 4); return result; }
    std::string string() {
        const auto length = u32(); require(length);
        std::string value(reinterpret_cast<const char*>(mapping.data + position), length);
        position += length; return value;
    }
};

class Workers {
public:
    explicit Workers(unsigned threads) : threads_(threads) {
        check(threads >= 1 && threads <= 64, "thread count must be 1..64");
        try {
        for (unsigned i = 1; i < threads_; ++i) workers_.emplace_back([this] {
            std::size_t observed = 0;
            for (;;) {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [&] { return stop_ || epoch_ != observed; });
                if (stop_) return;
                observed = epoch_; lock.unlock(); drain(); lock.lock();
                if (--pending_ == 0) finished_.notify_one();
            }
        });
        } catch (...) {
            { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
            ready_.notify_all();
            for (auto& worker : workers_) worker.join();
            throw;
        }
    }
    ~Workers() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        ready_.notify_all(); for (auto& worker : workers_) worker.join();
    }
    void run(std::size_t count, std::function<void(std::size_t, std::size_t)> task) {
        if (threads_ == 1) { task(0, count); return; }
        { std::lock_guard<std::mutex> lock(mutex_);
          task_ = std::move(task); count_ = count; next_ = 0; pending_ = threads_ - 1; ++epoch_; }
        ready_.notify_all(); drain();
        std::unique_lock<std::mutex> lock(mutex_);
        finished_.wait(lock, [&] { return pending_ == 0; });
    }
private:
    void drain() {
        for (;;) {
            const std::size_t first = next_.fetch_add(32, std::memory_order_relaxed);
            if (first >= count_) return;
            task_(first, std::min(first + 32, count_));
        }
    }
    unsigned threads_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable ready_, finished_;
    bool stop_ = false;
    std::size_t epoch_ = 0, count_ = 0;
    unsigned pending_ = 0;
    std::atomic<std::size_t> next_{0};
    std::function<void(std::size_t, std::size_t)> task_;
};

struct Weight {
    std::size_t rows = 0, cols = 0, bits = 0, group = 0;
    const std::uint8_t* data = nullptr;
    const float* scales = nullptr;
    std::vector<std::int32_t> signed_sums;
    float value(std::size_t row, std::size_t col) const {
        const std::size_t index = row * cols + col;
        if (bits == 32) return reinterpret_cast<const float*>(data)[index];
        int quantized;
        if (bits == 8) quantized = reinterpret_cast<const std::int8_t*>(data)[index];
        else { quantized = (data[index / 2] >> ((index % 2) * 4)) & 15;
               if (quantized >= 8) quantized -= 16; }
        return static_cast<float>(quantized) * scales[row * (cols / group) + col / group];
    }
};

using Dot = void (*)(const Weight&, std::size_t, const float*, std::size_t, float*);
using QuantDot = void (*)(const Weight&, std::size_t, const std::int16_t*,
                         std::size_t, const float*, float*);
void dot_scalar(const Weight& w, std::size_t row, const float* inputs, std::size_t tokens, float* result) {
    std::fill_n(result, tokens, 0.0f);
    for (std::size_t col = 0; col < w.cols; ++col) {
        const float value = w.value(row, col);
        for (std::size_t t = 0; t < tokens; ++t) result[t] += value * inputs[t * w.cols + col];
    }
}

void quant_dot_scalar(const Weight& w, std::size_t row, const std::int16_t* inputs,
                      std::size_t tokens, const float* input_scales, float* result) {
    std::fill_n(result, tokens, 0.0f);
    for (std::size_t first = 0; first < w.cols; first += w.group) {
        for (std::size_t t = 0; t < tokens; ++t) {
            std::int32_t accumulator = 0;
            for (std::size_t col = first; col < first + w.group; ++col) {
                const auto index = row * w.cols + col;
                int value = w.bits == 8 ? reinterpret_cast<const std::int8_t*>(w.data)[index] :
                                         (w.data[index / 2] >> ((index % 2) * 4)) & 15;
                if (w.bits == 4 && value >= 8) value -= 16;
                accumulator += value * inputs[t * w.cols + col];
            }
            result[t] += static_cast<float>(accumulator) * input_scales[t] *
                         w.scales[row * (w.cols / w.group) + first / w.group];
        }
    }
}

#if LEAF_DECODER_AVX
LEAF_AVX_TARGET float reduce(__m256 v) {
    const __m128 halves = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 pairs = _mm_add_ps(halves, _mm_movehl_ps(halves, halves));
    return _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 1)));
}
__attribute__((always_inline)) LEAF_AVX_TARGET inline __m256 weight8(
        const Weight& w, std::size_t row, std::size_t col) {
        __m256 values;
        const std::size_t index = row * w.cols + col;
        if (w.bits == 32) values = _mm256_loadu_ps(reinterpret_cast<const float*>(w.data) + index);
        else {
            __m256i integers;
            if (w.bits == 8) {
                integers = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w.data + index)));
            } else {
                std::uint32_t packed;
                std::memcpy(&packed, w.data + index / 2, 4);
                const __m256i lanes = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
                integers = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(packed), lanes),
                                             _mm256_set1_epi32(15));
                integers = _mm256_sub_epi32(_mm256_xor_si256(integers, _mm256_set1_epi32(8)),
                                             _mm256_set1_epi32(8));
            }
            values = _mm256_cvtepi32_ps(integers);
        }
        return values;
}
template <unsigned Bits>
__attribute__((always_inline)) LEAF_AVX_TARGET inline __m256 fixed_weight8(
        const Weight& w, std::size_t row, std::size_t col) {
    if constexpr (Bits == 32)
        return _mm256_loadu_ps(reinterpret_cast<const float*>(w.data) + row * w.cols + col);
    if constexpr (Bits == 8)
        return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w.data + row * w.cols + col))));
    return weight8(w, row, col);
}
template <unsigned Bits>
LEAF_AVX_TARGET void dot_one_avx(const Weight& w, std::size_t row, const float* input, float* result) {
    *result = 0.0f;
    const auto group = Bits == 32 ? w.cols : w.group;
    const auto groups = w.cols / group;
    for (std::size_t first = 0; first < w.cols; first += group) {
        __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;
        std::size_t col = first;
        const auto end = first + group;
        for (; col + 32 <= end; col += 32) {
            a = _mm256_fmadd_ps(fixed_weight8<Bits>(w, row, col), _mm256_loadu_ps(input + col), a);
            b = _mm256_fmadd_ps(fixed_weight8<Bits>(w, row, col + 8), _mm256_loadu_ps(input + col + 8), b);
            c = _mm256_fmadd_ps(fixed_weight8<Bits>(w, row, col + 16), _mm256_loadu_ps(input + col + 16), c);
            d = _mm256_fmadd_ps(fixed_weight8<Bits>(w, row, col + 24), _mm256_loadu_ps(input + col + 24), d);
        }
        a = _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d));
        for (; col + 8 <= end; col += 8)
            a = _mm256_fmadd_ps(fixed_weight8<Bits>(w, row, col), _mm256_loadu_ps(input + col), a);
        float partial = reduce(a);
        const auto scale = Bits == 32 ? 1.0f : w.scales[row * groups + first / group];
        for (; col < end; ++col) partial += (w.value(row, col) / scale) * input[col];
        *result += partial * scale;
    }
}
LEAF_AVX_TARGET void dot_avx(const Weight& w, std::size_t row, const float* inputs,
                            std::size_t tokens, float* result) {
    std::fill_n(result, tokens, 0.0f);
    const std::size_t group = w.bits == 32 ? w.cols : w.group;
    const std::size_t groups = w.cols / group;
    for (std::size_t index = 0; index < groups; ++index) {
        __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;
        std::size_t col = index * group;
        const std::size_t end = col + group;
        if (tokens == 1) {
            // Four independent sums hide FMA dependency latency during GEMV.
            for (; col + 32 <= end; col += 32) {
                a = _mm256_fmadd_ps(weight8(w, row, col), _mm256_loadu_ps(inputs + col), a);
                b = _mm256_fmadd_ps(weight8(w, row, col + 8), _mm256_loadu_ps(inputs + col + 8), b);
                c = _mm256_fmadd_ps(weight8(w, row, col + 16), _mm256_loadu_ps(inputs + col + 16), c);
                d = _mm256_fmadd_ps(weight8(w, row, col + 24), _mm256_loadu_ps(inputs + col + 24), d);
            }
            a = _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d));
        }
        // Explicit registers avoid spills; scales are applied once per group,
        // removing per-vector integer division and dequantization multiplies.
        for (; col + 8 <= end; col += 8) {
            const auto values = weight8(w, row, col);
            a = _mm256_fmadd_ps(values, _mm256_loadu_ps(inputs + col), a);
            if (tokens > 1) b = _mm256_fmadd_ps(values, _mm256_loadu_ps(inputs + w.cols + col), b);
            if (tokens > 2) c = _mm256_fmadd_ps(values, _mm256_loadu_ps(inputs + 2 * w.cols + col), c);
            if (tokens > 3) d = _mm256_fmadd_ps(values, _mm256_loadu_ps(inputs + 3 * w.cols + col), d);
        }
        float partial[4] = {reduce(a), reduce(b), reduce(c), reduce(d)};
        const float scale = w.bits == 32 ? 1.0f : w.scales[row * groups + index];
        for (; col < end; ++col) {
            const float value = w.value(row, col) / scale;
            for (std::size_t t = 0; t < tokens; ++t) partial[t] += value * inputs[t * w.cols + col];
        }
        for (std::size_t t = 0; t < tokens; ++t) result[t] += partial[t] * scale;
    }
}
LEAF_AVX_TARGET void dot_specialized_avx(const Weight& w, std::size_t row, const float* inputs,
                                       std::size_t tokens, float* result) {
    if (tokens == 1 && w.bits == 32) { dot_one_avx<32>(w, row, inputs, result); return; }
    if (tokens == 1 && w.bits == 8) { dot_one_avx<8>(w, row, inputs, result); return; }
    dot_avx(w, row, inputs, tokens, result);
}
LEAF_AVX_TARGET void float_tile_avx(const Weight& w, std::size_t row,
        const float* inputs, std::size_t tokens, float* result) {
    // Two rows share four input vectors. Eight independent FMA chains avoid
    // the four-chain dependency bottleneck without packing or expanding weights.
    // Preserve the per-output accumulation/reduction order of dot_avx.
    std::fill_n(result, 8, 0.0f);
    if (tokens == 1) {
        // An odd-token prefill tail must keep dot_avx's four-chain GEMV order.
        dot_avx(w, row, inputs, 1, result);
        dot_avx(w, row + 1, inputs, 1, result + 4);
        return;
    }
    const auto group = w.bits == 32 ? w.cols : w.group;
    const auto groups = w.cols / group;
    for (std::size_t first = 0; first < w.cols; first += group) {
        __m256 a0 = _mm256_setzero_ps(), b0 = a0, c0 = a0, d0 = a0;
        __m256 a1 = a0, b1 = a0, c1 = a0, d1 = a0;
        std::size_t col = first;
        const auto end = first + group;
        for (; col + 8 <= end; col += 8) {
            const auto x0 = _mm256_loadu_ps(inputs + col);
            const auto x1 = tokens > 1 ? _mm256_loadu_ps(inputs + w.cols + col) : _mm256_setzero_ps();
            const auto x2 = tokens > 2 ? _mm256_loadu_ps(inputs + 2 * w.cols + col) : _mm256_setzero_ps();
            const auto x3 = tokens > 3 ? _mm256_loadu_ps(inputs + 3 * w.cols + col) : _mm256_setzero_ps();
            auto weight = weight8(w, row, col);
            a0 = _mm256_fmadd_ps(weight, x0, a0); b0 = _mm256_fmadd_ps(weight, x1, b0);
            c0 = _mm256_fmadd_ps(weight, x2, c0); d0 = _mm256_fmadd_ps(weight, x3, d0);
            weight = weight8(w, row + 1, col);
            a1 = _mm256_fmadd_ps(weight, x0, a1); b1 = _mm256_fmadd_ps(weight, x1, b1);
            c1 = _mm256_fmadd_ps(weight, x2, c1); d1 = _mm256_fmadd_ps(weight, x3, d1);
        }
        float partial[8] = {reduce(a0), reduce(b0), reduce(c0), reduce(d0),
                            reduce(a1), reduce(b1), reduce(c1), reduce(d1)};
        for (std::size_t r = 0; r < 2; ++r) {
            const auto scale = w.bits == 32 ? 1.0f : w.scales[(row + r) * groups + first / group];
            for (std::size_t k = col; k < end; ++k) {
                const auto value = w.value(row + r, k) / scale;
                for (std::size_t t = 0; t < tokens; ++t)
                    partial[r * 4 + t] += value * inputs[t * w.cols + k];
            }
            for (std::size_t t = 0; t < tokens; ++t) result[r * 4 + t] += partial[r * 4 + t] * scale;
        }
    }
}
template <unsigned Bits>
LEAF_AVX_TARGET void float_tile4_avx(const Weight& w, std::size_t row,
        const float* inputs, float* result) {
    static_assert(Bits == 32 || Bits == 8, "full-token float tile requires FP32 or INT8 weights");
    // A full four-token tile has no token-count/precision branches in its
    // column loop. Keep the same FMA chains, reductions and group scaling as
    // the generic tile; partial tiles and packed INT4 use that existing path.
    std::fill_n(result, 8, 0.0f);
    const auto group = Bits == 32 ? w.cols : w.group;
    const auto groups = w.cols / group;
    for (std::size_t first = 0; first < w.cols; first += group) {
        __m256 a0 = _mm256_setzero_ps(), b0 = a0, c0 = a0, d0 = a0;
        __m256 a1 = a0, b1 = a0, c1 = a0, d1 = a0;
        std::size_t col = first;
        const auto end = first + group;
        for (; col + 8 <= end; col += 8) {
            const auto x0 = _mm256_loadu_ps(inputs + col);
            const auto x1 = _mm256_loadu_ps(inputs + w.cols + col);
            const auto x2 = _mm256_loadu_ps(inputs + 2 * w.cols + col);
            const auto x3 = _mm256_loadu_ps(inputs + 3 * w.cols + col);
            __m256 weight;
            if constexpr (Bits == 32)
                weight = _mm256_loadu_ps(reinterpret_cast<const float*>(w.data) + row * w.cols + col);
            else
                weight = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                    _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w.data + row * w.cols + col))));
            a0 = _mm256_fmadd_ps(weight, x0, a0); b0 = _mm256_fmadd_ps(weight, x1, b0);
            c0 = _mm256_fmadd_ps(weight, x2, c0); d0 = _mm256_fmadd_ps(weight, x3, d0);
            if constexpr (Bits == 32)
                weight = _mm256_loadu_ps(reinterpret_cast<const float*>(w.data) + (row + 1) * w.cols + col);
            else
                weight = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                    _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w.data + (row + 1) * w.cols + col))));
            a1 = _mm256_fmadd_ps(weight, x0, a1); b1 = _mm256_fmadd_ps(weight, x1, b1);
            c1 = _mm256_fmadd_ps(weight, x2, c1); d1 = _mm256_fmadd_ps(weight, x3, d1);
        }
        float partial[8] = {reduce(a0), reduce(b0), reduce(c0), reduce(d0),
                            reduce(a1), reduce(b1), reduce(c1), reduce(d1)};
        for (std::size_t r = 0; r < 2; ++r) {
            const auto scale = Bits == 32 ? 1.0f : w.scales[(row + r) * groups + first / group];
            for (std::size_t k = col; k < end; ++k) {
                const auto value = w.value(row + r, k) / scale;
                for (std::size_t t = 0; t < 4; ++t)
                    partial[r * 4 + t] += value * inputs[t * w.cols + k];
            }
            for (std::size_t t = 0; t < 4; ++t) result[r * 4 + t] += partial[r * 4 + t] * scale;
        }
    }
}
__attribute__((always_inline)) LEAF_AVX_TARGET inline __m256i weight16(
        const Weight& w, std::size_t row, std::size_t col) {
    const auto index = row * w.cols + col;
    if (w.bits == 8) return _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w.data + index)));
    const auto packed = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w.data + index / 2));
    const auto nibble_mask = _mm_set1_epi8(15);
    const auto low = _mm_and_si128(packed, nibble_mask);
    const auto high = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
    const auto expanded = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(low, high));
    return _mm256_sub_epi16(_mm256_xor_si256(expanded, _mm256_set1_epi16(8)), _mm256_set1_epi16(8));
}
LEAF_AVX_TARGET std::int32_t reduce_integer(__m256i value) {
    auto half = _mm_add_epi32(_mm256_castsi256_si128(value), _mm256_extracti128_si256(value, 1));
    half = _mm_hadd_epi32(half, half); half = _mm_hadd_epi32(half, half);
    return _mm_cvtsi128_si32(half);
}
LEAF_AVX_TARGET void quant_dot_avx(const Weight& w, std::size_t row,
        const std::int16_t* inputs, std::size_t tokens, const float* input_scales, float* result) {
    std::fill_n(result, tokens, 0.0f);
    for (std::size_t first = 0; first < w.cols; first += w.group) {
        __m256i a = _mm256_setzero_si256(), b = a, c = a, d = a;
        std::size_t col = first;
        const auto end = first + w.group;
        if (tokens == 1) {
            for (; col + 64 <= end; col += 64) {
                a = _mm256_add_epi32(a, _mm256_madd_epi16(weight16(w, row, col), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + col))));
                b = _mm256_add_epi32(b, _mm256_madd_epi16(weight16(w, row, col + 16), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + col + 16))));
                c = _mm256_add_epi32(c, _mm256_madd_epi16(weight16(w, row, col + 32), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + col + 32))));
                d = _mm256_add_epi32(d, _mm256_madd_epi16(weight16(w, row, col + 48), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + col + 48))));
            }
            a = _mm256_add_epi32(_mm256_add_epi32(a, b), _mm256_add_epi32(c, d));
        }
        for (; col + 16 <= end; col += 16) {
            const auto weight = weight16(w, row, col);
            a = _mm256_add_epi32(a, _mm256_madd_epi16(weight, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + col))));
            if (tokens > 1) b = _mm256_add_epi32(b, _mm256_madd_epi16(weight, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + w.cols + col))));
            if (tokens > 2) c = _mm256_add_epi32(c, _mm256_madd_epi16(weight, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + 2 * w.cols + col))));
            if (tokens > 3) d = _mm256_add_epi32(d, _mm256_madd_epi16(weight, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + 3 * w.cols + col))));
        }
        std::int32_t accumulators[4] = {reduce_integer(a), reduce_integer(b), reduce_integer(c), reduce_integer(d)};
        for (; col < end; ++col) {
            const auto index = row * w.cols + col;
            int value = w.bits == 8 ? reinterpret_cast<const std::int8_t*>(w.data)[index] :
                                     (w.data[index / 2] >> ((index % 2) * 4)) & 15;
            if (w.bits == 4 && value >= 8) value -= 16;
            for (std::size_t t = 0; t < tokens; ++t) accumulators[t] += value * inputs[t * w.cols + col];
        }
        const float scale = w.scales[row * (w.cols / w.group) + first / w.group];
        for (std::size_t t = 0; t < tokens; ++t) result[t] += static_cast<float>(accumulators[t]) * input_scales[t] * scale;
    }
}
LEAF_AVX_TARGET void quant_tile_avx(const Weight& w, std::size_t row,
        const std::int16_t* inputs, std::size_t tokens, const float* input_scales, float* result) {
    // Four output rows share each pair of input vectors. Eight independent
    // sums fit in AVX2 registers without spilling the prefill micro-tile.
    std::fill_n(result, 8, 0.0f);
    const auto groups = w.cols / w.group;
    for (std::size_t first = 0; first < w.cols; first += w.group) {
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
        __m256i b0 = a0, b1 = a0, b2 = a0, b3 = a0;
        std::size_t col = first;
        const auto end = first + w.group;
        for (; col + 16 <= end; col += 16) {
            const auto x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + col));
            const auto x1 = tokens > 1
                ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + w.cols + col)) : _mm256_setzero_si256();
            auto weight = weight16(w, row, col);
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(weight, x0));
            b0 = _mm256_add_epi32(b0, _mm256_madd_epi16(weight, x1));
            weight = weight16(w, row + 1, col);
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(weight, x0));
            b1 = _mm256_add_epi32(b1, _mm256_madd_epi16(weight, x1));
            weight = weight16(w, row + 2, col);
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(weight, x0));
            b2 = _mm256_add_epi32(b2, _mm256_madd_epi16(weight, x1));
            weight = weight16(w, row + 3, col);
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(weight, x0));
            b3 = _mm256_add_epi32(b3, _mm256_madd_epi16(weight, x1));
        }
        std::int32_t sums[8] = {reduce_integer(a0), reduce_integer(b0),
            reduce_integer(a1), reduce_integer(b1), reduce_integer(a2), reduce_integer(b2),
            reduce_integer(a3), reduce_integer(b3)};
        for (; col < end; ++col) {
            for (std::size_t r = 0; r < 4; ++r) {
                const auto index = (row + r) * w.cols + col;
                int value = w.bits == 8 ? reinterpret_cast<const std::int8_t*>(w.data)[index] :
                                         (w.data[index / 2] >> ((index % 2) * 4)) & 15;
                if (w.bits == 4 && value >= 8) value -= 16;
                for (std::size_t t = 0; t < tokens; ++t) sums[r * 2 + t] += value * inputs[t * w.cols + col];
            }
        }
        for (std::size_t r = 0; r < 4; ++r) {
            const float scale = w.scales[(row + r) * groups + first / w.group];
            for (std::size_t t = 0; t < tokens; ++t)
                result[r * 2 + t] += static_cast<float>(sums[r * 2 + t]) * input_scales[t] * scale;
        }
    }
}
LEAF_AVX_TARGET void quantize_input_avx(const float* input, std::int16_t* output, std::size_t count, float multiplier) {
    std::size_t i = 0;
    const auto factor = _mm256_set1_ps(multiplier);
    for (; i + 8 <= count; i += 8) {
        const auto integers = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(input + i), factor));
        const auto halves = _mm_packs_epi32(_mm256_castsi256_si128(integers), _mm256_extracti128_si256(integers, 1));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i), halves);
    }
    for (; i < count; ++i) output[i] = static_cast<std::int16_t>(std::max(-127L, std::min(127L, std::lrint(input[i] * multiplier))));
}
#if LEAF_DECODER_VNNI
LEAF_VNNI_TARGET void prepare_signed_sums(Weight& weight) {
    const auto groups = weight.cols / weight.group;
    weight.signed_sums.resize(weight.rows * groups);
    const auto ones = _mm256_set1_epi8(1);
    for (std::size_t row = 0; row < weight.rows; ++row) {
        for (std::size_t group = 0; group < groups; ++group) {
            const auto first = group * weight.group, end = first + weight.group;
            auto accumulated = _mm256_setzero_si256();
            std::size_t col = first;
            for (; col + 32 <= end; col += 32) {
                const auto bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight.data + row * weight.cols + col));
                accumulated = _mm256_dpbusd_avx_epi32(accumulated, ones, bytes);
            }
            auto total = reduce_integer(accumulated);
            for (; col < end; ++col) total += reinterpret_cast<const std::int8_t*>(weight.data)[row * weight.cols + col];
            weight.signed_sums[row * groups + group] = total;
        }
    }
}
LEAF_AVX_TARGET void quantize_input_bytes(const float* input, std::uint8_t* output, std::size_t count, float multiplier) {
    const auto factor = _mm256_set1_ps(multiplier);
    const auto offset = _mm_set1_epi16(128);
    std::size_t col = 0;
    for (; col + 8 <= count; col += 8) {
        const auto integers = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(input + col), factor));
        const auto halves = _mm_packs_epi32(_mm256_castsi256_si128(integers), _mm256_extracti128_si256(integers, 1));
        const auto shifted = _mm_add_epi16(halves, offset);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(output + col), _mm_packus_epi16(shifted, shifted));
    }
    for (; col < count; ++col)
        output[col] = static_cast<std::uint8_t>(128 + std::max(-127L, std::min(127L, std::lrint(input[col] * multiplier))));
}
LEAF_VNNI_TARGET void quant_vnni_tile(const Weight& w, std::size_t row, std::size_t rows,
        const std::uint8_t* inputs, std::size_t tokens, const float* input_scales, float* result) {
    std::fill_n(result, 8, 0.0f);
    const auto groups = w.cols / w.group;
    for (std::size_t first = 0; first < w.cols; first += w.group) {
        __m256i a0 = _mm256_setzero_si256(), b0 = a0, c0 = a0, d0 = a0;
        __m256i a1 = a0, b1 = a0, c1 = a0, d1 = a0;
        std::size_t col = first;
        const auto end = first + w.group;
        for (; col + 32 <= end; col += 32) {
            const auto x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + col));
            const auto x1 = tokens > 1 ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + w.cols + col)) : _mm256_setzero_si256();
            const auto x2 = tokens > 2 ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + 2 * w.cols + col)) : _mm256_setzero_si256();
            const auto x3 = tokens > 3 ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + 3 * w.cols + col)) : _mm256_setzero_si256();
            auto weight = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.data + row * w.cols + col));
            a0 = _mm256_dpbusd_avx_epi32(a0, x0, weight);
            b0 = _mm256_dpbusd_avx_epi32(b0, x1, weight);
            c0 = _mm256_dpbusd_avx_epi32(c0, x2, weight);
            d0 = _mm256_dpbusd_avx_epi32(d0, x3, weight);
            if (rows > 1) {
                weight = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.data + (row + 1) * w.cols + col));
                a1 = _mm256_dpbusd_avx_epi32(a1, x0, weight);
                b1 = _mm256_dpbusd_avx_epi32(b1, x1, weight);
                c1 = _mm256_dpbusd_avx_epi32(c1, x2, weight);
                d1 = _mm256_dpbusd_avx_epi32(d1, x3, weight);
            }
        }
        std::int32_t sums[8] = {reduce_integer(a0), reduce_integer(b0), reduce_integer(c0), reduce_integer(d0),
            reduce_integer(a1), reduce_integer(b1), reduce_integer(c1), reduce_integer(d1)};
        for (; col < end; ++col)
            for (std::size_t r = 0; r < rows; ++r) {
                const auto value = reinterpret_cast<const std::int8_t*>(w.data)[(row + r) * w.cols + col];
                for (std::size_t t = 0; t < tokens; ++t) sums[r * 4 + t] += value * inputs[t * w.cols + col];
            }
        for (std::size_t r = 0; r < rows; ++r) {
            const auto group_index = (row + r) * groups + first / w.group;
            const auto correction = 128 * w.signed_sums[group_index];
            for (std::size_t t = 0; t < tokens; ++t)
                result[r * 4 + t] += static_cast<float>(sums[r * 4 + t] - correction) * input_scales[t] * w.scales[group_index];
        }
    }
}
#endif
#endif

void norm(const std::vector<float>& input, const Weight& weight, const Weight* bias,
          float epsilon, unsigned kind, std::vector<float>& output, std::size_t width) {
    output.resize(input.size());
    for (std::size_t row = 0; row < input.size() / width; ++row) {
        float mean = 0, square = 0;
        if (kind == 1) {
            for (std::size_t col = 0; col < width; ++col) mean += input[row * width + col];
            mean /= static_cast<float>(width);
        }
        for (std::size_t col = 0; col < width; ++col) {
            const float value = input[row * width + col] - mean; square += value * value;
        }
        const float scale = 1.0f / std::sqrt(square / static_cast<float>(width) + epsilon);
        for (std::size_t col = 0; col < width; ++col)
            output[row * width + col] = (input[row * width + col] - mean) * scale * weight.value(0, col)
                                        + (bias ? bias->value(0, col) : 0.0f);
    }
}
}  // namespace

struct Decoder::Impl {
    Mapping mapping;
    Workers workers;
    DecoderProfile profile;
    std::unordered_map<std::string, Weight> weights;
    std::vector<KVCache> caches;
    std::size_t h, f, heads, kv_heads, layers, vocab, max_positions, eos, dim;
    float theta, epsilon;
    unsigned norm_kind, activation, position_kind, gated, parallel, rotary_dim, position_offset, final_norm;
    float embedding_scale;
    bool avx = false;
    bool experimental_float_tiles = false;
    bool vnni = false, used_vnni = false;
    Dot dot = dot_scalar;
    QuantDot quant_dot = quant_dot_scalar;
    unsigned activation_precision;
    std::vector<std::int16_t> quant_input;
    std::vector<std::uint8_t> quant_bytes;
    std::vector<float> quant_scales;
    std::vector<float> transformed_input;
    leaf::kernels::TokenPanelsF32 packed_float_input;
    std::vector<float> rotary_frequencies, rotary_cosines, rotary_sines;
    std::vector<float> hidden, normalized, q, k, v, q_heads, k_heads, v_heads,
                       attention, projected, attn_projected, gate, up, mask;

    Impl(const std::string& path, unsigned threads, bool scalar, unsigned activation_bits)
        : mapping(path), workers(threads), activation_precision(activation_bits) {
        check(activation_bits == 8 || activation_bits == 32, "activation bits must be 8 or 32");
#if LEAF_DECODER_AVX
        avx = !scalar && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
        if (avx) { dot = dot_avx; quant_dot = quant_dot_avx; }
        // Unqualified experiments are explicit opt-ins, never default kernels.
        const char* tiles = std::getenv("LEAF_EXPERIMENTAL_FLOAT_TILES");
        const char* gemv = std::getenv("LEAF_EXPERIMENTAL_FLOAT_GEMV");
        experimental_float_tiles = avx && tiles && tiles[0];
        if (avx && gemv && gemv[0]) dot = dot_specialized_avx;
#if LEAF_DECODER_VNNI
        const char* disabled = std::getenv("LEAF_DISABLE_VNNI");
        vnni = avx && activation_bits == 8 && (!disabled || !disabled[0]) && __builtin_cpu_supports("avxvnni");
#endif
#else
        (void)scalar;
#endif
        Reader reader{mapping}; reader.require(8);
        check(std::memcmp(mapping.data, "LEAFDC02", 8) == 0, "unsupported artifact magic");
        reader.position = 8;
        check(reader.u32() == 2, "unsupported decoder artifact version");
        const auto count = reader.u32(); const auto base = reader.u64();
        h = reader.u32(); f = reader.u32(); heads = reader.u32(); kv_heads = reader.u32();
        layers = reader.u32(); vocab = reader.u32(); max_positions = reader.u32(); eos = reader.u32();
        dim = reader.u32(); check(reader.u32() == 0, "unsupported reserved configuration");
        theta = reader.f32(); epsilon = reader.f32();
        norm_kind = reader.u32(); activation = reader.u32(); position_kind = reader.u32();
        gated = reader.u32(); parallel = reader.u32(); rotary_dim = reader.u32();
        position_offset = reader.u32(); final_norm = reader.u32(); embedding_scale = reader.f32();
        check(h && f && heads && kv_heads && layers && vocab && dim && max_positions &&
              heads % kv_heads == 0 && std::uint64_t(h) == std::uint64_t(heads) * dim &&
              norm_kind <= 1 && activation <= 3 && position_kind <= 2 && gated <= 1 && parallel <= 1 &&
              rotary_dim <= dim && rotary_dim % 2 == 0 && (position_kind != 0 || rotary_dim > 0) &&
              final_norm <= 1 && std::isfinite(embedding_scale) && embedding_scale > 0 &&
              std::isfinite(theta) && theta > 0 && std::isfinite(epsilon) && epsilon > 0,
              "invalid decoder configuration");
        check(max_positions <= std::numeric_limits<std::size_t>::max() - position_offset &&
              max_positions <= std::numeric_limits<std::size_t>::max() / sizeof(float) / std::max({h, f, vocab, max_positions}),
              "decoder workspace dimensions overflow address space");
        check(base % 64 == 0 && base <= mapping.size && count <= mapping.size / 56,
              "invalid tensor table");
        for (std::size_t i = 0; i < count; ++i) {
            const auto name = reader.string();
            const auto rank = reader.u32(); Weight weight;
            weight.rows = reader.u32(); weight.cols = reader.u32(); weight.bits = reader.u32();
            weight.group = reader.u32(); const auto offset = reader.u64(); const auto size = reader.u64();
            const auto scale_offset = reader.u64(); const auto scale_count = reader.u64();
            check(rank == 2 && weight.rows && weight.cols &&
                  (weight.bits == 32 || weight.bits == 8 || weight.bits == 4), "invalid tensor " + name);
            const std::uint64_t elements = std::uint64_t(weight.rows) * weight.cols;
            check(elements <= std::numeric_limits<std::size_t>::max() / 4 &&
                  (weight.bits != 4 || weight.cols % 2 == 0) && size == elements * weight.bits / 8 &&
                  offset % 64 == 0 && offset <= mapping.size - base && size <= mapping.size - base - offset,
                  "invalid tensor data range " + name);
            if (weight.bits != 32) {
                check(weight.group && weight.cols % weight.group == 0 &&
                      (weight.bits != 4 || weight.group % 2 == 0) &&
                      scale_count == weight.rows * (weight.cols / weight.group) && scale_offset % 64 == 0 &&
                      scale_offset <= mapping.size - base && scale_count <= (mapping.size - base - scale_offset) / 4,
                      "invalid quantization metadata " + name);
                weight.scales = reinterpret_cast<const float*>(mapping.data + base + scale_offset);
                for (std::size_t j = 0; j < scale_count; ++j)
                    check(std::isfinite(weight.scales[j]) && weight.scales[j] > 0, "invalid quantization scale");
            } else check(scale_count == 0 && weight.group == 0, "unexpected FP32 quantization metadata");
            weight.data = mapping.data + base + offset;
#if LEAF_DECODER_VNNI
            if (vnni && weight.bits == 8 && weight.group <= 65536 &&
                name != "model.embed_tokens.weight" && name != "model.position_embeddings.weight")
                prepare_signed_sums(weight);
#endif
            if (name.size() >= 12 && name.compare(name.size() - 12, 12, ".input_scale") == 0) {
                check(weight.bits == 32 && weight.rows == 1, "invalid calibrated input scale tensor");
                const auto* factors = reinterpret_cast<const float*>(weight.data);
                for (std::size_t j = 0; j < weight.cols; ++j)
                    check(std::isfinite(factors[j]) && factors[j] > 0, "invalid calibrated input scale value");
            }
            check(weights.emplace(name, weight).second, "duplicate tensor " + name);
        }
        check(reader.position <= base, "metadata overlaps tensor payload");
        require_weight("model.embed_tokens.weight", vocab, h);
        require_weight("lm_head.weight", vocab, h);
        if (final_norm) require_weight("model.norm.weight", 1, h);
        if (position_kind == 1) require_weight("model.position_embeddings.weight", max_positions + position_offset, h);
        for (std::size_t i = 0; i < layers; ++i) {
            const auto prefix = "model.layers." + std::to_string(i) + ".";
            require_weight(prefix + "input_layernorm.weight", 1, h);
            require_weight(prefix + "post_attention_layernorm.weight", 1, h);
            require_weight(prefix + "self_attn.q_proj.weight", h, h);
            require_weight(prefix + "self_attn.k_proj.weight", kv_heads * dim, h);
            require_weight(prefix + "self_attn.v_proj.weight", kv_heads * dim, h);
            require_weight(prefix + "self_attn.o_proj.weight", h, h);
            if (gated) require_weight(prefix + "mlp.gate_proj.weight", f, h);
            require_weight(prefix + "mlp.up_proj.weight", f, h);
            require_weight(prefix + "mlp.down_proj.weight", h, f);
            caches.emplace_back(1, kv_heads, dim);
        }
        if (position_kind == 0) {
            rotary_frequencies.resize(rotary_dim / 2);
            for (std::size_t i = 0; i < rotary_dim / 2; ++i)
                rotary_frequencies[i] = 1.0f / std::pow(theta, static_cast<float>(2 * i) / rotary_dim);
        }
    }
    ~Impl() { profile.emit(); }
    void normalize(const std::vector<float>& input, const std::string& name, std::vector<float>& output) {
        DecoderProfile::Scope scope(profile, norm_kind ? DecoderProfile::Phase::LayerNorm : DecoderProfile::Phase::RMSNorm,
                                     input.size() / h);
        const auto found = weights.find(name + ".bias");
        const Weight* bias = found == weights.end() ? nullptr : &found->second;
        if (bias) check(bias->rows == 1 && bias->cols == h, "incompatible normalization bias");
        norm(input, weights.at(name + ".weight"), bias, epsilon, norm_kind, output, h);
    }
    float activate(float x) const {
        if (activation == 0) return x / (1.0f + std::exp(-x));
        if (activation == 1) return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475f));
        if (activation == 2) return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        return std::max(x, 0.0f);
    }
    const Weight& require_weight(const std::string& name, std::size_t rows, std::size_t cols) const {
        const auto found = weights.find(name);
        check(found != weights.end() && found->second.rows == rows && found->second.cols == cols,
              "missing or incompatible tensor " + name);
        return found->second;
    }
    void linear(const float* input, std::size_t tokens, const std::string& name, std::vector<float>& output) {
        const auto& weight = weights.at(name + ".weight");
        const auto phase = weight.bits == 32 ? DecoderProfile::Phase::LinearFP32 :
            weight.bits == 8 ? (activation_precision == 8 ? DecoderProfile::Phase::LinearW8A8 : DecoderProfile::Phase::LinearW8A32) :
                               (activation_precision == 8 ? DecoderProfile::Phase::LinearW4A8 : DecoderProfile::Phase::LinearW4A32);
        DecoderProfile::Scope scope(profile, phase, tokens);
        const auto found = weights.find(name + ".bias");
        const Weight* bias = found == weights.end() ? nullptr : &found->second;
        if (bias) check(bias->rows == 1 && bias->cols == weight.rows, "incompatible linear bias");
        const auto scaling = weights.find(name + ".input_scale");
        if (scaling != weights.end()) {
            const auto& scale = scaling->second;
            check(scale.bits == 32 && scale.rows == 1 && scale.cols == weight.cols,
                  "incompatible calibrated input scale");
            transformed_input.resize(tokens * weight.cols);
            const auto* factors = reinterpret_cast<const float*>(scale.data);
            for (std::size_t t = 0; t < tokens; ++t)
                for (std::size_t col = 0; col < weight.cols; ++col)
                    transformed_input[t * weight.cols + col] = input[t * weight.cols + col] * factors[col];
            input = transformed_input.data();
        }
        const bool integer = weight.bits != 32 && activation_precision == 8;
        const bool vnni_integer = integer && vnni && tokens > 1 && weight.bits == 8 && weight.group <= 65536;
        if (integer) {
            check(weight.group <= 131000, "integer accumulation exceeds safe input width");
            if (vnni_integer) quant_bytes.resize(tokens * weight.cols);
            else quant_input.resize(tokens * weight.cols);
            quant_scales.resize(tokens);
            for (std::size_t t = 0; t < tokens; ++t) {
                const auto* values = input + t * weight.cols;
                float maximum = 0;
                for (std::size_t col = 0; col < weight.cols; ++col) maximum = std::max(maximum, std::abs(values[col]));
                quant_scales[t] = std::max(maximum / 127.0f, 1e-12f);
#if LEAF_DECODER_AVX
#if LEAF_DECODER_VNNI
                if (vnni_integer) quantize_input_bytes(values, quant_bytes.data() + t * weight.cols, weight.cols, 1.0f / quant_scales[t]);
                else
#endif
                if (avx) quantize_input_avx(values, quant_input.data() + t * weight.cols, weight.cols, 1.0f / quant_scales[t]);
                else
#endif
                    for (std::size_t col = 0; col < weight.cols; ++col)
                        quant_input[t * weight.cols + col] = static_cast<std::int16_t>(std::max(-127L, std::min(127L, std::lrint(values[col] / quant_scales[t]))));
            }
        }
        const bool packed_float = experimental_float_tiles && weight.bits == 32 && tokens >= 8 &&
                                  (!bias || bias->bits == 32);
        if (packed_float) packed_float_input.pack(input, tokens, weight.cols, weight.cols);
        output.resize(tokens * weight.rows);
        used_vnni = used_vnni || vnni_integer;
        workers.run(weight.rows, [&](std::size_t first, std::size_t last) {
            if (packed_float) {
                leaf::kernels::gemm_token_panels_f32(
                    reinterpret_cast<const float*>(weight.data) + first * weight.cols,
                    last - first, weight.cols, weight.cols, packed_float_input,
                    output.data() + first, weight.rows,
                    bias ? reinterpret_cast<const float*>(bias->data) + first : nullptr);
                return;
            }
            std::size_t row = first;
#if LEAF_DECODER_VNNI
            if (vnni_integer) {
                constexpr std::size_t tile_rows = 2, tile_tokens = 4;
                float result[tile_rows * tile_tokens];
                for (; row < last; row += tile_rows) {
                    const auto rows = std::min(tile_rows, last - row);
                    for (std::size_t token = 0; token < tokens; token += tile_tokens) {
                        const auto count = std::min(tile_tokens, tokens - token);
                        quant_vnni_tile(weight, row, rows, quant_bytes.data() + token * weight.cols,
                                        count, quant_scales.data() + token, result);
                        for (std::size_t r = 0; r < rows; ++r)
                            for (std::size_t t = 0; t < count; ++t)
                                output[(token + t) * weight.rows + row + r] = result[r * tile_tokens + t] +
                                    (bias ? bias->value(0, row + r) : 0);
                    }
                }
                return;
            }
#endif
#if LEAF_DECODER_AVX
            // The opt-in token-panel experiment only changes FP32 GEMM above.
            // Keep short inputs and weight-only INT8 on their default paths so
            // a scoped before/after run isolates this kernel's real cost.
            if (integer && avx && tokens > 1) {
                float result[8];
                for (; row + 4 <= last; row += 4) {
                    for (std::size_t token = 0; token < tokens; token += 2) {
                        const auto count = std::min<std::size_t>(2, tokens - token);
                        quant_tile_avx(weight, row, quant_input.data() + token * weight.cols,
                                       count, quant_scales.data() + token, result);
                        for (std::size_t r = 0; r < 4; ++r)
                            for (std::size_t t = 0; t < count; ++t)
                                output[(token + t) * weight.rows + row + r] = result[r * 2 + t] +
                                    (bias ? bias->value(0, row + r) : 0);
                    }
                }
            }
#endif
            float result[4];
            for (; row < last; ++row) {
                for (std::size_t token = 0; token < tokens; token += 4) {
                    const auto count = std::min<std::size_t>(4, tokens - token);
                    if (integer) quant_dot(weight, row, quant_input.data() + token * weight.cols,
                                           count, quant_scales.data() + token, result);
                    else dot(weight, row, input + token * weight.cols, count, result);
                    for (std::size_t t = 0; t < count; ++t)
                        output[(token + t) * weight.rows + row] = result[t] + (bias ? bias->value(0, row) : 0);
                }
            }
        });
    }
    void prepare_rotary_tables(std::size_t tokens, std::size_t position) {
        if (position_kind != 0) return;
        DecoderProfile::Scope scope(profile, DecoderProfile::Phase::RoPE, tokens);
        const std::size_t half = rotary_dim / 2;
        check(half && tokens <= std::numeric_limits<std::size_t>::max() / sizeof(float) / half,
              "rotary coefficient workspace dimensions overflow address space");
        const std::size_t count = tokens * half;
        check(count <= rotary_cosines.max_size() && count <= rotary_sines.max_size(),
              "rotary coefficient workspace exceeds vector limits");
        // Logical size follows the current chunk, not the accumulated KV cache
        // or max_positions. Reuse capacity like the other decoder workspaces.
        rotary_cosines.resize(count);
        rotary_sines.resize(count);
        for (std::size_t token = 0; token < tokens; ++token) {
            for (std::size_t i = 0; i < half; ++i) {
                // Keep the original float expressions and libm calls: changing
                // precision, angle order, or sin/cos implementation can affect
                // whole-model parity even if a local rotation looks close.
                const float angle = static_cast<float>(position + token) * rotary_frequencies[i];
                const float cosine = std::cos(angle), sine = std::sin(angle);
                rotary_cosines[token * half + i] = cosine;
                rotary_sines[token * half + i] = sine;
            }
        }
    }
    void rotary(std::vector<float>& values, std::vector<float>& reordered,
                std::size_t nheads, std::size_t tokens) {
        DecoderProfile::Scope scope(profile, position_kind == 0 ? DecoderProfile::Phase::RoPE : DecoderProfile::Phase::PositionReorder,
                                     tokens);
        reordered.resize(values.size());
        for (std::size_t token = 0; token < tokens; ++token) {
            for (std::size_t head = 0; head < nheads; ++head) {
                for (std::size_t i = 0; i < dim; ++i) {
                    reordered[(head * tokens + token) * dim + i] = values[(token * nheads + head) * dim + i];
                }
                if (position_kind != 0) continue;
                for (std::size_t i = 0; i < rotary_dim / 2; ++i) {
                    const auto coefficient = token * (rotary_dim / 2) + i;
                    const float cosine = rotary_cosines[coefficient], sine = rotary_sines[coefficient];
                    const auto from = (token * nheads + head) * dim + i;
                    const auto to = (head * tokens + token) * dim + i;
                    const float a = values[from], b = values[from + rotary_dim / 2];
                    reordered[to] = a * cosine - b * sine;
                    reordered[to + rotary_dim / 2] = b * cosine + a * sine;
                }
            }
        }
    }
    std::vector<float> forward(const std::vector<std::uint32_t>& ids, bool all_logits) {
        check(!ids.empty(), "input tokens are empty");
        const std::size_t tokens = ids.size(), position = caches[0].length();
        check(position <= max_positions && tokens <= max_positions - position, "context length exceeds artifact limit");
        for (auto id : ids) check(id < vocab, "token id exceeds vocabulary");
        profile.forward_tokens(tokens);
        DecoderProfile::Scope forward_scope(profile, DecoderProfile::Phase::Forward, tokens);
        prepare_rotary_tables(tokens, position);
        {
            DecoderProfile::Scope scope(profile, DecoderProfile::Phase::Embedding, tokens);
            hidden.resize(tokens * h);
            const auto& embedding = weights.at("model.embed_tokens.weight");
            for (std::size_t t = 0; t < tokens; ++t)
                for (std::size_t col = 0; col < h; ++col) {
                    hidden[t * h + col] = embedding.value(ids[t], col) * embedding_scale;
                    if (position_kind == 1) hidden[t * h + col] +=
                        weights.at("model.position_embeddings.weight").value(position + t + position_offset, col);
                }
        }
        const auto total = position + tokens;
        {
            DecoderProfile::Scope scope(profile, DecoderProfile::Phase::CausalMask, tokens);
            mask.assign(tokens * total, 0.0f);
            for (std::size_t t = 0; t < tokens; ++t)
                std::fill_n(mask.data() + t * total, position + t + 1, 1.0f);
        }
        const std::size_t mask_shape[4] = {1, 1, tokens, total};
        for (std::size_t layer = 0; layer < layers; ++layer) {
            const auto prefix = "model.layers." + std::to_string(layer) + ".";
            normalize(hidden, prefix + "input_layernorm", normalized);
            linear(normalized.data(), tokens, prefix + "self_attn.q_proj", q);
            linear(normalized.data(), tokens, prefix + "self_attn.k_proj", k);
            linear(normalized.data(), tokens, prefix + "self_attn.v_proj", v);
            rotary(q, q_heads, heads, tokens); rotary(k, k_heads, kv_heads, tokens);
            v_heads.resize(v.size());
            for (std::size_t head = 0; head < kv_heads; ++head)
                for (std::size_t t = 0; t < tokens; ++t)
                    std::copy_n(v.data() + (t * kv_heads + head) * dim, dim,
                                v_heads.data() + (head * tokens + t) * dim);
            {
                DecoderProfile::Scope scope(profile, DecoderProfile::Phase::KVAppend, tokens);
                caches[layer].append(k_heads.data(), v_heads.data(), tokens);
            }
            {
                DecoderProfile::Scope scope(profile, DecoderProfile::Phase::Attention, tokens);
                attention.resize(tokens * h);
                caches[layer].attend(q_heads.data(), mask.data(), attention.data(), heads, tokens,
                                      // The shared fused graph kernel scales both Q and K.
                                      mask_shape, 1.0f / std::sqrt(std::sqrt(static_cast<float>(dim))));
            }
            linear(attention.data(), tokens, prefix + "self_attn.o_proj", attn_projected);
            if (!parallel) {
                DecoderProfile::Scope scope(profile, DecoderProfile::Phase::Residual, tokens);
                for (std::size_t j = 0; j < hidden.size(); ++j) hidden[j] += attn_projected[j];
            }
            normalize(hidden, prefix + "post_attention_layernorm", normalized);
            linear(normalized.data(), tokens, prefix + "mlp.up_proj", up);
            if (gated) {
                linear(normalized.data(), tokens, prefix + "mlp.gate_proj", gate);
                DecoderProfile::Scope scope(profile, DecoderProfile::Phase::Activation, tokens);
                for (std::size_t j = 0; j < gate.size(); ++j) up[j] *= activate(gate[j]);
            } else {
                DecoderProfile::Scope scope(profile, DecoderProfile::Phase::Activation, tokens);
                for (auto& value : up) value = activate(value);
            }
            linear(up.data(), tokens, prefix + "mlp.down_proj", projected);
            {
                DecoderProfile::Scope scope(profile, DecoderProfile::Phase::Residual, tokens);
                for (std::size_t j = 0; j < hidden.size(); ++j) hidden[j] += projected[j] + (parallel ? attn_projected[j] : 0.0f);
            }
        }
        if (final_norm) normalize(hidden, "model.norm", normalized);
        else normalized = hidden;
        std::vector<float> logits;
        linear(normalized.data() + (all_logits ? 0 : (tokens - 1) * h), all_logits ? tokens : 1,
               "lm_head", logits);
        return logits;
    }
};

Decoder::Decoder(const std::string& path, unsigned threads, bool force_scalar, unsigned activation_bits)
    : impl_(std::make_unique<Impl>(path, threads, force_scalar, activation_bits)) {}
Decoder::~Decoder() = default;
void Decoder::reset() { for (auto& cache : impl_->caches) cache.reset(); }
std::vector<float> Decoder::forward(const std::vector<std::uint32_t>& tokens, bool all_logits) {
    return impl_->forward(tokens, all_logits);
}
std::uint32_t Decoder::vocabulary_size() const { return static_cast<std::uint32_t>(impl_->vocab); }
std::uint32_t Decoder::eos_token() const { return static_cast<std::uint32_t>(impl_->eos); }
std::size_t Decoder::cache_tokens() const { return impl_->caches[0].length(); }
std::uint64_t Decoder::weight_bytes() const { return impl_->mapping.size; }
bool Decoder::uses_avx2() const { return impl_->avx; }
bool Decoder::uses_vnni() const { return impl_->used_vnni; }
unsigned Decoder::activation_bits() const { return impl_->activation_precision; }
}  // namespace leaf::runtime
