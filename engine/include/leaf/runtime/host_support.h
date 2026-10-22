#pragma once
// Host facilities shared by decoder and embedding runtimes; no model semantics.
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <chrono>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "leaf/runtime/worker_affinity.h"
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
namespace leaf::runtime::detail {
inline void check(bool valid, const std::string& message) {
    if (!valid) throw std::runtime_error("Leaf runtime: " + message);
}
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
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
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
    explicit Workers(unsigned threads, const std::vector<unsigned>& cpus = {}) : threads_(threads) {
        check(threads >= 1 && threads <= 64, "thread count must be 1..64");
        validate_worker_cpus(cpus, threads);
        if (!cpus.empty()) caller_cpu_ = cpus.front();
        try {
        for (unsigned i = 1; i < threads_; ++i) {
            if (cpus.empty()) {
                workers_.emplace_back([this] { worker_loop(); });
            } else {
                std::promise<void> startup;
                auto ready = startup.get_future();
                workers_.emplace_back([this, cpu=cpus[i], startup=std::move(startup)]() mutable {
                    std::optional<AffinityScope> binding;
                    try { binding.emplace(cpu); startup.set_value(); }
                    catch (...) { startup.set_exception(std::current_exception()); return; }
                    worker_loop();
                });
                // Fail construction cleanly if any owned thread cannot bind.
                ready.get();
            }
        }
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
    AffinityScope bind_caller() const { return AffinityScope(caller_cpu_); }
    void run(std::size_t count, std::function<void(std::size_t, std::size_t)> task) {
        AffinityScope caller(caller_cpu_);
        if (threads_ == 1) { task(0, count); caller.restore(); return; }
        { std::lock_guard<std::mutex> lock(mutex_);
          task_ = std::move(task); task_error_ = nullptr;
          count_ = count; next_ = 0; pending_ = threads_ - 1; ++epoch_;
          if (spins()) visible_epoch_.store(epoch_, std::memory_order_release); }
        ready_.notify_all(); drain();
        if (spins()) spin_until(completed_epoch_, epoch_, true);
        std::unique_lock<std::mutex> lock(mutex_);
        finished_.wait(lock, [&] { return pending_ == 0; });
        const auto error = task_error_;
        lock.unlock(); caller.restore();
        if (error) std::rethrow_exception(error);
    }
private:
    void worker_loop() {
        std::size_t observed = 0;
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [&] { return stop_ || epoch_ != observed; });
            if (stop_) return;
            observed = epoch_; lock.unlock(); drain(); lock.lock();
            if (--pending_ == 0) {
                if (spins()) completed_epoch_.store(observed, std::memory_order_release);
                finished_.notify_one();
            }
            lock.unlock();
            if (spins()) spin_until(visible_epoch_, observed, false);
        }
    }
    bool spins() const noexcept { return caller_cpu_ != unbound_cpu && threads_ > 1; }
    static void spin_until(const std::atomic<std::size_t>& counter, std::size_t target, bool equal) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(50);
        unsigned checks = 0;
        for (;;) {
            if ((counter.load(std::memory_order_acquire) == target) == equal) return;
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
            _mm_pause();
#else
            std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
            if ((++checks & 63) == 0 && std::chrono::steady_clock::now() >= deadline) return;
        }
    }
    void drain() {
        for (;;) {
            const std::size_t first = next_.fetch_add(32, std::memory_order_relaxed);
            if (first >= count_) return;
            try { task_(first, std::min(first + 32, count_)); }
            catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!task_error_) task_error_ = std::current_exception();
                return;
            }
        }
    }
    unsigned threads_;
    unsigned caller_cpu_ = unbound_cpu;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable ready_, finished_;
    bool stop_ = false;
    std::size_t epoch_ = 0, count_ = 0;
    unsigned pending_ = 0;
    std::atomic<std::size_t> next_{0};
    std::function<void(std::size_t, std::size_t)> task_;
    std::exception_ptr task_error_;
    std::atomic<std::size_t> visible_epoch_{0}, completed_epoch_{0};
};

} // namespace leaf::runtime::detail
