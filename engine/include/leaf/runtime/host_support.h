#pragma once
// Host facilities shared by decoder and embedding runtimes; no model semantics.
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
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

} // namespace leaf::runtime::detail
