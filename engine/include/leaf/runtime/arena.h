#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace leaf::runtime {

class Arena {
 public:
  explicit Arena(std::size_t bytes, std::size_t alignment = 64)
      : storage_(bytes + alignment), size_(bytes), alignment_(alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
      throw std::invalid_argument("arena alignment must be a power of two");
    }
    const auto raw = reinterpret_cast<std::uintptr_t>(storage_.data());
    const auto aligned = (raw + alignment - 1) & ~(alignment - 1);
    base_ = reinterpret_cast<std::byte*>(aligned);
  }

  void* at(std::size_t offset, std::size_t bytes) {
    if (offset % alignment_ != 0 || offset > size_ || bytes > size_ - offset) {
      throw std::out_of_range("memory-plan allocation is outside the arena");
    }
    return base_ + offset;
  }

  std::size_t size() const noexcept { return size_; }

 private:
  std::vector<std::byte> storage_;
  std::byte* base_{};
  std::size_t size_{};
  std::size_t alignment_{};
};

}  // namespace leaf::runtime
