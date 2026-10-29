// Packing moves bits; it must not perform floating-point arithmetic. This is
// deliberately separate from GEMM/quality tests, where nonfinite inputs fail.
#include "leaf/kernels/token_panel.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
static_assert(sizeof(float) == sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559,
              "The FP32 packer requires IEEE binary32 values");

constexpr std::array<std::uint32_t, 18> patterns{{
    0x00000000u, 0x80000000u, // Positive and negative zero.
    0x00000001u, 0x80000001u, 0x007fffffu, 0x807fffffu, // Subnormals.
    0x00800000u, 0x80800000u, 0x3f800000u, 0xbf800000u,
    0x3eaaaaabu, 0x7f7fffffu, 0xff7fffffu,
    0x7f800000u, 0xff800000u, // Infinities.
    0x7fc12345u, 0xffc54321u, 0x7f812345u // NaN signs/payloads, including sNaN.
}};
constexpr std::uint32_t padding_bits = 0xffc0dead;
std::size_t checked_values = 0;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void store(float& destination, std::uint32_t bits) {
    std::memcpy(&destination, &bits, sizeof(bits));
}

std::uint32_t bits(const float& value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

void fill(std::vector<float>& input, std::size_t tokens, std::size_t columns,
          std::size_t stride, std::size_t generation) {
    for (auto& value : input) store(value, padding_bits);
    for (std::size_t token = 0; token < tokens; ++token)
        for (std::size_t column = 0; column < columns; ++column)
            store(input[token * stride + column],
                  patterns[(token * 11 + column * 7 + generation) % patterns.size()]);
}

void verify(const leaf::kernels::TokenPanelsF32& packed, const std::vector<float>& input,
            std::size_t tokens, std::size_t columns, std::size_t stride) {
    const auto panels = (tokens + 15) / 16;
    require(packed.tokens() == tokens && packed.columns() == columns && packed.panels() == panels,
            "Repacking exposed stale logical extents");
    require(packed.storage_size() == panels * columns * 16, "Incorrect active packed size");
    for (std::size_t panel = 0; panel < panels; ++panel)
        for (std::size_t column = 0; column < columns; ++column)
            for (std::size_t lane = 0; lane < 16; ++lane) {
                const auto token = panel * 16 + lane;
                const auto expected = token < tokens ? bits(input[token * stride + column]) : 0u;
                require(bits(packed.panel(panel)[column * 16 + lane]) == expected,
                        "Packing changed FP32 bits, copied stride padding or retained an inactive lane");
                ++checked_values;
            }
}

void equivalent_bits() {
    leaf::kernels::TokenPanelsF32 scalar, vector;
    constexpr std::size_t capacity = 65 * 532;
    std::vector<float> input(capacity);
    const auto* const address = input.data();

    // Prime a larger workspace, then use many smaller/misaligned shapes. The
    // internal high-water allocation must not leak into the active extents.
    fill(input, 65, 513, 532, 0);
    scalar.pack(input.data(), 65, 513, 532, false);
    vector.pack(input.data(), 65, 513, 532, true);
    std::size_t generation = 1;
    for (const auto tokens : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 23u, 24u, 25u, 31u, 32u, 33u})
        for (const auto columns : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u, 257u})
            for (const auto gap : {0u, 3u, 19u}) {
                const auto stride = columns + gap;
                fill(input, tokens, columns, stride, generation++);
                const auto saved = input;
                scalar.pack(input.data(), tokens, columns, stride, false);
                vector.pack(input.data(), tokens, columns, stride, true);
                verify(scalar, input, tokens, columns, stride);
                verify(vector, input, tokens, columns, stride);
                for (std::size_t panel = 0; panel < scalar.panels(); ++panel)
                    require(std::memcmp(scalar.panel(panel), vector.panel(panel), columns * 16 * sizeof(float)) == 0,
                            "Scalar and SIMD packed representations differ");
                require(std::memcmp(saved.data(), input.data(), input.size() * sizeof(float)) == 0,
                        "Packer changed caller source bits");
                require(input.data() == address, "Test unexpectedly replaced the source allocation");
            }
}

void same_address_and_path_switches() {
    leaf::kernels::TokenPanelsF32 packed;
    std::vector<float> input(65 * 532);
    const auto* const address = input.data();
    // A new pack at the same source address must observe new values. Switching
    // SIMD/scalar paths cannot make old values or old padding reusable.
    const std::size_t shapes[][3] = {{33, 257, 276}, {8, 8, 11}, {16, 17, 20},
                                    {1, 1, 1}, {65, 513, 532}, {17, 9, 28}};
    for (std::size_t generation = 0; generation < 12; ++generation) {
        const auto& shape = shapes[generation % 6];
        fill(input, shape[0], shape[1], shape[2], generation * 5);
        packed.pack(input.data(), shape[0], shape[1], shape[2], generation % 2 != 0);
        verify(packed, input, shape[0], shape[1], shape[2]);
        // Repeat identical dimensions but change every source pattern.
        fill(input, shape[0], shape[1], shape[2], generation * 5 + 1);
        packed.pack(input.data(), shape[0], shape[1], shape[2], generation % 2 == 0);
        verify(packed, input, shape[0], shape[1], shape[2]);
        require(input.data() == address, "Same-address test source moved");
    }
}
} // namespace

int main() {
    try {
        equivalent_bits();
        same_address_and_path_switches();
        std::cout << "PASS input packing: raw IEEE bits, SIMD/scalar, same-address updates, tails and high-water reuse; "
                  << checked_values << " values checked\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Input packing test: " << error.what() << '\n';
        return 1;
    }
}
