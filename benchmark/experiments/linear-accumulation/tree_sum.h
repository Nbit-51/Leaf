#pragma once
#include "blocked_sum.h"

namespace leaf::experiments {
#ifndef LEAF_ACCUMULATION_TREE_CHUNK
#define LEAF_ACCUMULATION_TREE_CHUNK 128
#endif
inline constexpr std::size_t tree_chunk_size = LEAF_ACCUMULATION_TREE_CHUNK;
static_assert(tree_chunk_size >= 1, "tree reduction requires a nonempty chunk");
#if defined(__x86_64__) && defined(__GNUC__)
namespace tree_sum_detail {
template <unsigned Rows>
__attribute__((target("avx2,fma"))) inline void tile(
        const float* weights, std::size_t columns, std::size_t weight_stride,
        const float* packed, std::size_t count, float* output,
        std::size_t output_stride, const float* bias, float* levels) {
    alignas(32) float part[Rows * 16];
    std::size_t blocks = 0;
    unsigned depth = 0;
    for (std::size_t first = 0; first < columns;) {
        const auto width = std::min(tree_chunk_size, columns - first);
        blocked_sum_detail::partial<Rows>(weights + first, width, weight_stride, packed + first * 16, part);
        std::size_t occupied = blocks;
        unsigned level = 0;
        // Binary carries form a balanced FP32 reduction of the short partials.
        // Only occupied levels are read; no clearing or FP64 conversions.
        for (; occupied & 1; occupied >>= 1, ++level)
            for (unsigned lane = 0; lane < Rows * 16; lane += 8)
                _mm256_store_ps(part + lane, _mm256_add_ps(_mm256_load_ps(levels + level * 96 + lane),
                                                         _mm256_load_ps(part + lane)));
        for (unsigned lane = 0; lane < Rows * 16; lane += 8)
            _mm256_store_ps(levels + level * 96 + lane, _mm256_load_ps(part + lane));
        depth = std::max(depth, level + 1);
        ++blocks;
        first += width;
    }
    bool initialized = false;
    for (unsigned level = depth; level-- > 0;) {
        if (!(blocks & (std::size_t(1) << level))) continue;
        for (unsigned lane = 0; lane < Rows * 16; lane += 8) {
            auto value = _mm256_load_ps(levels + level * 96 + lane);
            if (initialized) value = _mm256_add_ps(_mm256_load_ps(part + lane), value);
            _mm256_store_ps(part + lane, value);
        }
        initialized = true;
    }
    for (std::size_t token = 0; token < count; ++token)
        for (unsigned row = 0; row < Rows; ++row)
            output[token * output_stride + row] = part[row * 16 + token] + (bias ? bias[row] : 0.0f);
}

__attribute__((target("avx2,fma"))) inline void gemm(
        const float* weights, std::size_t rows, std::size_t columns,
        std::size_t weight_stride, const leaf::kernels::TokenPanelsF32& input,
        float* output, std::size_t output_stride, const float* bias) {
    // Reserve once per matrix call/worker, never allocate per output tile.
    // Only log2(ceil(K/128))+1 levels are touched. Capacity covers size_t K.
    alignas(32) float levels[std::numeric_limits<std::size_t>::digits * 96];
    for (std::size_t row = 0; row < rows;) {
        const auto live = std::min<std::size_t>(6, rows - row);
        for (std::size_t panel = 0; panel < input.panels(); ++panel) {
            const auto first = panel * 16;
            const auto count = std::min<std::size_t>(16, input.tokens() - first);
            const auto* w = weights + row * weight_stride;
            auto* y = output + first * output_stride + row;
            const auto* b = bias ? bias + row : nullptr;
            switch (live) {
            case 1: tile<1>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b, levels); break;
            case 2: tile<2>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b, levels); break;
            case 3: tile<3>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b, levels); break;
            case 4: tile<4>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b, levels); break;
            case 5: tile<5>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b, levels); break;
            case 6: tile<6>(w, columns, weight_stride, input.panel(panel), count, y, output_stride, b, levels); break;
            }
        }
        row += live;
    }
}
template <unsigned Rows>
__attribute__((target("avx2,fma"), always_inline)) inline void cached_chunk(
        const float* weights, std::size_t columns, std::size_t weight_stride,
        const float* packed, std::size_t blocks, float* levels) {
    alignas(32) float part[Rows * 16];
    blocked_sum_detail::partial<Rows>(weights, columns, weight_stride, packed, part);
    unsigned level = 0;
    for (; blocks & 1; blocks >>= 1, ++level)
        for (unsigned lane = 0; lane < Rows * 16; lane += 8)
            _mm256_store_ps(part + lane, _mm256_add_ps(_mm256_load_ps(levels + level * 96 + lane),
                                                     _mm256_load_ps(part + lane)));
    for (unsigned lane = 0; lane < Rows * 16; lane += 8)
        _mm256_store_ps(levels + level * 96 + lane, _mm256_load_ps(part + lane));
}

template <unsigned Rows>
__attribute__((target("avx2,fma"), always_inline)) inline void cached_store(
        const float* levels, std::size_t blocks, unsigned depth, std::size_t count,
        float* output, std::size_t stride, const float* bias) {
    alignas(32) float part[Rows * 16];
    bool initialized = false;
    for (unsigned level = depth; level-- > 0;) {
        if (!(blocks & (std::size_t(1) << level))) continue;
        for (unsigned lane = 0; lane < Rows * 16; lane += 8) {
            auto value = _mm256_load_ps(levels + level * 96 + lane);
            if (initialized) value = _mm256_add_ps(_mm256_load_ps(part + lane), value);
            _mm256_store_ps(part + lane, value);
        }
        initialized = true;
    }
    for (std::size_t token = 0; token < count; ++token)
        for (unsigned row = 0; row < Rows; ++row)
            output[token * stride + row] = part[row * 16 + token] + (bias ? bias[row] : 0.0f);
}

__attribute__((target("avx2,fma"))) inline void gemm_cached(
        const float* weights, std::size_t rows, std::size_t columns,
        std::size_t weight_stride, const leaf::kernels::TokenPanelsF32& input,
        float* output, std::size_t output_stride, const float* bias) {
    // At K=256, 12 weight rows and one 16-token input chunk use 28 KiB.
    // Two independent trees touch a few KiB more. Reuse the input chunk in L1
    // without the older 96-row variant's larger weights/totals working set.
    constexpr std::size_t level_capacity = std::numeric_limits<std::size_t>::digits * 96;
    alignas(32) float levels[2 * level_capacity];
    for (std::size_t row = 0; row < rows;) {
        const auto live = std::min<std::size_t>(12, rows - row);
        for (std::size_t panel = 0; panel < input.panels(); ++panel) {
            const auto token = panel * 16;
            const auto count = std::min<std::size_t>(16, input.tokens() - token);
            std::size_t blocks = 0;
            for (std::size_t col = 0; col < columns;) {
                const auto width = std::min(tree_chunk_size, columns - col);
                const auto* x = input.panel(panel) + col * 16;
                for (std::size_t local = 0; local < live; local += 6) {
                    const auto* w = weights + (row + local) * weight_stride + col;
                    auto* tree = levels + (local / 6) * level_capacity;
                    switch (std::min<std::size_t>(6, live - local)) {
                    case 1: cached_chunk<1>(w, width, weight_stride, x, blocks, tree); break;
                    case 2: cached_chunk<2>(w, width, weight_stride, x, blocks, tree); break;
                    case 3: cached_chunk<3>(w, width, weight_stride, x, blocks, tree); break;
                    case 4: cached_chunk<4>(w, width, weight_stride, x, blocks, tree); break;
                    case 5: cached_chunk<5>(w, width, weight_stride, x, blocks, tree); break;
                    case 6: cached_chunk<6>(w, width, weight_stride, x, blocks, tree); break;
                    }
                }
                ++blocks;
                col += width;
            }
            unsigned depth = 0;
            for (auto remaining = blocks; remaining; remaining >>= 1) ++depth;
            for (std::size_t local = 0; local < live; local += 6) {
                const auto* tree = levels + (local / 6) * level_capacity;
                auto* y = output + token * output_stride + row + local;
                const auto* b = bias ? bias + row + local : nullptr;
                switch (std::min<std::size_t>(6, live - local)) {
                case 1: cached_store<1>(tree, blocks, depth, count, y, output_stride, b); break;
                case 2: cached_store<2>(tree, blocks, depth, count, y, output_stride, b); break;
                case 3: cached_store<3>(tree, blocks, depth, count, y, output_stride, b); break;
                case 4: cached_store<4>(tree, blocks, depth, count, y, output_stride, b); break;
                case 5: cached_store<5>(tree, blocks, depth, count, y, output_stride, b); break;
                case 6: cached_store<6>(tree, blocks, depth, count, y, output_stride, b); break;
                }
            }
        }
        row += live;
    }
}
} // namespace tree_sum_detail
#endif

inline void gemm_tree_sum(const float* weights, std::size_t rows,
        std::size_t columns, std::size_t weight_stride,
        const leaf::kernels::TokenPanelsF32& input, float* output,
        std::size_t output_stride, const float* bias = nullptr, bool use_avx2 = true) {
    if (!weights || !output || !rows || !columns || columns != input.columns() ||
        !input.tokens() || weight_stride < columns || output_stride < rows)
        throw std::invalid_argument("tree-sum GEMM requires compatible nonempty strided matrices");
    leaf::kernels::token_panel_detail::extent(rows, weight_stride, columns);
    leaf::kernels::token_panel_detail::extent(input.tokens(), output_stride, rows);
#if defined(__x86_64__) && defined(__GNUC__)
    if (use_avx2 && leaf::kernels::token_panel_f32_avx2_available()) {
#ifdef LEAF_ACCUMULATION_TREE_CACHE
        tree_sum_detail::gemm_cached(weights, rows, columns, weight_stride, input, output, output_stride, bias);
#else
        tree_sum_detail::gemm(weights, rows, columns, weight_stride, input, output, output_stride, bias);
#endif
        return;
    }
#endif
    leaf::kernels::gemm_token_panels_f32_row_reuse(
        weights, rows, columns, weight_stride, input, output, output_stride, bias, use_avx2);
}
} // namespace leaf::experiments
