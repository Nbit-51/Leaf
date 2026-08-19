#include "im2col.h"

namespace leaf {

void conv2d_output_shape(
    size_t in_h, size_t in_w,
    size_t kh, size_t kw,
    size_t pad_h0, size_t pad_w0, size_t pad_h1, size_t pad_w1,
    size_t stride_h, size_t stride_w,
    size_t dilation_h, size_t dilation_w,
    size_t& out_h, size_t& out_w
) {
    size_t padded_h = in_h + pad_h0 + pad_h1;
    size_t padded_w = in_w + pad_w0 + pad_w1;
    out_h = (padded_h - (dilation_h * (kh - 1) + 1)) / stride_h + 1;
    out_w = (padded_w - (dilation_w * (kw - 1) + 1)) / stride_w + 1;
}

// Direct (non-vectorized) im2col. This runs once per conv layer per
// image and is memory-bound, not compute-bound -- the payoff is in the
// GEMM it feeds, not in this transform itself, so it's left scalar for
// correctness first. Can be revisited if profiling shows it's a real
// bottleneck once the full pipeline is measured end-to-end.
void im2col(
    const float* input, size_t in_c, size_t in_h, size_t in_w,
    size_t kh, size_t kw,
    size_t pad_h0, size_t pad_w0, size_t pad_h1, size_t pad_w1,
    size_t stride_h, size_t stride_w,
    size_t dilation_h, size_t dilation_w,
    float* col_out
) {
    size_t out_h, out_w;
    conv2d_output_shape(
        in_h, in_w, kh, kw, pad_h0, pad_w0, pad_h1, pad_w1,
        stride_h, stride_w, dilation_h, dilation_w, out_h, out_w
    );

    size_t out_spatial = out_h * out_w;
    size_t patch_size = in_c * kh * kw;

    for (size_t c = 0; c < in_c; ++c) {
        for (size_t ki = 0; ki < kh; ++ki) {
            for (size_t kj = 0; kj < kw; ++kj) {
                // Row index into col_out for this (channel, kernel_row,
                // kernel_col) combination -- matches weight layout
                // (OC, IC, KH, KW) flattened to (IC*KH*KW) per output
                // channel, so gemm_f32(weights, col_out) lines up
                // directly with no extra reshuffling.
                size_t row = c * kh * kw + ki * kw + kj;
                float* row_ptr = col_out + row * out_spatial;

                for (size_t oh = 0; oh < out_h; ++oh) {
                    long in_row = static_cast<long>(oh * stride_h + ki * dilation_h) - static_cast<long>(pad_h0);

                    for (size_t ow = 0; ow < out_w; ++ow) {
                        long in_col = static_cast<long>(ow * stride_w + kj * dilation_w) - static_cast<long>(pad_w0);

                        float value = 0.0f;
                        if (in_row >= 0 && in_row < static_cast<long>(in_h) &&
                            in_col >= 0 && in_col < static_cast<long>(in_w)) {
                            value = input[c * in_h * in_w + static_cast<size_t>(in_row) * in_w + static_cast<size_t>(in_col)];
                        }
                        row_ptr[oh * out_w + ow] = value;
                    }
                }
            }
        }
    }
}

}  // namespace leaf
