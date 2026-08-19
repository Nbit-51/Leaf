#pragma once

#include <cstddef>

namespace leaf {

// Full Conv2D via im2col + GEMM. Single image (no batch dim) -- caller
// loops over batch if needed. NCHW layout throughout.
//
// weights: (out_c, in_c, kh, kw), bias: (out_c) or nullptr for no bias.
// pads: [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end] (ONNX order).
//
// output must be pre-allocated by the caller, sized
// out_c * out_h * out_w (use conv2d_output_shape from im2col.h to
// compute out_h/out_w first).
void conv2d(
    const float* input, size_t in_c, size_t in_h, size_t in_w,
    const float* weights, size_t out_c, size_t kh, size_t kw,
    const float* bias,
    size_t pad_h0, size_t pad_w0, size_t pad_h1, size_t pad_w1,
    size_t stride_h, size_t stride_w,
    size_t dilation_h, size_t dilation_w,
    float* output
);

}  // namespace leaf
