#pragma once

#include <cstddef>

namespace leaf {

// Rearranges a (C, H, W) input into a (C*KH*KW, out_h*out_w) column
// matrix, where each column holds one flattened receptive-field patch.
// This turns Conv2D into a single matmul: output = weights @ im2col(input).
//
// Assumes NCHW layout, batch size handled by the caller (one image at a
// time -- called once per batch element).
//
// pads: [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end] (ONNX order,
// matching tools/graph_opt/executor.py's _conv2d convention exactly).
void im2col(
    const float* input, size_t in_c, size_t in_h, size_t in_w,
    size_t kh, size_t kw,
    size_t pad_h0, size_t pad_w0, size_t pad_h1, size_t pad_w1,
    size_t stride_h, size_t stride_w,
    size_t dilation_h, size_t dilation_w,
    float* col_out  // caller-allocated, size (in_c*kh*kw) * (out_h*out_w)
);

// Computes the output spatial dims for the given conv parameters --
// callers need this to size col_out and the final conv output buffer
// before calling im2col.
void conv2d_output_shape(
    size_t in_h, size_t in_w,
    size_t kh, size_t kw,
    size_t pad_h0, size_t pad_w0, size_t pad_h1, size_t pad_w1,
    size_t stride_h, size_t stride_w,
    size_t dilation_h, size_t dilation_w,
    size_t& out_h, size_t& out_w
);

}  // namespace leaf
