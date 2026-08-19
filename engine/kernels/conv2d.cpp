#include "conv2d.h"
#include "im2col.h"
#include "gemm.h"

#include <vector>
#include <cstring>

namespace leaf {

void conv2d(
    const float* input, size_t in_c, size_t in_h, size_t in_w,
    const float* weights, size_t out_c, size_t kh, size_t kw,
    const float* bias,
    size_t pad_h0, size_t pad_w0, size_t pad_h1, size_t pad_w1,
    size_t stride_h, size_t stride_w,
    size_t dilation_h, size_t dilation_w,
    float* output
) {
    size_t out_h, out_w;
    conv2d_output_shape(
        in_h, in_w, kh, kw, pad_h0, pad_w0, pad_h1, pad_w1,
        stride_h, stride_w, dilation_h, dilation_w, out_h, out_w
    );

    size_t out_spatial = out_h * out_w;
    size_t patch_size = in_c * kh * kw;
    size_t needed = patch_size * out_spatial;

    // Reused across calls instead of a fresh heap allocation per layer per
    // forward pass. Every call fully overwrites [0, needed) before it's
    // read, so growing-only reuse is safe: no stale data ever leaks through.
    static thread_local std::vector<float> col;
    if (col.size() < needed) {
        col.resize(needed);
    }

    im2col(
        input, in_c, in_h, in_w, kh, kw,
        pad_h0, pad_w0, pad_h1, pad_w1,
        stride_h, stride_w, dilation_h, dilation_w,
        col.data()
    );

    std::memset(output, 0, out_c * out_spatial * sizeof(float));
    gemm_f32(weights, col.data(), output, out_c, patch_size, out_spatial, 1.0f, 0.0f);

    if (bias != nullptr) {
        for (size_t oc = 0; oc < out_c; ++oc) {
            float b = bias[oc];
            float* out_row = output + oc * out_spatial;
            for (size_t i = 0; i < out_spatial; ++i) {
                out_row[i] += b;
            }
        }
    }
}

}  // namespace leaf
