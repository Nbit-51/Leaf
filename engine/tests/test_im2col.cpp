// Standalone correctness check for im2col -- hand-verifiable small case
// before it's trusted to feed a real conv.

#include "../kernels/im2col.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

bool close(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) < tol;
}

int main() {
    // 1 channel, 3x3 input, 2x2 kernel, stride 1, no padding, no dilation.
    // Input:
    //   1 2 3
    //   4 5 6
    //   7 8 9
    // Expected output shape: 2x2 (out_h=2, out_w=2), so 4 patches, each
    // of size 1*2*2=4 -> col matrix is 4 rows x 4 cols.
    // Patch order (row-major over output positions):
    //   (0,0): [1,2,4,5]  (0,1): [2,3,5,6]  (1,0): [4,5,7,8]  (1,1): [5,6,8,9]
    // im2col lays these out as columns, with rows = (c,ki,kj) combos:
    //   row0 (ki=0,kj=0): [1, 2, 4, 5]
    //   row1 (ki=0,kj=1): [2, 3, 5, 6]
    //   row2 (ki=1,kj=0): [4, 5, 7, 8]
    //   row3 (ki=1,kj=1): [5, 6, 8, 9]
    {
        std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};

        size_t out_h, out_w;
        leaf::conv2d_output_shape(3, 3, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, out_h, out_w);
        assert(out_h == 2 && out_w == 2);

        std::vector<float> col(1 * 2 * 2 * out_h * out_w, -999.0f);
        leaf::im2col(input.data(), 1, 3, 3, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, col.data());

        std::vector<float> expected = {
            1, 2, 4, 5,   // row0
            2, 3, 5, 6,   // row1
            4, 5, 7, 8,   // row2
            5, 6, 8, 9,   // row3
        };

        for (size_t i = 0; i < expected.size(); ++i) {
            assert(close(col[i], expected[i]));
        }
        std::cout << "PASS: small hand-checked im2col (no padding)\n";
    }

    // Same input, but with padding=1 on all sides, to check zero-padding
    // and out-of-bounds handling.
    {
        std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};

        size_t out_h, out_w;
        leaf::conv2d_output_shape(3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, out_h, out_w);
        assert(out_h == 4 && out_w == 4);  // padded to 5x5, kernel 2 -> 4x4 out

        std::vector<float> col(1 * 2 * 2 * out_h * out_w, -999.0f);
        leaf::im2col(input.data(), 1, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, col.data());

        // Spot-check the very first patch (top-left, mostly padding):
        // at output (0,0), the 2x2 window covers padded positions
        // (-1,-1),(-1,0),(0,-1),(0,0) -> only (0,0)=1 is real, rest are 0.
        size_t out_spatial = out_h * out_w;
        assert(close(col[0 * out_spatial + 0], 0.0f));  // (ki=0,kj=0) at out(0,0) -> padded row -1
        assert(close(col[3 * out_spatial + 0], 1.0f));  // (ki=1,kj=1) at out(0,0) -> real input[0,0]=1
        std::cout << "PASS: im2col zero-padding boundary handling\n";
    }

    std::cout << "\nAll im2col tests passed.\n";
    return 0;
}
