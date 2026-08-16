"""
Leaf Reference Executor (test-only)
====================================

A minimal, slow, numpy-only interpreter for the Graph IR. This is NOT the
production runtime -- that is the C++ engine in engine/, built around the
AVX2 kernels. This executor exists purely so fusion (and later,
quantization/pruning) passes can be verified numerically: run the graph
before and after a rewrite, on the same input, and assert the outputs
match. A structural check alone ("the BN node is gone") does not prove
the rewrite preserves behavior -- running both graphs does.

Supports exactly the ops needed for the Conv/BatchNorm/Relu fusion tests.
Extend as new op types are added to the passes under test.
"""

from __future__ import annotations

import numpy as np

from ir import Graph


def _conv2d(x, w, b, pads, strides, dilations):
    """Direct (non-im2col) Conv2D reference implementation.
    x: (N, C, H, W), w: (OC, IC, KH, KW), b: (OC,)
    pads: [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end] (ONNX order)
    """
    n_batch, in_c, in_h, in_w = x.shape
    out_c, _, kh, kw = w.shape

    pad_h0, pad_w0, pad_h1, pad_w1 = pads
    sh, sw = strides
    dh, dw = dilations

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1)))

    padded_h = x_padded.shape[2]
    padded_w = x_padded.shape[3]
    out_h = (padded_h - (dh * (kh - 1) + 1)) // sh + 1
    out_w = (padded_w - (dw * (kw - 1) + 1)) // sw + 1

    out = np.zeros((n_batch, out_c, out_h, out_w), dtype=np.float32)

    for oc in range(out_c):
        acc = np.full((n_batch, out_h, out_w), b[oc], dtype=np.float32)
        for ic in range(in_c):
            for i in range(kh):
                for j in range(kw):
                    row_start = i * dh
                    col_start = j * dw
                    patch = x_padded[
                        :, ic,
                        row_start: row_start + out_h * sh: sh,
                        col_start: col_start + out_w * sw: sw,
                    ]
                    acc += patch * w[oc, ic, i, j]
        out[:, oc, :, :] = acc

    return out


def _batchnorm(x, scale, bias, mean, var, eps):
    c = x.shape[1]
    scale = scale.reshape(1, c, 1, 1)
    bias = bias.reshape(1, c, 1, 1)
    mean = mean.reshape(1, c, 1, 1)
    var = var.reshape(1, c, 1, 1)
    return scale * (x - mean) / np.sqrt(var + eps) + bias


def _relu(x):
    return np.maximum(x, 0)


_ACTIVATIONS = {"Relu": _relu}


def run_graph(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Execute the graph on concrete numpy inputs, returning every named
    tensor produced (including intermediates), keyed by tensor name."""
    tensors: dict[str, np.ndarray] = dict(graph.initializers)
    tensors.update(inputs)

    for node in graph.topological_order():
        if node.op_type == "Conv":
            x = tensors[node.inputs[0]]
            w = tensors[node.inputs[1]]
            b = tensors[node.inputs[2]] if len(node.inputs) > 2 else np.zeros(w.shape[0], dtype=w.dtype)

            pads = list(node.attributes.get("pads", [0, 0, 0, 0]))
            strides = list(node.attributes.get("strides", [1, 1]))
            dilations = list(node.attributes.get("dilations", [1, 1]))

            out = _conv2d(x, w, b, pads, strides, dilations)

            activation = node.attributes.get("activation")
            if activation is not None:
                out = _ACTIVATIONS[activation](out)

            tensors[node.outputs[0]] = out

        elif node.op_type == "BatchNormalization":
            x = tensors[node.inputs[0]]
            scale = tensors[node.inputs[1]]
            bias = tensors[node.inputs[2]]
            mean = tensors[node.inputs[3]]
            var = tensors[node.inputs[4]]
            eps = float(node.attributes.get("epsilon", 1e-5))
            tensors[node.outputs[0]] = _batchnorm(x, scale, bias, mean, var, eps)

        elif node.op_type == "Relu":
            tensors[node.outputs[0]] = _relu(tensors[node.inputs[0]])

        else:
            raise NotImplementedError(f"reference executor: unsupported op '{node.op_type}'")

    return tensors
