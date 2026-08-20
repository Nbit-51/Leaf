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

Supports the ops needed for Conv/BatchNorm/Relu fusion tests, plus the
extra ops (Add, MaxPool, GlobalAveragePool, Flatten, Gemm) needed for
full end-to-end numerical verification on real classifier architectures
like ResNet-18 (residual adds, pooling, and the final FC head).
Extend as new op types are added to the passes under test.
"""

from __future__ import annotations

import numpy as np
from onnx import numpy_helper

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


def _maxpool2d(x, kernel_shape, pads, strides):
    """Direct MaxPool2D reference implementation.
    x: (N, C, H, W)
    pads: [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end] (ONNX order)
    """
    n_batch, c, in_h, in_w = x.shape
    kh, kw = kernel_shape
    pad_h0, pad_w0, pad_h1, pad_w1 = pads
    sh, sw = strides

    x_padded = np.pad(
        x, ((0, 0), (0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1)),
        mode="constant", constant_values=-np.inf,
    )

    padded_h = x_padded.shape[2]
    padded_w = x_padded.shape[3]
    out_h = (padded_h - kh) // sh + 1
    out_w = (padded_w - kw) // sw + 1

    out = np.full((n_batch, c, out_h, out_w), -np.inf, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            patch = x_padded[
                :, :,
                i: i + out_h * sh: sh,
                j: j + out_w * sw: sw,
            ]
            out = np.maximum(out, patch)

    return out


def _global_average_pool(x):
    """x: (N, C, H, W) -> (N, C, 1, 1), mean over spatial dims."""
    return x.mean(axis=(2, 3), keepdims=True)


def _flatten(x, axis):
    """Flatten all dims from `axis` onward into one, keeping dims before
    `axis` as-is -- matches ONNX Flatten semantics."""
    shape = x.shape
    outer = int(np.prod(shape[:axis])) if axis > 0 else 1
    inner = int(np.prod(shape[axis:])) if axis < len(shape) else 1
    return x.reshape(outer, inner)


def _gemm(a, b, c, alpha, beta, trans_a, trans_b):
    """General matrix multiply: out = alpha * (A' @ B') + beta * C,
    matching ONNX Gemm semantics (transA/transB optionally transpose
    A/B before multiplying)."""
    if trans_a:
        a = a.T
    if trans_b:
        b = b.T
    out = alpha * (a @ b)
    if c is not None:
        out = out + beta * c
    return out


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

        elif node.op_type == "Add":
            a = tensors[node.inputs[0]]
            b = tensors[node.inputs[1]]
            tensors[node.outputs[0]] = a + b

        elif node.op_type == "MaxPool":
            x = tensors[node.inputs[0]]
            kernel_shape = list(node.attributes.get("kernel_shape"))
            pads = list(node.attributes.get("pads", [0, 0, 0, 0]))
            strides = list(node.attributes.get("strides", [1, 1]))
            tensors[node.outputs[0]] = _maxpool2d(x, kernel_shape, pads, strides)

        elif node.op_type == "GlobalAveragePool":
            x = tensors[node.inputs[0]]
            tensors[node.outputs[0]] = _global_average_pool(x)

        elif node.op_type == "Flatten":
            x = tensors[node.inputs[0]]
            axis = int(node.attributes.get("axis", 1))
            tensors[node.outputs[0]] = _flatten(x, axis)

        elif node.op_type == "Gemm":
            a = tensors[node.inputs[0]]
            b = tensors[node.inputs[1]]
            c = tensors[node.inputs[2]] if len(node.inputs) > 2 else None
            alpha = float(node.attributes.get("alpha", 1.0))
            beta = float(node.attributes.get("beta", 1.0))
            trans_a = bool(node.attributes.get("transA", 0))
            trans_b = bool(node.attributes.get("transB", 0))
            tensors[node.outputs[0]] = _gemm(a, b, c, alpha, beta, trans_a, trans_b)

        elif node.op_type == "Constant":
            value = node.attributes.get("value")
            tensors[node.outputs[0]] = numpy_helper.to_array(value)

        elif node.op_type == "Cast":
            # Leaf's runtime is fp32 throughout -- exporter Cast round-trips
            # are treated as identity (matches fusion.py's RMSNorm docstring
            # rationale for absorbing these Casts rather than preserving them).
            tensors[node.outputs[0]] = tensors[node.inputs[0]].astype(np.float32)

        elif node.op_type == "Pow":
            base = tensors[node.inputs[0]]
            exponent = tensors[node.inputs[1]]
            tensors[node.outputs[0]] = np.power(base, exponent)

        elif node.op_type == "ReduceMean":
            x = tensors[node.inputs[0]]
            axes = tensors[node.inputs[1]] if len(node.inputs) > 1 else node.attributes.get("axes")
            keepdims = bool(node.attributes.get("keepdims", 1))
            axes_tuple = tuple(int(a) for a in np.asarray(axes).reshape(-1))
            tensors[node.outputs[0]] = np.mean(x, axis=axes_tuple, keepdims=keepdims)

        elif node.op_type == "Sqrt":
            tensors[node.outputs[0]] = np.sqrt(tensors[node.inputs[0]])

        elif node.op_type == "Div":
            a = tensors[node.inputs[0]]
            b = tensors[node.inputs[1]]
            tensors[node.outputs[0]] = a / b

        elif node.op_type == "Mul":
            a = tensors[node.inputs[0]]
            b = tensors[node.inputs[1]]
            tensors[node.outputs[0]] = a * b

        elif node.op_type == "RMSNorm":
            x = tensors[node.inputs[0]]
            weight = tensors[node.inputs[1]]
            eps = float(node.attributes.get("eps", 1e-6))
            variance = np.mean(np.power(x.astype(np.float32), 2), axis=-1, keepdims=True)
            normed = x.astype(np.float32) / np.sqrt(variance + eps)
            tensors[node.outputs[0]] = (weight * normed).astype(np.float32)

        else:
            raise NotImplementedError(f"reference executor: unsupported op '{node.op_type}'")

    return tensors
