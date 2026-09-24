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
from onnx import TensorProto, numpy_helper

try:
    from .ir import Graph
except ImportError:
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
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _silu(x):
    return x * _sigmoid(x)


def _reshape_scales(scales: np.ndarray, axis: int, ndim: int) -> np.ndarray:
    shape = [1] * ndim
    shape[axis] = scales.size
    return scales.reshape(shape)


def _fake_quantize(value: np.ndarray, params: dict | None) -> np.ndarray:
    if not params:
        return value
    scale = np.asarray(params["scale"], dtype=np.float32)
    axis = params.get("axis")
    if axis is not None and scale.ndim:
        scale = _reshape_scales(scale, int(axis), value.ndim)
    scale = np.maximum(scale, np.finfo(np.float32).tiny)
    quantized = np.clip(np.rint(value / scale), -127, 127).astype(np.int8)
    return quantized.astype(np.float32) * scale


def _dequantize_weight(value: np.ndarray, params: dict | None) -> np.ndarray:
    if not params:
        return value
    scales = np.asarray(params["scale"], dtype=np.float32)
    return value.astype(np.float32) * _reshape_scales(scales, int(params["axis"]), value.ndim)

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


_ACTIVATIONS = {"Relu": _relu, "Gelu": _gelu, "Silu": _silu}


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

            quantization = node.attributes.get("quantization")
            if quantization:
                x = _fake_quantize(x, quantization.get("input"))
                w = _dequantize_weight(w, quantization.get("weight"))

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
        elif node.op_type == "Sigmoid":
            tensors[node.outputs[0]] = _sigmoid(tensors[node.inputs[0]])
        elif node.op_type in ("Gelu", "Silu"):
            tensors[node.outputs[0]] = _ACTIVATIONS[node.op_type](tensors[node.inputs[0]])
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
            quantization = node.attributes.get("quantization")
            if quantization:
                a = _fake_quantize(a, quantization.get("input"))
                b = _dequantize_weight(b, quantization.get("weight"))
            c = tensors[node.inputs[2]] if len(node.inputs) > 2 else None
            alpha = float(node.attributes.get("alpha", 1.0))
            beta = float(node.attributes.get("beta", 1.0))
            trans_a = bool(node.attributes.get("transA", 0))
            trans_b = bool(node.attributes.get("transB", 0))
            output = _gemm(a, b, c, alpha, beta, trans_a, trans_b)
            activation = node.attributes.get("activation")
            tensors[node.outputs[0]] = _ACTIVATIONS[activation](output) if activation else output

        elif node.op_type == "MatMul":
            a = tensors[node.inputs[0]]
            b = tensors[node.inputs[1]]
            quantization = node.attributes.get("quantization")
            if quantization:
                a = _fake_quantize(a, quantization.get("input"))
                b = _dequantize_weight(b, quantization.get("weight"))
            tensors[node.outputs[0]] = np.matmul(a, b)

        elif node.op_type == "Constant":
            value = node.attributes.get("value")
            tensors[node.outputs[0]] = (
                numpy_helper.to_array(value)
                if hasattr(value, "data_type")
                else np.asarray(value)
            )

        elif node.op_type == "Cast":
            dtype = {
                TensorProto.FLOAT: np.float32, TensorProto.DOUBLE: np.float64,
                TensorProto.INT64: np.int64, TensorProto.INT32: np.int32,
                TensorProto.INT8: np.int8,
            }[int(node.attributes.get("to", TensorProto.FLOAT))]
            tensors[node.outputs[0]] = tensors[node.inputs[0]].astype(dtype)

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

        elif node.op_type == "Sub":
            tensors[node.outputs[0]] = tensors[node.inputs[0]] - tensors[node.inputs[1]]

        elif node.op_type == "Identity":
            tensors[node.outputs[0]] = tensors[node.inputs[0]]

        elif node.op_type == "Reshape":
            shape = tuple(int(value) for value in tensors[node.inputs[1]].reshape(-1))
            tensors[node.outputs[0]] = np.reshape(tensors[node.inputs[0]], shape)

        elif node.op_type == "Transpose":
            tensors[node.outputs[0]] = np.transpose(
                tensors[node.inputs[0]], axes=node.attributes.get("perm")
            )

        elif node.op_type == "Concat":
            tensors[node.outputs[0]] = np.concatenate(
                [tensors[name] for name in node.inputs], axis=int(node.attributes.get("axis", 0))
            )

        elif node.op_type in ("Squeeze", "Unsqueeze"):
            axes = node.attributes.get("axes")
            if axes is None and len(node.inputs) > 1:
                axes = tensors[node.inputs[1]].reshape(-1).tolist()
            value = tensors[node.inputs[0]]
            if node.op_type == "Squeeze":
                tensors[node.outputs[0]] = np.squeeze(value, axis=None if axes is None else tuple(axes))
            else:
                for axis in sorted(int(item) for item in axes):
                    value = np.expand_dims(value, axis)
                tensors[node.outputs[0]] = value

        elif node.op_type == "Gather":
            tensors[node.outputs[0]] = np.take(
                tensors[node.inputs[0]], tensors[node.inputs[1]],
                axis=int(node.attributes.get("axis", 0)),
            )

        elif node.op_type == "Shape":
            shape = tensors[node.inputs[0]].shape
            start = int(node.attributes.get("start", 0))
            end = int(node.attributes.get("end", len(shape)))
            tensors[node.outputs[0]] = np.asarray(shape[start:end], dtype=np.int64)

        elif node.op_type == "LayerNormalization":
            value = tensors[node.inputs[0]]
            axis = int(node.attributes.get("axis", -1)) % value.ndim
            epsilon = float(node.attributes.get("epsilon", 1e-5))
            axes = tuple(range(axis, value.ndim))
            mean = value.mean(axis=axes, keepdims=True)
            result = (value - mean) / np.sqrt(value.var(axis=axes, keepdims=True) + epsilon)
            result *= tensors[node.inputs[1]]
            if len(node.inputs) > 2:
                result += tensors[node.inputs[2]]
            tensors[node.outputs[0]] = result

        elif node.op_type == "RMSNorm":
            x = tensors[node.inputs[0]]
            weight = tensors[node.inputs[1]]
            eps = float(node.attributes.get("eps", 1e-6))
            variance = np.mean(np.power(x.astype(np.float32), 2), axis=-1, keepdims=True)
            normed = x.astype(np.float32) / np.sqrt(variance + eps)
            tensors[node.outputs[0]] = (weight * normed).astype(np.float32)

        elif node.op_type == "RoPE_Table":
            frequencies = tensors[node.inputs[0]]
            positions = tensors[node.inputs[1]]
            if (frequencies.ndim != 3 or positions.ndim != 3 or
                    frequencies.shape[2] != 1 or positions.shape[1] != 1 or
                    frequencies.shape[0] != positions.shape[0]):
                raise ValueError("RoPE_Table expects [batch, half_dim, 1] and [batch, 1, tokens]")
            angles = np.matmul(frequencies, positions).transpose(0, 2, 1)
            duplicated = np.concatenate((angles, angles), axis=-1)
            tensors[node.outputs[0]] = (
                np.cos(duplicated) * float(node.attributes.get("cos_scale", 1.0))
            ).astype(np.float32)
            tensors[node.outputs[1]] = (
                np.sin(duplicated) * float(node.attributes.get("sin_scale", 1.0))
            ).astype(np.float32)

        elif node.op_type == "RepeatKV":
            repeats = int(node.attributes["n_rep"])
            if repeats < 1 or tensors[node.inputs[0]].ndim != 4:
                raise ValueError("RepeatKV requires a positive repeat count and rank-4 input")
            tensors[node.outputs[0]] = np.repeat(tensors[node.inputs[0]], repeats,
                                                  axis=1).astype(np.float32)

        elif node.op_type == "SwiGLU_MLP":
            x = tensors[node.inputs[0]]
            gate_weight = tensors[node.inputs[1]]
            up_weight = tensors[node.inputs[2]]
            down_weight = tensors[node.inputs[3]]
            residual = tensors[node.inputs[4]]
            gate = np.matmul(x, gate_weight)
            up = np.matmul(x, up_weight)
            tensors[node.outputs[0]] = (
                np.matmul(_silu(gate) * up, down_weight) + residual
            ).astype(np.float32)

        elif node.op_type == "Attention":
            query = tensors[node.inputs[0]]
            key = tensors[node.inputs[1]]
            value = tensors[node.inputs[2]]
            mask = tensors[node.inputs[3]].astype(bool)
            if not bool(node.attributes.get("mask_nonzero_is_valid", 1)):
                mask = ~mask
            scale = float(node.attributes.get("scale", 1.0))
            scores = np.matmul(query, np.swapaxes(key, -1, -2)) * (scale * scale)
            scores = np.where(mask, scores, -np.inf)
            maximum = np.max(scores, axis=-1, keepdims=True)
            safe_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
            probabilities = np.exp(scores - safe_maximum)
            probabilities /= np.where(np.sum(probabilities, axis=-1, keepdims=True) > 0,
                                      np.sum(probabilities, axis=-1, keepdims=True), 1.0)
            tensors[node.outputs[0]] = np.transpose(
                np.matmul(probabilities, value), (0, 2, 1, 3)
            ).astype(np.float32)

        else:
            raise NotImplementedError(f"reference executor: unsupported op '{node.op_type}'")

    return tensors
