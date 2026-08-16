# ============================================================
# Leaf fusion pass scaffold
# Creates: tools/graph_opt/fusion.py
#          tools/graph_opt/executor.py
#          tests/unit/test_fusion.py
# Run from inside your existing "leaf" folder (the one with
# tools/graph_opt/ir.py already in it from the previous script).
# ============================================================

New-Item -ItemType Directory -Force -Path "tools\graph_opt" | Out-Null
New-Item -ItemType Directory -Force -Path "tests\unit" | Out-Null

# ---------------- tools/graph_opt/fusion.py ----------------
@'
"""
Leaf Fusion Pass
================

Two fusions, applied as separate passes over the Graph IR:

1. Conv + BatchNormalization -> Conv (weight folding)
   BatchNorm at inference time is an affine transform per-channel:
       y = scale * (x - mean) / sqrt(var + eps) + bias
   Since Conv is also affine (y = W*x + b), a Conv immediately followed by
   BatchNorm can be collapsed into a single Conv with adjusted weights:
       W' = W * (scale / sqrt(var + eps))
       b' = (b - mean) * (scale / sqrt(var + eps)) + bias
   This removes the BatchNorm node entirely and is mathematically exact
   (not an approximation) -- verified numerically in tests/unit/test_fusion.py.

2. Conv + Activation -> Conv (attribute fusion)
   Not a weight fold -- this just marks the Conv node with an
   `activation` attribute and removes the separate activation node.
   Mirrors how real CPU kernels work: applying ReLU in-place on the
   Conv output avoids materializing an extra intermediate tensor and
   an extra pass over memory. This is exactly what the AVX2 kernel in
   engine/ will implement later (kernels/read the `activation`
   attribute and apply it before writing output).

Both passes only fuse when the intermediate tensor has exactly one
consumer and is not itself a required graph output -- fusing away a
tensor that's needed elsewhere would silently change behavior.
"""

from __future__ import annotations

import numpy as np

from ir import Graph, Node


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_producer_consumer_maps(graph: Graph):
    """producer: tensor_name -> Node that outputs it
       consumers: tensor_name -> list of Nodes that take it as input"""
    producer: dict[str, Node] = {}
    consumers: dict[str, list[Node]] = {}
    for n in graph.nodes:
        for out in n.outputs:
            producer[out] = n
        for inp in n.inputs:
            consumers.setdefault(inp, []).append(n)
    return producer, consumers


def _shallow_copy_graph_shell(graph: Graph) -> Graph:
    """New Graph with the same inputs/outputs/value_info/initializers,
    but an empty node list -- callers fill in new_graph.nodes."""
    new_graph = Graph()
    new_graph.inputs = list(graph.inputs)
    new_graph.outputs = list(graph.outputs)
    new_graph.value_info = dict(graph.value_info)
    new_graph.initializers = dict(graph.initializers)
    return new_graph


# ---------------------------------------------------------------------------
# Pass 1: Conv + BatchNorm folding
# ---------------------------------------------------------------------------

def _fold_conv_bn_weights(conv_w, conv_b, bn_scale, bn_bias, bn_mean, bn_var, eps):
    """Compute the folded Conv weight/bias. conv_w: (OC, IC, KH, KW)."""
    std = np.sqrt(bn_var + eps)
    factor = bn_scale / std                      # shape (OC,)
    folded_w = conv_w * factor.reshape(-1, 1, 1, 1)
    folded_b = (conv_b - bn_mean) * factor + bn_bias
    return folded_w.astype(conv_w.dtype), folded_b.astype(conv_w.dtype)


def fuse_conv_batchnorm(graph: Graph) -> Graph:
    producer, consumers = _build_producer_consumer_maps(graph)
    new_graph = _shallow_copy_graph_shell(graph)

    skip_names: set[str] = set()
    new_nodes: list[Node] = []

    for node in graph.nodes:
        if node.name in skip_names:
            continue

        if node.op_type == "Conv":
            conv_out = node.outputs[0]
            cons = consumers.get(conv_out, [])
            fusable = (
                len(cons) == 1
                and cons[0].op_type == "BatchNormalization"
                and conv_out not in new_graph.outputs
            )
            if fusable:
                bn_node = cons[0]

                conv_w = new_graph.initializers[node.inputs[1]]
                oc = conv_w.shape[0]
                conv_b = (
                    new_graph.initializers[node.inputs[2]]
                    if len(node.inputs) > 2
                    else np.zeros(oc, dtype=conv_w.dtype)
                )

                bn_scale = new_graph.initializers[bn_node.inputs[1]]
                bn_bias = new_graph.initializers[bn_node.inputs[2]]
                bn_mean = new_graph.initializers[bn_node.inputs[3]]
                bn_var = new_graph.initializers[bn_node.inputs[4]]
                eps = float(bn_node.attributes.get("epsilon", 1e-5))

                folded_w, folded_b = _fold_conv_bn_weights(
                    conv_w, conv_b, bn_scale, bn_bias, bn_mean, bn_var, eps
                )

                w_name = node.inputs[1] + "_bnfused"
                b_name = (node.inputs[1] + "_bnfused_bias")
                new_graph.initializers[w_name] = folded_w
                new_graph.initializers[b_name] = folded_b

                fused_node = Node(
                    name=node.name + "_bnfused",
                    op_type="Conv",
                    inputs=[node.inputs[0], w_name, b_name],
                    outputs=[bn_node.outputs[0]],
                    attributes=dict(node.attributes),
                )
                new_nodes.append(fused_node)
                skip_names.add(bn_node.name)
                continue

        new_nodes.append(node)

    new_graph.nodes = new_nodes
    return new_graph


# ---------------------------------------------------------------------------
# Pass 2: Conv + Activation fusion
# ---------------------------------------------------------------------------

_SUPPORTED_ACTIVATIONS = ("Relu",)  # extend as the kernel side gains support


def fuse_conv_activation(graph: Graph) -> Graph:
    producer, consumers = _build_producer_consumer_maps(graph)
    new_graph = _shallow_copy_graph_shell(graph)

    skip_names: set[str] = set()
    new_nodes: list[Node] = []

    for node in graph.nodes:
        if node.name in skip_names:
            continue

        if node.op_type == "Conv":
            conv_out = node.outputs[0]
            cons = consumers.get(conv_out, [])
            fusable = (
                len(cons) == 1
                and cons[0].op_type in _SUPPORTED_ACTIVATIONS
                and conv_out not in new_graph.outputs
            )
            if fusable:
                act_node = cons[0]
                fused_attrs = dict(node.attributes)
                fused_attrs["activation"] = act_node.op_type

                fused_node = Node(
                    name=node.name + "_actfused",
                    op_type="Conv",
                    inputs=list(node.inputs),
                    outputs=[act_node.outputs[0]],
                    attributes=fused_attrs,
                )
                new_nodes.append(fused_node)
                skip_names.add(act_node.name)
                continue

        new_nodes.append(node)

    new_graph.nodes = new_nodes
    return new_graph


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_fusion_passes(graph: Graph) -> Graph:
    """Apply all fusion passes in order. BN folding must run before
    activation fusion, since activation fusion looks for a Conv node
    directly feeding an activation -- BN folding is what makes that
    true when the original graph was Conv->BN->Relu."""
    g = fuse_conv_batchnorm(graph)
    g = fuse_conv_activation(g)
    return g
'@ | Set-Content -Path "tools\graph_opt\fusion.py" -Encoding UTF8

# ---------------- tools/graph_opt/executor.py ----------------
@'
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
'@ | Set-Content -Path "tools\graph_opt\executor.py" -Encoding UTF8

# ---------------- tests/unit/test_fusion.py ----------------
@'
"""
Fusion correctness test.

Builds the same Conv->BatchNorm->Relu synthetic graph used in
test_ir_loader.py, runs it through the reference executor BEFORE and
AFTER fusion on identical random input, and asserts the outputs match
to floating-point tolerance. This is the actual correctness proof for
the fusion pass -- structural checks (node count, op types) are useful
sanity checks but do not prove the rewrite preserves behavior.
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "graph_opt"))
from ir import Graph
from fusion import run_fusion_passes, fuse_conv_batchnorm, fuse_conv_activation
from executor import run_graph

sys.path.insert(0, os.path.dirname(__file__))
from test_ir_loader import build_conv_bn_relu_model


def _random_input():
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((1, 3, 8, 8)).astype(np.float32)


def test_fused_graph_matches_reference_numerically():
    model = build_conv_bn_relu_model()
    graph = Graph.from_onnx(model)

    x = _random_input()

    baseline = run_graph(graph, {"x": x})
    y_baseline = baseline["y"]

    fused_graph = run_fusion_passes(graph)
    fused = run_graph(fused_graph, {"x": x})
    y_fused = fused["y"]

    np.testing.assert_allclose(y_baseline, y_fused, rtol=1e-4, atol=1e-5)
    print(f"PASS: numerical equivalence (max abs diff = {np.max(np.abs(y_baseline - y_fused)):.2e})")


def test_fusion_reduces_node_count():
    model = build_conv_bn_relu_model()
    graph = Graph.from_onnx(model)

    assert len(graph.nodes) == 3
    fused_graph = run_fusion_passes(graph)
    assert len(fused_graph.nodes) == 1, f"expected 1 fused node, got {len(fused_graph.nodes)}"

    fused_node = fused_graph.nodes[0]
    assert fused_node.op_type == "Conv"
    assert fused_node.attributes.get("activation") == "Relu"

    print(f"PASS: 3 nodes -> 1 fused node ({fused_node})")


def test_bn_fold_only():
    """BN folding alone (no activation fusion) should leave 2 nodes:
    a fused Conv and the original Relu, unchanged."""
    model = build_conv_bn_relu_model()
    graph = Graph.from_onnx(model)

    bn_fused = fuse_conv_batchnorm(graph)
    assert len(bn_fused.nodes) == 2, f"expected 2 nodes after BN-only fold, got {len(bn_fused.nodes)}"
    op_types = [n.op_type for n in bn_fused.nodes]
    assert op_types == ["Conv", "Relu"], op_types

    print(f"PASS: BN-only fold -> {op_types}")


def test_no_fusion_when_output_has_multiple_consumers():
    """If the Conv output feeds two different nodes, fusing it away
    would silently drop a required value -- the pass must refuse."""
    from ir import Node

    model = build_conv_bn_relu_model()
    graph = Graph.from_onnx(model)

    # Add a second, unrelated consumer of the Conv's output tensor.
    branch_node = Node(
        name="extra_consumer",
        op_type="Relu",  # arbitrary op, just needs to consume conv_out
        inputs=["conv_out"],
        outputs=["extra_out"],
    )
    graph.nodes.append(branch_node)
    graph.outputs.append("extra_out")  # keep it live so it cannot be pruned away

    fused_graph = fuse_conv_batchnorm(graph)
    op_types = [n.op_type for n in fused_graph.nodes]
    assert "BatchNormalization" in op_types, (
        f"BN should NOT have been fused away (conv_out has 2 consumers), got {op_types}"
    )

    print(f"PASS: fusion correctly skipped (multi-consumer tensor) -> {op_types}")


if __name__ == "__main__":
    test_fused_graph_matches_reference_numerically()
    print()
    test_fusion_reduces_node_count()
    print()
    test_bn_fold_only()
    print()
    test_no_fusion_when_output_has_multiple_consumers()
    print("\nAll fusion tests passed.")
'@ | Set-Content -Path "tests\unit\test_fusion.py" -Encoding UTF8

Write-Host "Fusion pass files created." -ForegroundColor Green
Write-Host "Next: python tests\unit\test_fusion.py" -ForegroundColor Cyan
