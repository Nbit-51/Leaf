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
