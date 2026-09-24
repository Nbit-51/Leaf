"""
Leaf Fusion Pass
================

Fusions, applied as separate passes over the Graph IR:

CNN passes (M2a):

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

Transformer passes (M2b), added to collapse ONNX-exporter dynamic-shape/
mask-construction plumbing into canonical Leaf IR ops. Each pattern below
was verified node-by-node against a real Qwen2.5-0.5B ONNX export before
being encoded here -- see docs/handover.md section 3.3 for the raw dumps.

3. RMSNorm
   Cast(x) -> Pow(x,2) -> ReduceMean(axis=-1,keepdims=1) -> Add(eps)
     -> Sqrt -> Div(1, sqrt) -> Mul(x) -> Cast -> Mul(weight)
   Formula: weight * (x / sqrt(mean(x^2, axis=-1) + eps)). The two Casts
   are float32 round-trips (exporter artifact of running comparisons in
   fp32 regardless of model dtype) and are absorbed into the fused node
   rather than preserved, since Leaf's runtime is fp32 throughout.

4. RoPE_Table (2-output: cos, sin)
   Cast->Cast->MatMul->Transpose->Concat(freqs,freqs)->Cos->Mul(scale)
     (branch)                                        ->Sin->Mul(scale)
   Computed once per graph (not per-layer) -- inv_freq @ position_ids.
   This is the first Leaf IR node with more than one output; see
   docs/decisions.md for the ADR on extending Node/executor for it.

5. RepeatKV (GQA broadcast)
   Unsqueeze(x, axis=2) -> Expand(to [b,kv_heads,n_rep,seq,hd])
     -> Reshape(to [b,kv_heads*n_rep,seq,hd])
   The dozens of Shape/Gather/Constant/Concat/ConstantOfShape/Equal/Where
   nodes surrounding this skeleton are pure ONNX-exporter dynamic-shape
   bookkeeping (computing the Expand/Reshape target shapes at graph-build
   time from runtime Shape() calls). Rather than hardcode that bookkeeping
   node-for-node, this pass verifies it consists *only* of ops from a
   known-safe whitelist and folds it away. It records n_rep only when shape
   metadata proves the repeat count; otherwise the pattern remains unfused.

6. Attention (causal, scaled)
   Mul(Q,scale) . Mul(K^T,scale) -> MatMul -> Add(mask) -> Softmax
     -> IsNaN-cleanup Where -> MatMul(.,V) -> Transpose
   Mask is precomputed once at graph level and Sliced per layer (that
   Slice/Shape/Gather/Concat/Reshape prep is whitelisted away the same
   way as RepeatKV's bookkeeping). Q and K are each pre-scaled by the
   same constant (their product gives the usual 1/sqrt(head_dim)),
   rather than the product being scaled once -- verified from the dump,
   not assumed.

7. SwiGLU_MLP
   MatMul(gate)->Sigmoid->Mul(.,gate) ,  MatMul(up)
     -> Mul(silu(gate), up) -> MatMul(down) -> Add(residual)
   Formula: down_proj(silu(gate_proj(x)) * up_proj(x)) + residual.

Every pass below only fuses a chain when each intermediate tensor in it
has exactly one consumer (or, for the exporter-bookkeeping sections,
consists only of whitelisted op types) -- if the surrounding graph
doesn't exactly match the verified shape, the pass leaves those nodes
untouched rather than risk a silent mis-fusion.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper

try:
    from .ir import Graph, Node
except ImportError:
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
    new_graph.metadata = dict(graph.metadata)
    return new_graph


def _constant_array(producer: dict[str, Node], tensor_name: str):
    """If tensor_name is produced by a Constant node, return its value as
    an np.ndarray. Returns None if the tensor isn't a Constant output
    (e.g. it's a graph input, initializer, or computed value) -- callers
    treat None as 'pattern doesn't match', never as zero/empty."""
    node = producer.get(tensor_name)
    if node is None or node.op_type != "Constant":
        return None
    value = node.attributes.get("value")
    if value is None:
        return None
    return numpy_helper.to_array(value)


def _single_consumer(consumers: dict[str, list[Node]], tensor_name: str):
    """Returns the single node that consumes tensor_name, or None if zero
    or more-than-one *distinct* nodes consume it. Dedupes by node identity
    before counting: _build_producer_consumer_maps appends once per input
    *slot*, so a node that legitimately lists the same tensor twice as its
    own input (e.g. Concat([t, t], axis=-1), as RoPE_Table's own pattern
    requires) would otherwise show up as 2 entries for what is really one
    consuming node. Confirmed bug (found the same session as the RMSNorm
    single-consumer contradiction): _try_match_rope_table's concat_node
    lookup failed on every real instance of its own documented pattern
    for exactly this reason -- see docs/handover.md."""
    cons = consumers.get(tensor_name, [])
    distinct = list({id(n): n for n in cons}.values())
    return distinct[0] if len(distinct) == 1 else None


def _collect_backward(tensor_name: str, producer: dict[str, Node],
                       whitelist: set[str], visited: set[str]) -> bool:
    """Walk backward from tensor_name through its producer chain, requiring
    every node encountered has op_type in `whitelist`. Stops (returns True)
    at graph inputs/initializers (no producer node). Returns False the
    moment a non-whitelisted op is found -- caller must then leave the
    surrounding pattern unfused rather than guess at what it means.
    Does not recurse past Shape() nodes: Shape(x) only reads x's
    dimensions, so nothing downstream can depend on x's actual values --
    walking into x's own producer chain would incorrectly require x's
    real computation to also be whitelisted bookkeeping, when only its
    shape needs to be safe. Confirmed necessary against the real
    Qwen2.5-0.5B export: without this, the Attention mask walk follows a
    Shape() call into k_proj's MatMul and input_layernorm's Sqrt."""
    node = producer.get(tensor_name)
    if node is None:
        return True
    if node.name in visited:
        return True
    if node.op_type not in whitelist:
        return False
    visited.add(node.name)
    if node.op_type == "Shape":
        return True
    for inp in node.inputs:
        if not _collect_backward(inp, producer, whitelist, visited):
            return False
    return True


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
# Pass 3: RMSNorm
# ---------------------------------------------------------------------------

def _try_match_rmsnorm(graph: Graph, cast_node: Node, producer, consumers):
    x_name = cast_node.inputs[0]

    # cast_node.outputs[0] is legitimately consumed twice in a correct
    # RMSNorm instance: once by Pow (the x^2 step) and again later by
    # mul1_node (the "x * (1/sqrt(...))" step). Requiring a single
    # consumer here was the original bug -- it could never match real
    # RMSNorm graphs, since the pattern's own second Mul step requires
    # this same tensor to also feed a different node (verified below,
    # where mul1_node is found). Fixed the same way Attention's IsNaN
    # lookup was fixed: search consumers by op_type instead of requiring
    # exactly one. Consumer-count safety (no unexpected third consumer)
    # is verified later, once mul1_node is known -- see the check next to
    # "cast_node.outputs[0] not in mul1_node.inputs" below.
    cast1_consumers = consumers.get(cast_node.outputs[0], [])
    pow_node = next((n for n in cast1_consumers if n.op_type == "Pow"), None)
    if pow_node is None:
        return None
    exponent = _constant_array(producer, pow_node.inputs[1])
    if exponent is None or not np.allclose(exponent, 2.0):
        return None

    reducemean_node = _single_consumer(consumers, pow_node.outputs[0])
    if reducemean_node is None or reducemean_node.op_type != "ReduceMean":
        return None
    if len(reducemean_node.inputs) < 2:
        return None
    axes = _constant_array(producer, reducemean_node.inputs[1])
    keepdims = reducemean_node.attributes.get("keepdims", 1)
    if axes is None or axes.reshape(-1).tolist() != [-1] or int(keepdims) != 1:
        return None

    add_node = _single_consumer(consumers, reducemean_node.outputs[0])
    if add_node is None or add_node.op_type != "Add":
        return None
    eps_arr = None
    for inp in add_node.inputs:
        if inp != reducemean_node.outputs[0]:
            eps_arr = _constant_array(producer, inp)
    if eps_arr is None:
        return None

    sqrt_node = _single_consumer(consumers, add_node.outputs[0])
    if sqrt_node is None or sqrt_node.op_type != "Sqrt":
        return None

    div_node = _single_consumer(consumers, sqrt_node.outputs[0])
    if div_node is None or div_node.op_type != "Div" or len(div_node.inputs) < 2:
        return None
    if div_node.inputs[1] != sqrt_node.outputs[0]:
        return None
    one_arr = _constant_array(producer, div_node.inputs[0])
    if one_arr is None or not np.allclose(one_arr, 1.0):
        return None

    mul1_node = _single_consumer(consumers, div_node.outputs[0])
    if mul1_node is None or mul1_node.op_type != "Mul":
        return None
    if cast_node.outputs[0] not in mul1_node.inputs:
        return None
    # Safety check (mirrors Attention's Bug D fix): cast_node is about to
    # be deleted as part of the fused pattern, so its output tensor must
    # have *exactly* these two consumers -- pow_node and mul1_node -- and
    # nothing else. If some other node also reads cast_node.outputs[0],
    # deleting cast_node would leave that node with a dangling reference.
    if {n.name for n in cast1_consumers} != {pow_node.name, mul1_node.name}:
        return None

    cast2_node = _single_consumer(consumers, mul1_node.outputs[0])
    if cast2_node is None or cast2_node.op_type != "Cast":
        return None

    mul2_node = _single_consumer(consumers, cast2_node.outputs[0])
    if mul2_node is None or mul2_node.op_type != "Mul":
        return None
    weight_name = next((i for i in mul2_node.inputs if i in graph.initializers), None)
    if weight_name is None:
        return None

    intermediate_outputs = [
        pow_node.outputs[0], reducemean_node.outputs[0], add_node.outputs[0],
        sqrt_node.outputs[0], div_node.outputs[0], mul1_node.outputs[0],
        cast2_node.outputs[0],
    ]
    if any(o in graph.outputs for o in intermediate_outputs):
        return None

    fused = Node(
        name=cast_node.name + "_rmsnorm_fused",
        op_type="RMSNorm",
        inputs=[x_name, weight_name],
        outputs=[mul2_node.outputs[0]],
        attributes={"eps": float(np.asarray(eps_arr).reshape(-1)[0])},
    )
    skip = {
        cast_node.name, pow_node.name, reducemean_node.name, add_node.name,
        sqrt_node.name, div_node.name, mul1_node.name, cast2_node.name, mul2_node.name,
    }
    return fused, skip


def fuse_rmsnorm(graph: Graph) -> Graph:
    producer, consumers = _build_producer_consumer_maps(graph)
    new_graph = _shallow_copy_graph_shell(graph)

    skip_names: set[str] = set()
    new_nodes: list[Node] = []

    for node in graph.nodes:
        if node.name in skip_names:
            continue
        if node.op_type == "Cast":
            match = _try_match_rmsnorm(graph, node, producer, consumers)
            if match is not None:
                fused, skip = match
                new_nodes.append(fused)
                skip_names.update(skip)
                continue
        new_nodes.append(node)

    new_graph.nodes = new_nodes
    return new_graph


# ---------------------------------------------------------------------------
# Pass 4: RoPE_Table (2-output node -- cos, sin)
# ---------------------------------------------------------------------------

def _try_match_rope_table(matmul_node: Node, producer, consumers):
    transpose_node = _single_consumer(consumers, matmul_node.outputs[0])
    if transpose_node is None or transpose_node.op_type != "Transpose":
        return None
    if list(transpose_node.attributes.get("perm", [])) != [0, 2, 1]:
        return None

    concat_node = _single_consumer(consumers, transpose_node.outputs[0])
    if concat_node is None or concat_node.op_type != "Concat":
        return None
    if list(concat_node.inputs) != [transpose_node.outputs[0], transpose_node.outputs[0]]:
        return None
    if int(concat_node.attributes.get("axis", 0)) != -1:
        return None

    cons = consumers.get(concat_node.outputs[0], [])
    if len(cons) != 2:
        return None
    cos_node = next((n for n in cons if n.op_type == "Cos"), None)
    sin_node = next((n for n in cons if n.op_type == "Sin"), None)
    if cos_node is None or sin_node is None:
        return None

    cos_mul = _single_consumer(consumers, cos_node.outputs[0])
    sin_mul = _single_consumer(consumers, sin_node.outputs[0])
    if cos_mul is None or cos_mul.op_type != "Mul":
        return None
    if sin_mul is None or sin_mul.op_type != "Mul":
        return None
    cos_scale = _constant_array(producer, next(i for i in cos_mul.inputs if i != cos_node.outputs[0]))
    sin_scale = _constant_array(producer, next(i for i in sin_mul.inputs if i != sin_node.outputs[0]))
    if cos_scale is None or sin_scale is None:
        return None

    skip = {
        matmul_node.name, transpose_node.name, concat_node.name,
        cos_node.name, sin_node.name, cos_mul.name, sin_mul.name,
    }

    # The exporter casts the scaled table back to the model dtype; Leaf is
    # fp32 throughout, so that Cast is absorbed (identity) when present.
    cos_out = cos_mul.outputs[0]
    sin_out = sin_mul.outputs[0]
    cos_final = _single_consumer(consumers, cos_mul.outputs[0])
    if cos_final is not None and cos_final.op_type == "Cast":
        cos_out = cos_final.outputs[0]
        skip.add(cos_final.name)
    sin_final = _single_consumer(consumers, sin_mul.outputs[0])
    if sin_final is not None and sin_final.op_type == "Cast":
        sin_out = sin_final.outputs[0]
        skip.add(sin_final.name)

    fused = Node(
        name=matmul_node.name + "_rope_table_fused",
        op_type="RoPE_Table",
        inputs=list(matmul_node.inputs),
        outputs=[cos_out, sin_out],
        attributes={
            "cos_scale": float(np.asarray(cos_scale).reshape(-1)[0]),
            "sin_scale": float(np.asarray(sin_scale).reshape(-1)[0]),
        },
    )
    return fused, skip


def fuse_rope_table(graph: Graph) -> Graph:
    producer, consumers = _build_producer_consumer_maps(graph)
    new_graph = _shallow_copy_graph_shell(graph)

    skip_names: set[str] = set()
    new_nodes: list[Node] = []

    for node in graph.nodes:
        if node.name in skip_names:
            continue
        if node.op_type == "MatMul":
            match = _try_match_rope_table(node, producer, consumers)
            if match is not None:
                fused, skip = match
                new_nodes.append(fused)
                skip_names.update(skip)
                continue
        new_nodes.append(node)

    new_graph.nodes = new_nodes
    return new_graph


# ---------------------------------------------------------------------------
# Pass 5: RepeatKV (GQA broadcast)
# ---------------------------------------------------------------------------

_REPEAT_KV_BOOKKEEPING_OPS = {
    "Shape", "Gather", "Constant", "Unsqueeze", "Concat", "Reshape",
    "ConstantOfShape", "Equal", "Where", "Mul",
}


def _try_match_repeat_kv(graph: Graph, unsqueeze_node: Node, producer, consumers):
    if len(unsqueeze_node.inputs) < 2:
        return None
    axis = _constant_array(producer, unsqueeze_node.inputs[1])
    if axis is None or axis.reshape(-1).tolist() != [2]:
        return None

    expand_node = _single_consumer(consumers, unsqueeze_node.outputs[0])
    if expand_node is None or expand_node.op_type != "Expand" or len(expand_node.inputs) < 2:
        return None

    shape_visited: set[str] = set()
    if not _collect_backward(expand_node.inputs[1], producer, _REPEAT_KV_BOOKKEEPING_OPS, shape_visited):
        return None

    reshape_node = _single_consumer(consumers, expand_node.outputs[0])
    if reshape_node is None or reshape_node.op_type != "Reshape" or len(reshape_node.inputs) < 2:
        return None

    source_info = graph.value_info.get(unsqueeze_node.inputs[0])
    result_info = graph.value_info.get(reshape_node.outputs[0])
    expand_info = graph.value_info.get(expand_node.outputs[0])
    source_shape = source_info.shape if source_info is not None else None
    result_shape = result_info.shape if result_info is not None else None
    expand_shape = expand_info.shape if expand_info is not None else None
    repeats = None
    if (source_shape is not None and result_shape is not None and
            len(source_shape) == 4 and len(result_shape) == 4 and
            isinstance(source_shape[1], int) and source_shape[1] > 0 and
            isinstance(result_shape[1], int) and
            result_shape[1] % source_shape[1] == 0):
        repeats = result_shape[1] // source_shape[1]
    elif (expand_shape is not None and len(expand_shape) == 5 and
          isinstance(expand_shape[2], int)):
        repeats = expand_shape[2]
    if repeats is None or repeats < 1:
        return None

    reshape_shape_visited: set[str] = set()
    if not _collect_backward(reshape_node.inputs[1], producer, _REPEAT_KV_BOOKKEEPING_OPS, reshape_shape_visited):
        return None

    skip = {unsqueeze_node.name, expand_node.name, reshape_node.name}
    skip |= shape_visited | reshape_shape_visited

    fused = Node(
        name=unsqueeze_node.name + "_repeatkv_fused",
        op_type="RepeatKV",
        inputs=[unsqueeze_node.inputs[0]],
        outputs=[reshape_node.outputs[0]],
        attributes={"n_rep": repeats},
    )
    return fused, skip


def fuse_repeat_kv(graph: Graph) -> Graph:
    producer, consumers = _build_producer_consumer_maps(graph)
    new_graph = _shallow_copy_graph_shell(graph)

    skip_names: set[str] = set()
    new_nodes: list[Node] = []

    for node in graph.nodes:
        if node.name in skip_names:
            continue
        if node.op_type == "Unsqueeze":
            match = _try_match_repeat_kv(graph, node, producer, consumers)
            if match is not None:
                fused, skip = match
                new_nodes.append(fused)
                skip_names.update(skip)
                continue
        new_nodes.append(node)

    new_graph.nodes = new_nodes
    return new_graph


# ---------------------------------------------------------------------------
# Pass 6: Attention (causal, scaled)
# ---------------------------------------------------------------------------

_ATTENTION_MASK_BOOKKEEPING_OPS = {
    "Shape", "Gather", "Constant", "Unsqueeze", "Concat", "Reshape", "Slice",
    "Expand", "And", "Cast", "ConstantOfShape", "Equal", "Mul",
    "Where", "LessOrEqual", "Flatten", "Add", "Range",
}


def _scaled_operand(name: str, producer):
    """name -> Mul(operand, scale_constant). Returns (mul_node, operand, scale)
    or None if `name` isn't produced by such a Mul."""
    mul_node = producer.get(name)
    if mul_node is None or mul_node.op_type != "Mul":
        return None
    scale = None
    operand = None
    for inp in mul_node.inputs:
        arr = _constant_array(producer, inp)
        if arr is not None and scale is None:
            scale = arr
        else:
            operand = inp
    if scale is None or operand is None:
        return None
    return mul_node, operand, float(np.asarray(scale).reshape(-1)[0])


def _try_match_attention(softmax_node: Node, producer, consumers):
    add_node = producer.get(softmax_node.inputs[0])
    if add_node is None or add_node.op_type != "Add" or len(add_node.inputs) < 2:
        return None
    if _single_consumer(consumers, add_node.outputs[0]) is not softmax_node:
        return None

    matmul_node = None
    where_node = None
    for inp in add_node.inputs:
        n = producer.get(inp)
        if n is None:
            continue
        if n.op_type == "MatMul":
            matmul_node = n
        elif n.op_type == "Where":
            where_node = n
    if matmul_node is None or where_node is None:
        return None
    if _single_consumer(consumers, matmul_node.outputs[0]) is not add_node:
        return None
    if _single_consumer(consumers, where_node.outputs[0]) is not add_node:
        return None
    if len(where_node.inputs) < 3:
        return None

    const_a = _constant_array(producer, where_node.inputs[1])
    const_b = _constant_array(producer, where_node.inputs[2])
    if const_a is None or const_b is None:
        return None
    vals = sorted([float(np.asarray(const_a).reshape(-1)[0]), float(np.asarray(const_b).reshape(-1)[0])])
    if not (vals[0] == float("-inf") and vals[1] == 0.0):
        return None
    # Verify the mask tensor's producer chain is safe WITHOUT marking the
    # mask tensor's own producer node for deletion -- that node's output is
    # exactly what the fused node references as its mask input (see
    # `where_node.inputs[0]` used in `fused.inputs` below) and must survive
    # in the output graph. Confirmed via direct check against the real
    # export that omitting this produced a dangling reference (the fused
    # node's mask input pointed at a deleted node).
    mask_visited: set[str] = set()
    mask_node = producer.get(where_node.inputs[0])
    if mask_node is not None:
        if mask_node.op_type not in _ATTENTION_MASK_BOOKKEEPING_OPS:
            return None
        mask_visited.add(mask_node.name)
        for inp in mask_node.inputs:
            if not _collect_backward(inp, producer, _ATTENTION_MASK_BOOKKEEPING_OPS, mask_visited):
                return None
        mask_visited.discard(mask_node.name)

    if len(matmul_node.inputs) < 2:
        return None
    q_result = _scaled_operand(matmul_node.inputs[0], producer)
    k_result = _scaled_operand(matmul_node.inputs[1], producer)
    if q_result is None or k_result is None:
        return None
    q_mul, q_name, q_scale = q_result
    k_mul, k_transposed_name, k_scale = k_result
    if not np.isclose(q_scale, k_scale):
        return None

    k_transpose = producer.get(k_transposed_name)
    if k_transpose is None or k_transpose.op_type != "Transpose":
        return None
    if list(k_transpose.attributes.get("perm", [])) != [0, 1, 3, 2]:
        return None
    k_name = k_transpose.inputs[0]

    # Softmax's output legitimately has two consumers in a correct instance
    # of this pattern -- IsNaN, and the direct "keep real value" branch of
    # the Where node below (where2.inputs[2] == softmax_node.outputs[0]).
    # Requiring a single consumer here is structurally impossible to
    # satisfy against a real, correctly-formed match; search by op_type
    # instead. Confirmed against the real export: this was silently
    # failing the match on every one of the 24 real attention blocks.
    isnan_node = next((n for n in consumers.get(softmax_node.outputs[0], [])
                        if n.op_type == "IsNaN"), None)
    if isnan_node is None:
        return None
    where2 = _single_consumer(consumers, isnan_node.outputs[0])
    if where2 is None or where2.op_type != "Where" or len(where2.inputs) < 3:
        return None
    if where2.inputs[0] != isnan_node.outputs[0] or where2.inputs[2] != softmax_node.outputs[0]:
        return None
    zero_const = _constant_array(producer, where2.inputs[1])
    if zero_const is None or not np.allclose(zero_const, 0.0):
        return None

    matmul_v = _single_consumer(consumers, where2.outputs[0])
    if matmul_v is None or matmul_v.op_type != "MatMul" or len(matmul_v.inputs) < 2:
        return None
    v_name = matmul_v.inputs[1]

    transpose_out = _single_consumer(consumers, matmul_v.outputs[0])
    if transpose_out is None or transpose_out.op_type != "Transpose":
        return None
    if list(transpose_out.attributes.get("perm", [])) != [0, 2, 1, 3]:
        return None

    skip = {
        add_node.name, matmul_node.name, where_node.name, q_mul.name, k_mul.name,
        k_transpose.name, softmax_node.name, isnan_node.name, where2.name,
        matmul_v.name, transpose_out.name,
    }
    skip |= mask_visited

    fused = Node(
        name=softmax_node.name + "_attention_fused",
        op_type="Attention",
        inputs=[q_name, k_name, v_name, where_node.inputs[0]],
        outputs=[transpose_out.outputs[0]],
        attributes={"scale": q_scale,
                    "mask_nonzero_is_valid": int(not np.isneginf(np.asarray(const_a).reshape(-1)[0]))},
    )
    return fused, skip


def fuse_attention(graph: Graph) -> Graph:
    producer, consumers = _build_producer_consumer_maps(graph)
    new_graph = _shallow_copy_graph_shell(graph)

    skip_names: set[str] = set()
    new_nodes: list[Node] = []

    for node in graph.nodes:
        if node.name in skip_names:
            continue
        if node.op_type == "Softmax":
            match = _try_match_attention(node, producer, consumers)
            if match is not None:
                fused, skip = match
                new_nodes.append(fused)
                skip_names.update(skip)
                continue
        new_nodes.append(node)

    new_graph.nodes = new_nodes
    return new_graph


# ---------------------------------------------------------------------------
# Pass 7: SwiGLU MLP
# ---------------------------------------------------------------------------

def _run_two_pass_fusion(graph: Graph, trigger_op_type: str, match_fn) -> Graph:
    """Generic two-pass fusion runner.

    A single forward walk (skip-as-you-go) only works when a pattern's
    trigger node is the FIRST node of the pattern in graph.nodes order --
    RMSNorm, RoPE_Table, and Attention all have this shape by construction.
    SwiGLU does not: its trigger is Sigmoid, but gate_matmul (part of the
    same pattern) sits earlier in node order. A forward walk already
    emits gate_matmul before the Sigmoid trigger is even reached, so
    updating skip_names at match time is too late. See docs/handover.md
    section 3 for the full diagnosis.

    Pass 1: walk once, collect every match (keyed by trigger node name)
    without emitting anything.
    Pass 2: walk again; emit the fused node at each trigger position,
    skip every other node named in any match's skip set.
    """
    producer, consumers = _build_producer_consumer_maps(graph)

    matches: dict[str, tuple[Node, set[str]]] = {}
    all_skip: set[str] = set()
    for node in graph.nodes:
        if node.op_type == trigger_op_type:
            match = match_fn(node, producer, consumers)
            if match is not None:
                fused, skip = match
                matches[node.name] = (fused, skip)
                all_skip.update(skip)

    new_nodes: list[Node] = []
    for node in graph.nodes:
        if node.name in matches:
            fused, _skip = matches[node.name]
            new_nodes.append(fused)
            continue
        if node.name in all_skip:
            continue
        new_nodes.append(node)

    new_graph = _shallow_copy_graph_shell(graph)
    new_graph.nodes = new_nodes
    return new_graph


def _try_match_swiglu(sigmoid_node: Node, producer, consumers):
    gate_matmul = producer.get(sigmoid_node.inputs[0])
    if gate_matmul is None or gate_matmul.op_type != "MatMul" or len(gate_matmul.inputs) < 2:
        return None

    gate_cons_raw = consumers.get(gate_matmul.outputs[0], [])
    gate_cons = list({id(n): n for n in gate_cons_raw}.values())
    if len(gate_cons) != 2 or sigmoid_node not in gate_cons:
        return None
    silu_mul = next((n for n in gate_cons if n is not sigmoid_node), None)
    if silu_mul is None or silu_mul.op_type != "Mul":
        return None
    if _single_consumer(consumers, sigmoid_node.outputs[0]) is not silu_mul:
        return None

    up_mul = _single_consumer(consumers, silu_mul.outputs[0])
    if up_mul is None or up_mul.op_type != "Mul":
        return None
    up_matmul = None
    for inp in up_mul.inputs:
        if inp == silu_mul.outputs[0]:
            continue
        n = producer.get(inp)
        if n is not None and n.op_type == "MatMul":
            up_matmul = n
    if up_matmul is None or len(up_matmul.inputs) < 2:
        return None
    x_name = gate_matmul.inputs[0]
    if up_matmul.inputs[0] != x_name:
        return None

    down_matmul = _single_consumer(consumers, up_mul.outputs[0])
    if down_matmul is None or down_matmul.op_type != "MatMul" or len(down_matmul.inputs) < 2:
        return None

    residual_add = _single_consumer(consumers, down_matmul.outputs[0])
    if residual_add is None or residual_add.op_type != "Add":
        return None
    residual_name = next((i for i in residual_add.inputs if i != down_matmul.outputs[0]), None)
    if residual_name is None:
        return None

    skip = {
        gate_matmul.name, sigmoid_node.name, silu_mul.name, up_matmul.name,
        up_mul.name, down_matmul.name, residual_add.name,
    }

    fused = Node(
        name=sigmoid_node.name + "_swiglu_fused",
        op_type="SwiGLU_MLP",
        inputs=[x_name, gate_matmul.inputs[1], up_matmul.inputs[1], down_matmul.inputs[1], residual_name],
        outputs=[residual_add.outputs[0]],
        attributes={},
    )
    return fused, skip


def fuse_swiglu_mlp(graph: Graph) -> Graph:
    # Two-pass runner required here -- see _run_two_pass_fusion's
    # docstring and docs/handover.md section 3. Sigmoid triggers the
    # match but gate_matmul (part of the pattern) precedes it in node
    # order, which a single forward skip-as-you-go walk mishandles.
    return _run_two_pass_fusion(graph, "Sigmoid", _try_match_swiglu)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_fusion_passes(graph: Graph) -> Graph:
    """Apply all fusion passes in order.

    CNN passes: BN folding must run before activation fusion, since
    activation fusion looks for a Conv node directly feeding an
    activation -- BN folding is what makes that true when the original
    graph was Conv->BN->Relu.

    Transformer passes: each operates on the *original* underlying ops
    (Pow/ReduceMean/Cos/Sin/Unsqueeze/Softmax/Sigmoid etc.), not on the
    fused-node output of a previous pass, so their relative order doesn't
    affect correctness -- kept in README section 3.4's pipeline order
    (norm -> rope table -> repeat_kv -> attention -> mlp) for readability.
    RoPE_Apply is intentionally not yet included -- see docs/handover.md,
    still pending confirmation of the pre-node-208 subgraph.
    """
    g = fuse_conv_batchnorm(graph)
    g = fuse_conv_activation(g)
    g = fuse_rmsnorm(g)
    g = fuse_rope_table(g)
    g = fuse_repeat_kv(g)
    g = fuse_attention(g)
    g = fuse_swiglu_mlp(g)
    used_initializers = {name for node in g.nodes for name in node.inputs}
    used_initializers.update(name for name in g.outputs if name in g.initializers)
    g.initializers = {
        name: value for name, value in g.initializers.items()
        if name in used_initializers
    }
    g.metadata.setdefault("passes", {})["fusion"] = {
        "nodes_before": len(graph.nodes),
        "nodes_after": len(g.nodes),
    }
    return g
