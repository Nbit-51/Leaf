"""Compile-time evaluation of small constant-only subgraphs."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .executor import run_graph
from .ir import Graph, TensorInfo


FOLDABLE_OPS = {
    "Constant", "Identity", "Add", "Sub", "Mul", "Div", "Pow",
    "MatMul", "Gemm", "Reshape", "Transpose", "Concat", "Squeeze",
    "Unsqueeze", "Gather", "Shape", "Cast",
}


def fold_constants(graph: Graph, max_constant_bytes: int = 16 * 1024 * 1024) -> Graph:
    """Replace constant-only nodes with initializers.

    The byte limit prevents an innocent shape expression from materialising a
    very large tensor in the deployment artifact. Unsupported ops and dynamic
    branches are preserved unchanged.
    """
    result = graph.clone()
    constants = dict(result.initializers)
    kept = []
    folded_nodes: list[str] = []

    for node in result.topological_order():
        can_fold = node.op_type in FOLDABLE_OPS and all(name in constants for name in node.inputs)
        if not can_fold:
            kept.append(node)
            continue

        mini = Graph()
        mini.nodes = [deepcopy(node)]
        mini.initializers = {name: constants[name] for name in node.inputs}
        values = run_graph(mini, {})
        outputs = [np.asarray(values[name]) for name in node.outputs]
        if sum(value.nbytes for value in outputs) > max_constant_bytes:
            kept.append(node)
            continue

        for name, value in zip(node.outputs, outputs):
            constants[name] = value
            result.initializers[name] = value
            result.value_info[name] = TensorInfo(name, value.shape, value.dtype.name)
        folded_nodes.append(node.name)

    result.nodes = kept
    used = {name for node in result.nodes for name in node.inputs} | set(result.outputs)
    result.initializers = {
        name: value for name, value in result.initializers.items() if name in used
    }
    result.metadata.setdefault("passes", {})["constant_folding"] = {
        "folded_nodes": folded_nodes,
        "count": len(folded_nodes),
    }
    result.validate()
    return result
