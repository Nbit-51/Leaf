"""Graph rewrites for the dense feed-forward blocks used by transformers."""

from __future__ import annotations

from .fusion import _shallow_copy_graph_shell
from .ir import Graph, Node


_FUSABLE_ACTIVATIONS = {"Gelu", "Silu", "Relu"}


def fuse_matmul_bias(graph: Graph) -> Graph:
    """Fuse ``MatMul(x, constant_weight) + constant_bias`` into ``Gemm``."""
    result = _shallow_copy_graph_shell(graph)
    consumers = graph.consumers()
    skip: set[str] = set()
    nodes: list[Node] = []

    for node in graph.nodes:
        if node.name in skip:
            continue
        if node.op_type == "MatMul" and len(node.inputs) == 2 and node.inputs[1] in graph.initializers:
            output = node.outputs[0]
            next_nodes = consumers.get(output, [])
            if len(next_nodes) == 1 and next_nodes[0].op_type == "Add" and output not in graph.outputs:
                add = next_nodes[0]
                bias_inputs = [name for name in add.inputs if name != output]
                if len(bias_inputs) == 1 and bias_inputs[0] in graph.initializers:
                    bias = graph.initializers[bias_inputs[0]]
                    weight = graph.initializers[node.inputs[1]]
                    if bias.ndim <= 1 and bias.size == weight.shape[-1]:
                        nodes.append(Node(
                            name=f"{node.name}_biasfused",
                            op_type="Gemm",
                            inputs=[node.inputs[0], node.inputs[1], bias_inputs[0]],
                            outputs=list(add.outputs),
                            attributes={"alpha": 1.0, "beta": 1.0},
                        ))
                        skip.add(add.name)
                        continue
        nodes.append(node)

    result.nodes = nodes
    result.metadata.setdefault("passes", {})["matmul_bias_fusion"] = len(skip)
    return result


def fuse_linear_activation(graph: Graph) -> Graph:
    """Attach GELU/SiLU/ReLU to a Gemm so a runtime can use one kernel."""
    result = _shallow_copy_graph_shell(graph)
    consumers = graph.consumers()
    skip: set[str] = set()
    nodes: list[Node] = []
    for node in graph.nodes:
        if node.name in skip:
            continue
        if node.op_type == "Gemm":
            output = node.outputs[0]
            next_nodes = consumers.get(output, [])
            if (
                len(next_nodes) == 1
                and next_nodes[0].op_type in _FUSABLE_ACTIVATIONS
                and output not in graph.outputs
            ):
                activation = next_nodes[0]
                attrs = dict(node.attributes)
                attrs["activation"] = activation.op_type
                nodes.append(Node(
                    name=f"{node.name}_{activation.op_type.lower()}fused",
                    op_type="Gemm",
                    inputs=list(node.inputs),
                    outputs=list(activation.outputs),
                    attributes=attrs,
                ))
                skip.add(activation.name)
                continue
        nodes.append(node)
    result.nodes = nodes
    result.metadata.setdefault("passes", {})["linear_activation_fusion"] = len(skip)
    return result


def run_transformer_rewrites(graph: Graph) -> Graph:
    rewritten = fuse_matmul_bias(graph)
    rewritten = fuse_linear_activation(rewritten)
    rewritten.validate()
    return rewritten
