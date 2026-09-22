"""
Leaf Graph IR
================

A minimal, framework-independent graph representation used as the shared
substrate for every optimization pass (fusion, quantization, pruning,
memory planning). This module only defines the data structures and an
ONNX loader -- no optimization logic lives here.

Design notes:
- Nodes reference tensors by name (string), not by object, mirroring ONNX's
  own approach. This keeps the graph easy to serialize and easy to rewrite
  (fusion passes just splice node lists and re-point names).
- Initializers (weights) are stored separately from the node graph as
  {name: np.ndarray}, since passes like quantization operate on these
  directly without touching graph topology.
- shape/dtype info is optional (value_info in ONNX is not always fully
  populated) -- passes should not assume every tensor has known shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from os import PathLike
from typing import Optional

import numpy as np
import onnx
from onnx import numpy_helper


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TensorInfo:
    """Static metadata about a tensor (not the data itself)."""
    name: str
    shape: Optional[tuple] = None      # None if unknown/dynamic
    dtype: Optional[str] = None        # numpy dtype string, e.g. "float32"


@dataclass
class Node:
    """A single operation in the graph."""
    name: str
    op_type: str                        # e.g. "Conv", "BatchNormalization", "Relu"
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        ins = ", ".join(self.inputs)
        outs = ", ".join(self.outputs)
        return f"{self.op_type}[{self.name}]({ins}) -> ({outs})"


class Graph:
    """
    Framework-independent computation graph.

    nodes:        ordered list of Node (order as loaded; not guaranteed
                   topologically sorted until topological_order() is called)
    inputs:       list of graph input tensor names
    outputs:      list of graph output tensor names
    initializers: {name: np.ndarray} -- the actual weight/bias data
    value_info:   {name: TensorInfo} -- shape/dtype hints where available
    """

    def __init__(self):
        self.nodes: list[Node] = []
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.initializers: dict[str, np.ndarray] = {}
        self.value_info: dict[str, TensorInfo] = {}
        self.metadata: dict = {}

    # -- construction ------------------------------------------------------

    @classmethod
    def from_onnx(cls, path_or_model) -> "Graph":
        """Load a Graph from an .onnx file path or an already-loaded ModelProto."""
        if isinstance(path_or_model, (str, bytes, PathLike)):
            model = onnx.load(path_or_model)
        else:
            model = path_or_model

        onnx.checker.check_model(model)
        # Exporters frequently omit intermediate value_info. Shape inference
        # makes memory planning useful without a model-specific shape pass.
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception:
            # Shape inference is an aid, not a load requirement. Custom-domain
            # operators are valid Leaf inputs even when ONNX cannot infer them.
            pass
        graph_proto = model.graph

        g = cls()

        # Initializers (weights/biases) -- convert straight to numpy.
        for init in graph_proto.initializer:
            g.initializers[init.name] = numpy_helper.to_array(init)

        # Graph-level inputs/outputs. Inputs that are actually initializers
        # (common in newer ONNX exports) are excluded -- they're weights,
        # not runtime inputs.
        g.inputs = [
            vi.name for vi in graph_proto.input
            if vi.name not in g.initializers
        ]
        g.outputs = [vi.name for vi in graph_proto.output]

        # Shape/dtype hints, where the exporter provided them.
        for vi in list(graph_proto.input) + list(graph_proto.output) + list(graph_proto.value_info):
            shape = None
            dtype = None
            if vi.type.tensor_type.HasField("shape"):
                dims = []
                for d in vi.type.tensor_type.shape.dim:
                    dims.append(d.dim_value if d.dim_value > 0 else None)
                shape = tuple(dims)
            if vi.type.tensor_type.elem_type:
                try:
                    dtype = np.dtype(
                        onnx.helper.tensor_dtype_to_np_dtype(vi.type.tensor_type.elem_type)
                    ).name
                except (TypeError, ValueError):
                    dtype = onnx.TensorProto.DataType.Name(
                        vi.type.tensor_type.elem_type
                    ).lower()
            g.value_info[vi.name] = TensorInfo(name=vi.name, shape=shape, dtype=dtype)

        # Nodes, in the order ONNX stored them.
        for i, n in enumerate(graph_proto.node):
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            node = Node(
                name=n.name if n.name else f"{n.op_type}_{i}",
                op_type=n.op_type,
                inputs=list(n.input),
                outputs=list(n.output),
                attributes=attrs,
            )
            g.nodes.append(node)

        return g

    def clone(self) -> "Graph":
        """Return an independent graph suitable for destructive rewrites."""
        cloned = Graph()
        cloned.nodes = deepcopy(self.nodes)
        cloned.inputs = list(self.inputs)
        cloned.outputs = list(self.outputs)
        cloned.initializers = {
            name: np.array(value, copy=True) for name, value in self.initializers.items()
        }
        cloned.value_info = deepcopy(self.value_info)
        cloned.metadata = deepcopy(self.metadata)
        return cloned

    def consumers(self) -> dict[str, list[Node]]:
        result: dict[str, list[Node]] = {}
        for node in self.nodes:
            for tensor in node.inputs:
                result.setdefault(tensor, []).append(node)
        return result

    def producers(self) -> dict[str, Node]:
        return {tensor: node for node in self.nodes for tensor in node.outputs}

    def validate(self) -> None:
        """Validate invariants relied on by graph-rewrite passes."""
        node_names: set[str] = set()
        tensor_names: set[str] = set(self.inputs) | set(self.initializers)
        for node in self.nodes:
            if node.name in node_names:
                raise ValueError(f"duplicate node name: {node.name!r}")
            node_names.add(node.name)
            for output in node.outputs:
                if output in tensor_names:
                    raise ValueError(f"tensor has multiple producers: {output!r}")
                tensor_names.add(output)
        missing = [name for name in self.outputs if name not in tensor_names]
        if missing:
            raise ValueError(f"graph outputs have no producer: {missing}")
        self.topological_order()

    # -- traversal -----------------------------------------------------

    def topological_order(self) -> list[Node]:
        """
        Return nodes in a valid topological order (dependency-respecting).
        Raises ValueError if a cycle is detected (shouldn't happen with
        valid ONNX, but graph-rewriting passes could introduce one by
        mistake -- better to fail loudly here than to segfault later in
        the C++ runtime).
        """
        producer = {}  # tensor_name -> Node that produces it
        for n in self.nodes:
            for out in n.outputs:
                producer[out] = n

        visited: set[str] = set()
        temp_mark: set[str] = set()
        order: list[Node] = []

        def visit(n: Node):
            if n.name in visited:
                return
            if n.name in temp_mark:
                raise ValueError(f"Cycle detected in graph at node '{n.name}'")
            temp_mark.add(n.name)
            for inp in n.inputs:
                if inp in producer:
                    visit(producer[inp])
            temp_mark.discard(n.name)
            visited.add(n.name)
            order.append(n)

        for n in self.nodes:
            visit(n)

        return order

    def op_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for n in self.nodes:
            counts[n.op_type] = counts.get(n.op_type, 0) + 1
        return counts

    def summary(self) -> str:
        lines = [
            f"Graph: {len(self.nodes)} nodes, {len(self.initializers)} initializers",
            f"  inputs:  {self.inputs}",
            f"  outputs: {self.outputs}",
            "  op counts:",
        ]
        for op, count in sorted(self.op_counts().items(), key=lambda kv: -kv[1]):
            lines.append(f"    {op}: {count}")
        return "\n".join(lines)
