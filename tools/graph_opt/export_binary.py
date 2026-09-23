"""
Leaf Binary Graph Exporter
============================

Serializes a Graph (already loaded, fused, and topologically sorted) into
Leaf's binary format for consumption by the C++ inference engine. This is
the handoff boundary: Python builds and optimizes the graph once, then
writes a self-contained binary the C++ engine loads independently -- no
live Python<->C++ calls happen during inference.

Format (little-endian throughout):

  Header:
    4s   magic       b"LEAF"
    I    version     2 (reader also accepts legacy version 1)
    I    node_count
    I    initializer_count
    <graph input names, length-prefixed>
    <graph output names, length-prefixed>

  Node table (node_count entries):
    <op_type, length-prefixed string>
    <input_names, length-prefixed string array>
    <output_names, length-prefixed string array>
    <attributes_json, length-prefixed string -- JSON-encoded attrs dict>
    B           quantized flag (version 2 only)
    if quantized:
      f         input activation scale
      B         weight output-channel axis
      I         number of output-channel scales
      f[count]  output-channel scales

  Initializer table (initializer_count entries):
    <name, length-prefixed string>
    I           ndims
    I[ndims]    shape
    B           dtype tag (0 = float32, 1 = signed int8)
    Q           byte_offset (into data block)
    Q           byte_length

  Data block:
    raw typed bytes for every initializer, in the initializer-table order.
    The data block and each initializer start at a 32-byte boundary.

All lengths/counts are uint32 unless noted. Strings are UTF-8, written as
<uint32 length><raw bytes>, no null terminator.
"""

from __future__ import annotations

import json
import struct

import numpy as np

try:
    from .ir import Graph
except ImportError:
    from ir import Graph

MAGIC = b"LEAF"
VERSION = 2
DTYPE_FLOAT32 = 0
DTYPE_INT8 = 1
ALIGNMENT = 32


def _write_string(buf: bytearray, s: str) -> None:
    encoded = s.encode("utf-8")
    buf += struct.pack("<I", len(encoded))
    buf += encoded


def _write_string_array(buf: bytearray, strings: list[str]) -> None:
    buf += struct.pack("<I", len(strings))
    for s in strings:
        _write_string(buf, s)


def _pad_to_alignment(offset: int, alignment: int = ALIGNMENT) -> int:
    remainder = offset % alignment
    if remainder == 0:
        return offset
    return offset + (alignment - remainder)


def _initializer_payload(array: np.ndarray) -> tuple[int, bytes]:
    value = np.asarray(array)
    if value.dtype == np.int8:
        return DTYPE_INT8, np.ascontiguousarray(value).tobytes()
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"unsupported initializer dtype: {value.dtype}")
    return DTYPE_FLOAT32, np.ascontiguousarray(value, dtype=np.float32).tobytes()


def _json_compatible(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    raise TypeError(f"cannot encode attribute of type {type(value).__name__}")


def export_graph(graph: Graph, output_path: str) -> None:
    """Serialize `graph` to Leaf's binary format at `output_path`.

    Assumes `graph` is already fused/optimized and will be executed in
    `graph.nodes` order as given -- callers should run
    `graph.topological_order()` and rebuild the node list from that first
    if the graph wasn't already sorted (fusion passes typically preserve
    order, but this function does not re-sort).
    """
    header = bytearray()
    header += MAGIC
    header += struct.pack("<I", VERSION)
    header += struct.pack("<I", len(graph.nodes))
    header += struct.pack("<I", len(graph.initializers))
    _write_string_array(header, graph.inputs)
    _write_string_array(header, graph.outputs)

    node_table = bytearray()
    for node in graph.nodes:
        _write_string(node_table, node.op_type)
        _write_string_array(node_table, node.inputs)
        _write_string_array(node_table, node.outputs)
        attributes = dict(node.attributes)
        quantization = attributes.pop("quantization", None)
        attrs_json = json.dumps(attributes, default=_json_compatible)
        _write_string(node_table, attrs_json)
        if quantization is None:
            node_table += struct.pack("<B", 0)
        else:
            if quantization.get("scheme") != "symmetric_int8":
                raise ValueError(f"unsupported quantization scheme on {node.name}")
            if node.op_type not in ("Conv", "Gemm", "MatMul") or len(node.inputs) < 2:
                raise ValueError(f"unsupported quantized operation on {node.name}")
            weight_array = graph.initializers.get(node.inputs[1])
            if weight_array is None or np.asarray(weight_array).dtype != np.int8:
                raise ValueError(f"quantized weight must be an INT8 initializer on {node.name}")
            input_scale = float(quantization["input"]["scale"])
            weight = quantization["weight"]
            scales = np.asarray(weight["scale"], dtype=np.float32).reshape(-1)
            axis = int(weight["axis"])
            if (not np.isfinite(input_scale) or input_scale <= 0 or
                    scales.size == 0 or not np.all(np.isfinite(scales)) or
                    np.any(scales <= 0) or not 0 <= axis < np.asarray(weight_array).ndim or
                    scales.size != np.asarray(weight_array).shape[axis] or
                    quantization["input"].get("zero_point", 0) != 0 or
                    weight.get("zero_point", 0) != 0):
                raise ValueError(f"invalid INT8 scales or axis on {node.name}")
            node_table += struct.pack("<BfBI", 1, input_scale, axis, scales.size)
            node_table += scales.tobytes()

    # Two passes over initializers: first compute aligned offsets, then
    # write the initializer table (which needs those offsets), then the
    # data block itself.
    init_items = [(name, np.asarray(array), *_initializer_payload(array))
                  for name, array in graph.initializers.items()]
    offsets: list[int] = []
    running_offset = 0
    for _, _, _, payload in init_items:
        running_offset = _pad_to_alignment(running_offset)
        offsets.append(running_offset)
        running_offset += len(payload)

    init_table = bytearray()
    for (name, array, dtype_tag, payload), offset in zip(init_items, offsets):
        _write_string(init_table, name)
        init_table += struct.pack("<I", array.ndim)
        for dim in array.shape:
            init_table += struct.pack("<I", dim)
        init_table += struct.pack("<B", dtype_tag)
        init_table += struct.pack("<Q", offset)
        init_table += struct.pack("<Q", len(payload))

    data_block = bytearray()
    for (_, _, _, payload), offset in zip(init_items, offsets):
        if len(data_block) < offset:
            data_block += b"\x00" * (offset - len(data_block))
        data_block += payload

    table_size = len(header) + len(node_table) + len(init_table)
    table_padding = _pad_to_alignment(table_size) - table_size

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(node_table)
        f.write(init_table)
        f.write(b"\x00" * table_padding)
        f.write(data_block)


def _human_readable_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python3 export_binary.py <input.onnx> <output.leaf>")
        sys.exit(1)

    g = Graph.from_onnx(sys.argv[1])
    export_graph(g, sys.argv[2])

    import os
    size = os.path.getsize(sys.argv[2])
    print(f"Exported {len(g.nodes)} nodes, {len(g.initializers)} initializers "
          f"-> {sys.argv[2]} ({_human_readable_size(size)})")
