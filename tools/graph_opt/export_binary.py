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
    I    version     format version (starts at 1)
    I    node_count
    I    initializer_count
    <graph input names, length-prefixed>
    <graph output names, length-prefixed>

  Node table (node_count entries):
    <op_type, length-prefixed string>
    <input_names, length-prefixed string array>
    <output_names, length-prefixed string array>
    <attributes_json, length-prefixed string -- JSON-encoded attrs dict>

  Initializer table (initializer_count entries):
    <name, length-prefixed string>
    I           ndims
    I[ndims]    shape
    B           dtype tag (0 = float32; only float32 supported for now)
    Q           byte_offset (into data block)
    Q           byte_length

  Data block:
    raw float32 bytes for every initializer, back-to-back, in the same
    order as the initializer table, 32-byte aligned (each initializer's
    start offset is padded to a 32-byte boundary so the C++ side can
    load directly into AVX2-aligned buffers without a realignment copy).

All lengths/counts are uint32 unless noted. Strings are UTF-8, written as
<uint32 length><raw bytes>, no null terminator.
"""

from __future__ import annotations

import json
import struct

import numpy as np

from ir import Graph

MAGIC = b"LEAF"
VERSION = 1
DTYPE_FLOAT32 = 0
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
        attrs_json = json.dumps(node.attributes)
        _write_string(node_table, attrs_json)

    # Two passes over initializers: first compute aligned offsets, then
    # write the initializer table (which needs those offsets), then the
    # data block itself.
    init_items = list(graph.initializers.items())
    offsets: list[int] = []
    running_offset = 0
    for _, array in init_items:
        running_offset = _pad_to_alignment(running_offset)
        offsets.append(running_offset)
        running_offset += array.astype(np.float32).nbytes

    init_table = bytearray()
    for (name, array), offset in zip(init_items, offsets):
        arr32 = array.astype(np.float32)
        _write_string(init_table, name)
        init_table += struct.pack("<I", arr32.ndim)
        for dim in arr32.shape:
            init_table += struct.pack("<I", dim)
        init_table += struct.pack("<B", DTYPE_FLOAT32)
        init_table += struct.pack("<Q", offset)
        init_table += struct.pack("<Q", arr32.nbytes)

    data_block = bytearray()
    for (_, array), offset in zip(init_items, offsets):
        if len(data_block) < offset:
            data_block += b"\x00" * (offset - len(data_block))
        data_block += array.astype(np.float32).tobytes()

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(node_table)
        f.write(init_table)
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
