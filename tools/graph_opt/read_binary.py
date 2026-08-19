"""
Leaf Binary Graph Reader (verification tool)
==============================================

Pure-Python reader for Leaf's binary graph format (see export_binary.py
for the format spec). This exists purely to verify the exporter is
correct BEFORE writing a C++ parser -- round-tripping through Python,
where a format bug is a five-second traceback, is much cheaper than
debugging the same bug from inside C++.

Not meant to be fast or to be the "real" reader -- the C++ engine gets
its own independent parser for the same format.
"""

from __future__ import annotations

import json
import struct

import numpy as np

MAGIC = b"LEAF"


def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<I", data, offset)
    offset += 4
    s = data[offset:offset + length].decode("utf-8")
    offset += length
    return s, offset


def _read_string_array(data: bytes, offset: int) -> tuple[list[str], int]:
    (count,) = struct.unpack_from("<I", data, offset)
    offset += 4
    items = []
    for _ in range(count):
        s, offset = _read_string(data, offset)
        items.append(s)
    return items, offset


def read_graph(path: str) -> dict:
    """Parse a .leaf binary file, returning a dict with the same shape of
    information the exporter wrote: nodes, initializer metadata + arrays,
    and graph inputs/outputs. Used for round-trip verification."""
    with open(path, "rb") as f:
        data = f.read()

    offset = 0
    magic = data[offset:offset + 4]
    offset += 4
    assert magic == MAGIC, f"bad magic: {magic!r}"

    (version, node_count, initializer_count) = struct.unpack_from("<III", data, offset)
    offset += 12

    graph_inputs, offset = _read_string_array(data, offset)
    graph_outputs, offset = _read_string_array(data, offset)

    nodes = []
    for _ in range(node_count):
        op_type, offset = _read_string(data, offset)
        inputs, offset = _read_string_array(data, offset)
        outputs, offset = _read_string_array(data, offset)
        attrs_json, offset = _read_string(data, offset)
        attributes = json.loads(attrs_json)
        nodes.append({
            "op_type": op_type,
            "inputs": inputs,
            "outputs": outputs,
            "attributes": attributes,
        })

    init_meta = []
    for _ in range(initializer_count):
        name, offset = _read_string(data, offset)
        (ndims,) = struct.unpack_from("<I", data, offset)
        offset += 4
        shape = struct.unpack_from(f"<{ndims}I", data, offset)
        offset += 4 * ndims
        (dtype_tag,) = struct.unpack_from("<B", data, offset)
        offset += 1
        (byte_offset, byte_length) = struct.unpack_from("<QQ", data, offset)
        offset += 16
        init_meta.append({
            "name": name,
            "shape": shape,
            "dtype_tag": dtype_tag,
            "byte_offset": byte_offset,
            "byte_length": byte_length,
        })

    data_block_start = offset
    initializers = {}
    for meta in init_meta:
        start = data_block_start + meta["byte_offset"]
        end = start + meta["byte_length"]
        arr = np.frombuffer(data[start:end], dtype=np.float32).reshape(meta["shape"])
        initializers[meta["name"]] = arr

    return {
        "version": version,
        "inputs": graph_inputs,
        "outputs": graph_outputs,
        "nodes": nodes,
        "initializers": initializers,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 read_binary.py <input.leaf>")
        sys.exit(1)

    g = read_graph(sys.argv[1])
    print(f"version={g['version']}, nodes={len(g['nodes'])}, "
          f"initializers={len(g['initializers'])}")
    print(f"inputs={g['inputs']}, outputs={g['outputs']}")
    print(f"first node: {g['nodes'][0]}")
