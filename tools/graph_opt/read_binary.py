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
_DTYPES = {0: np.float32, 1: np.int8}


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
    if version not in (1, 2, 3):
        raise ValueError(f"unsupported .leaf version: {version}")

    graph_inputs, offset = _read_string_array(data, offset)
    graph_outputs, offset = _read_string_array(data, offset)

    nodes = []
    for _ in range(node_count):
        op_type, offset = _read_string(data, offset)
        inputs, offset = _read_string_array(data, offset)
        outputs, offset = _read_string_array(data, offset)
        attrs_json, offset = _read_string(data, offset)
        attributes = json.loads(attrs_json)
        if version >= 2:
            (quantized,) = struct.unpack_from("<B", data, offset)
            offset += 1
            if quantized not in (0, 1):
                raise ValueError(f"invalid quantization flag: {quantized}")
            if quantized:
                input_scale, axis, scale_count = struct.unpack_from("<fBI", data, offset)
                offset += 9
                scales = np.frombuffer(data, dtype="<f4", count=scale_count,
                                       offset=offset).copy()
                offset += 4 * scale_count
                attributes["quantization"] = {
                    "scheme": "symmetric_int8",
                    "input": {"scale": float(input_scale), "zero_point": 0},
                    "weight": {"scale": scales, "zero_point": 0, "axis": int(axis)},
                }
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

    memory_plan = None
    if version >= 3:
        alignment, arena_size = struct.unpack_from("<IQ", data, offset)
        offset += 12
        fingerprint, offset = _read_string(data, offset)
        (allocation_count,) = struct.unpack_from("<I", data, offset)
        offset += 4
        allocations = []
        for _ in range(allocation_count):
            tensor, offset = _read_string(data, offset)
            allocation_offset, size, first_node, last_node = struct.unpack_from(
                "<QQII", data, offset)
            offset += 24
            allocations.append({"tensor": tensor, "offset": allocation_offset,
                                "size": size, "first_node": first_node,
                                "last_node": last_node})
        unplanned, offset = _read_string_array(data, offset)
        memory_plan = {"alignment": alignment, "arena_size": arena_size,
                       "graph_fingerprint": fingerprint,
                       "allocations": allocations, "unplanned_tensors": unplanned}

    data_block_start = (offset + 31) // 32 * 32 if version >= 2 else offset
    initializers = {}
    for meta in init_meta:
        start = data_block_start + meta["byte_offset"]
        end = start + meta["byte_length"]
        if end > len(data):
            raise ValueError(f"truncated initializer: {meta['name']}")
        if meta["dtype_tag"] not in _DTYPES:
            raise ValueError(f"unsupported initializer dtype tag: {meta['dtype_tag']}")
        if version == 1 and meta["dtype_tag"] != 0:
            raise ValueError("version 1 only supports float32 initializers")
        arr = np.frombuffer(data[start:end], dtype=_DTYPES[meta["dtype_tag"]]).reshape(
            meta["shape"]
        )
        initializers[meta["name"]] = arr

    return {
        "version": version,
        "inputs": graph_inputs,
        "outputs": graph_outputs,
        "nodes": nodes,
        "initializers": initializers,
        "memory_plan": memory_plan,
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
