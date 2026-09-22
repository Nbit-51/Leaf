"""Static liveness analysis and reusable activation-arena planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .ir import Graph


_DTYPE_ALIASES = {"float": "float32", "double": "float64"}


@dataclass(frozen=True)
class Allocation:
    tensor: str
    offset: int
    size: int
    first_node: int
    last_node: int


@dataclass
class MemoryPlan:
    arena_size: int
    naive_size: int
    alignment: int
    allocations: list[Allocation]
    unplanned_tensors: list[str]
    graph_fingerprint: str

    @property
    def bytes_saved(self) -> int:
        return self.naive_size - self.arena_size

    def to_dict(self) -> dict:
        return {
            "format": "leaf-memory-plan-v1",
            "graph_fingerprint": self.graph_fingerprint,
            "alignment": self.alignment,
            "arena_size": self.arena_size,
            "naive_size": self.naive_size,
            "bytes_saved": self.bytes_saved,
            "allocations": [asdict(item) for item in self.allocations],
            "unplanned_tensors": self.unplanned_tensors,
        }

    def export_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return destination


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _fingerprint(graph: Graph) -> str:
    topology = [
        (node.name, node.op_type, tuple(node.inputs), tuple(node.outputs))
        for node in graph.topological_order()
    ]
    return hashlib.sha256(repr((graph.inputs, graph.outputs, topology)).encode()).hexdigest()[:16]


def _tensor_size(graph: Graph, name: str, alignment: int) -> int | None:
    info = graph.value_info.get(name)
    if info is None or info.shape is None or info.dtype is None or any(dim is None for dim in info.shape):
        return None
    dtype_name = _DTYPE_ALIASES.get(info.dtype, info.dtype)
    try:
        byte_count = int(np.prod(info.shape, dtype=np.int64)) * np.dtype(dtype_name).itemsize
    except TypeError:
        return None
    return _align(byte_count, alignment)


def plan_memory(graph: Graph, alignment: int = 64) -> MemoryPlan:
    """Plan one aligned arena using best-fit reuse of non-overlapping values."""
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    order = graph.topological_order()
    producer_index = {output: index for index, node in enumerate(order) for output in node.outputs}
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(order):
        for name in node.inputs:
            consumers.setdefault(name, []).append(index)

    intervals: list[tuple[int, int, int, str]] = []
    unplanned: list[str] = []
    for name, first in producer_index.items():
        size = _tensor_size(graph, name, alignment)
        if size is None:
            unplanned.append(name)
            continue
        uses = consumers.get(name, [])
        last = max(uses) if uses else first
        if name in graph.outputs:
            last = len(order)
        intervals.append((first, last, size, name))

    active: list[tuple[int, int, int]] = []
    free: list[tuple[int, int]] = []
    allocations: list[Allocation] = []
    arena_size = 0
    for first, last, size, name in sorted(intervals, key=lambda item: (item[0], -item[2], item[3])):
        still_active = []
        for active_end, offset, block_size in active:
            if active_end < first:
                free.append((offset, block_size))
            else:
                still_active.append((active_end, offset, block_size))
        active = still_active
        candidates = [(block_size, offset, index) for index, (offset, block_size) in enumerate(free) if block_size >= size]
        if candidates:
            block_size, offset, free_index = min(candidates)
            free.pop(free_index)
            if block_size > size:
                free.append((offset + size, block_size - size))
        else:
            offset = arena_size
            arena_size += size
        allocations.append(Allocation(name, offset, size, first, last))
        active.append((last, offset, size))

    allocations.sort(key=lambda item: (item.offset, item.first_node, item.tensor))
    return MemoryPlan(
        arena_size=arena_size,
        naive_size=sum(item[2] for item in intervals),
        alignment=alignment,
        allocations=allocations,
        unplanned_tensors=sorted(unplanned),
        graph_fingerprint=_fingerprint(graph),
    )
