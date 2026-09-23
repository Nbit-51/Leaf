"""Calibration and symmetric INT8 weight conversion.

Weights use a scale per output channel. Activations use a calibration-derived
per-tensor scale, matching the layout expected by common CPU INT8 kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .executor import run_graph
from .ir import Graph


@dataclass
class TensorRange:
    minimum: float = np.inf
    maximum: float = -np.inf

    def observe(self, value: np.ndarray) -> None:
        self.minimum = min(self.minimum, float(np.min(value)))
        self.maximum = max(self.maximum, float(np.max(value)))

    @property
    def scale(self) -> float:
        magnitude = max(abs(self.minimum), abs(self.maximum))
        return max(magnitude / 127.0, float(np.finfo(np.float32).tiny))


def calibrate(graph: Graph, samples: Iterable[dict[str, np.ndarray]]) -> dict[str, TensorRange]:
    """Collect deterministic min/max ranges across representative samples."""
    ranges: dict[str, TensorRange] = {}
    count = 0
    for sample in samples:
        count += 1
        for name, value in run_graph(graph, sample).items():
            if np.issubdtype(value.dtype, np.floating):
                ranges.setdefault(name, TensorRange()).observe(value)
    if count == 0:
        raise ValueError("calibration requires at least one sample")
    return ranges


def _per_channel_quantize(weight: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    reduction_axes = tuple(index for index in range(weight.ndim) if index != axis)
    maxima = np.max(np.abs(weight.astype(np.float32)), axis=reduction_axes)
    scales = np.maximum(maxima / 127.0, np.finfo(np.float32).tiny).astype(np.float32)
    shape = [1] * weight.ndim
    shape[axis] = scales.size
    quantized = np.clip(np.rint(weight / scales.reshape(shape)), -127, 127).astype(np.int8)
    return quantized, scales


def quantize_int8_per_channel(
    graph: Graph,
    calibration: dict[str, TensorRange],
) -> Graph:
    """Quantize Conv/Gemm constant weights and annotate activation scales."""
    result = graph.clone()
    quantized_nodes: list[str] = []
    candidates: dict[int, tuple[str, int]] = {}
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(result.nodes):
        for name in node.inputs:
            consumers.setdefault(name, []).append(index)
        if node.op_type not in ("Conv", "Gemm", "MatMul") or len(node.inputs) < 2:
            continue
        weight_name = node.inputs[1]
        if weight_name not in result.initializers:
            continue
        activation_range = calibration.get(node.inputs[0])
        if activation_range is None:
            continue
        weight_axis = 0 if node.op_type == "Conv" else (
            0 if int(node.attributes.get("transB", 0)) else 1
        )
        candidates[index] = (weight_name, weight_axis)

    quantized_weights: dict[tuple[str, int], tuple[str, np.ndarray]] = {}
    for index, node in enumerate(result.nodes):
        if index not in candidates:
            continue
        weight_name, weight_axis = candidates[index]
        key = (weight_name, weight_axis)
        if key not in quantized_weights:
            quantized, scales = _per_channel_quantize(graph.initializers[weight_name], weight_axis)
            exclusively_quantized = all(candidates.get(consumer) == key
                                        for consumer in consumers[weight_name])
            if exclusively_quantized and weight_name not in result.outputs:
                destination = weight_name
            else:
                destination = f"{weight_name}__leaf_int8_axis{weight_axis}"
                suffix = 1
                while destination in result.initializers:
                    destination = f"{weight_name}__leaf_int8_axis{weight_axis}_{suffix}"
                    suffix += 1
            result.initializers[destination] = quantized
            quantized_weights[key] = (destination, scales)
        destination, scales = quantized_weights[key]
        node.inputs[1] = destination
        node.attributes["quantization"] = {
            "scheme": "symmetric_int8",
            "input": {"scale": activation_range.scale, "zero_point": 0},
            "weight": {"scale": scales, "zero_point": 0, "axis": weight_axis},
        }
        quantized_nodes.append(node.name)

    referenced = set(result.outputs)
    referenced.update(name for node in result.nodes for name in node.inputs)
    for name in tuple(result.initializers):
        if name not in referenced:
            del result.initializers[name]

    result.metadata.setdefault("passes", {})["int8_quantization"] = {
        "scheme": "per_tensor_activations_per_output_channel_weights",
        "nodes": quantized_nodes,
        "calibrated_tensors": len(calibration),
    }
    return result
