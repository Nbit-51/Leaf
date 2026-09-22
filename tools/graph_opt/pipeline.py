"""Composable optimization pipeline used by tests and export tooling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .constant_folding import fold_constants
from .fusion import run_fusion_passes
from .ir import Graph
from .memory_planner import MemoryPlan, plan_memory
from .quantization import TensorRange, calibrate, quantize_int8_per_channel
from .transformer import run_transformer_rewrites


@dataclass
class OptimizationResult:
    graph: Graph
    calibration: dict[str, TensorRange]
    memory_plan: MemoryPlan


def optimize_graph(
    graph: Graph,
    calibration_samples: Iterable[dict[str, np.ndarray]],
    memory_plan_path: str | Path | None = None,
) -> OptimizationResult:
    optimized = fold_constants(graph)
    optimized = run_fusion_passes(optimized)
    optimized = run_transformer_rewrites(optimized)
    ranges = calibrate(optimized, calibration_samples)
    optimized = quantize_int8_per_channel(optimized, ranges)
    plan = plan_memory(optimized)
    if memory_plan_path is not None:
        plan.export_json(memory_plan_path)
    return OptimizationResult(optimized, ranges, plan)
