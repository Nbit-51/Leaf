"""Small offline calibration/evaluation sets used by the benchmark harness."""

from __future__ import annotations

import numpy as np


def vision_samples(count: int, seed: int = 100) -> list[dict[str, np.ndarray]]:
    """CIFAR-shaped samples with fixed channel statistics and no download."""
    rng = np.random.default_rng(seed)
    samples = []
    mean = np.asarray([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
    for _ in range(count):
        pixels = rng.uniform(0.0, 1.0, (1, 3, 16, 16)).astype(np.float32)
        samples.append({"x": (pixels - mean) / std})
    return samples


def transformer_samples(count: int, seed: int = 200) -> list[dict[str, np.ndarray]]:
    """Fixed hidden-state batches representing eight tokens with width 16."""
    rng = np.random.default_rng(seed)
    return [{"x": rng.normal(0.0, 0.7, (1, 8, 16)).astype(np.float32)} for _ in range(count)]
