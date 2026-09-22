"""Run correctness and latency checks against PyTorch CPU on fixed datasets."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from benchmark.datasets import transformer_samples, vision_samples
from benchmark.workloads import cnn_workload, transformer_workload
from tools.graph_opt.executor import run_graph
from tools.graph_opt.pipeline import optimize_graph


def _median_latency(function, samples, repeats=5):
    for sample in samples[:2]:
        function(sample)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for sample in samples:
            function(sample)
        timings.append((time.perf_counter_ns() - start) / 1e6 / len(samples))
    return statistics.median(timings)


def _run_workload(name, graph, torch_reference, calibration, evaluation, plan_path):
    optimized = optimize_graph(graph, calibration, plan_path)

    def pytorch_call(sample):
        with torch.inference_mode():
            return torch_reference(torch.from_numpy(sample["x"]))

    def leaf_fp32_call(sample):
        return run_graph(graph, sample)["y"]

    def leaf_int8_call(sample):
        return run_graph(optimized.graph, sample)["y"]

    fp32_errors, int8_errors = [], []
    for sample in evaluation:
        expected = pytorch_call(sample).numpy()
        fp32_errors.append(float(np.max(np.abs(leaf_fp32_call(sample) - expected))))
        int8_errors.append(float(np.max(np.abs(leaf_int8_call(sample) - expected))))

    return {
        "name": name,
        "dataset_samples": len(evaluation),
        "calibration_samples": len(calibration),
        "correctness": {
            "leaf_fp32_vs_pytorch_max_abs": max(fp32_errors),
            "leaf_int8_vs_pytorch_max_abs": max(int8_errors),
        },
        "latency_ms_per_sample": {
            "pytorch_cpu": _median_latency(pytorch_call, evaluation),
            "leaf_numpy_reference_fp32": _median_latency(leaf_fp32_call, evaluation),
            "leaf_numpy_quantized_simulation": _median_latency(leaf_int8_call, evaluation),
        },
        "optimized_nodes": len(optimized.graph.nodes),
        "memory_plan": optimized.memory_plan.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/latest.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan_directory = args.output.parent / "memory_plans"
    cnn_graph, cnn_torch = cnn_workload()
    transformer_graph, transformer_torch = transformer_workload()
    payload = {
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__, "torch_threads": 1,
        },
        "note": "NumPy timings are reference-oracle measurements; production latency is measured by leaf_bench.",
        "workloads": [
            _run_workload("cnn", cnn_graph, cnn_torch, vision_samples(32), vision_samples(6, 101),
                          plan_directory / "cnn.json"),
            _run_workload("transformer_ffn", transformer_graph, transformer_torch,
                          transformer_samples(32), transformer_samples(6, 201),
                          plan_directory / "transformer_ffn.json"),
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
