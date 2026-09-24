"""Compare native fused SwiGLU FFN with unfused Leaf and PyTorch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.graph_opt.executor import run_graph
from tools.graph_opt.export_binary import export_graph
from tools.graph_opt.fusion import fuse_swiglu_mlp
from tools.graph_opt.ir import Graph, Node, TensorInfo
from tools.graph_opt.memory_planner import plan_memory


def _infer(executable: Path, artifact: Path, input_path: Path,
           output_path: Path) -> np.ndarray:
    result = subprocess.run([str(executable), str(artifact), str(input_path),
                             "1,16,128", str(output_path)],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return np.fromfile(output_path, dtype=np.float32).reshape(1, 16, 128)


def _bench(executable: Path, artifact: Path, input_path: Path) -> float:
    result = subprocess.run([str(executable), str(artifact), str(input_path),
                             "1,16,128", "10", "100"],
                            capture_output=True, text=True, check=True)
    match = re.search(r"p50=([0-9.]+)", result.stdout)
    if match is None:
        raise RuntimeError("native benchmark did not report p50")
    return float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf-infer", type=Path, required=True)
    parser.add_argument("--leaf-bench", type=Path, required=True)
    parser.add_argument("--enforce-no-slowdown", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rng = np.random.default_rng(915)
    value = rng.normal(0, 0.8, (1, 16, 128)).astype(np.float32)
    gate_weight = rng.normal(0, 0.05, (128, 256)).astype(np.float32)
    up_weight = rng.normal(0, 0.05, (128, 256)).astype(np.float32)
    down_weight = rng.normal(0, 0.05, (256, 128)).astype(np.float32)
    graph = Graph()
    graph.inputs, graph.outputs = ["x"], ["y"]
    graph.initializers = {"gate_weight": gate_weight, "up_weight": up_weight,
                          "down_weight": down_weight}
    graph.nodes = [
        Node("gate", "MatMul", ["x", "gate_weight"], ["gate"]),
        Node("sigmoid", "Sigmoid", ["gate"], ["sigmoid"]),
        Node("silu", "Mul", ["gate", "sigmoid"], ["silu"]),
        Node("up", "MatMul", ["x", "up_weight"], ["up"]),
        Node("gated", "Mul", ["silu", "up"], ["gated"]),
        Node("down", "MatMul", ["gated", "down_weight"], ["down"]),
        Node("residual", "Add", ["down", "x"], ["y"]),
    ]
    shapes = {"x": (1, 16, 128), "y": (1, 16, 128), "down": (1, 16, 128)}
    shapes.update({name: (1, 16, 256) for name in
                   ("gate", "sigmoid", "silu", "up", "gated")})
    graph.value_info = {name: TensorInfo(name, shape, "float32")
                        for name, shape in shapes.items()}
    fused = fuse_swiglu_mlp(graph)
    if len(fused.nodes) != 1 or fused.nodes[0].op_type != "SwiGLU_MLP":
        raise AssertionError("SwiGLU fusion did not produce one native node")

    with torch.inference_mode():
        torch_input = torch.from_numpy(value)
        torch_gate = torch.from_numpy(gate_weight)
        torch_up = torch.from_numpy(up_weight)
        torch_down = torch.from_numpy(down_weight)

        def torch_call():
            return (functional.silu(torch_input @ torch_gate) *
                    (torch_input @ torch_up)) @ torch_down + torch_input

        expected = torch_call().numpy()
        for _ in range(10):
            torch_call()
        torch_timings = []
        for _ in range(100):
            start = time.perf_counter_ns()
            torch_call()
            torch_timings.append((time.perf_counter_ns() - start) / 1e6)

    for label, candidate in (("unfused", graph), ("fused", fused)):
        actual = run_graph(candidate, {"x": value})["y"]
        if not np.allclose(actual, expected, rtol=1e-4, atol=1e-4):
            raise AssertionError(f"Python {label} SwiGLU differs from PyTorch")

    with tempfile.TemporaryDirectory(prefix="leaf_swiglu_") as temporary:
        directory = Path(temporary)
        input_path = directory / "input.bin"
        value.tofile(input_path)
        artifacts = {
            "unfused_v2": (graph, None),
            "fused_v2": (fused, None),
            "fused_v3": (fused, plan_memory(fused)),
        }
        measurements = {}
        artifact_paths = {}
        for label, (candidate, plan) in artifacts.items():
            artifact = directory / f"{label}.leaf"
            export_graph(candidate, str(artifact), memory_plan=plan)
            artifact_paths[label] = artifact
            actual = _infer(args.leaf_infer.resolve(), artifact, input_path,
                            directory / f"{label}.bin")
            max_abs = float(np.max(np.abs(actual - expected)))
            if not np.allclose(actual, expected, rtol=1e-4, atol=1e-4):
                raise AssertionError(f"native {label} differs from PyTorch: {max_abs}")
            measurements[label] = {"max_abs_vs_pytorch": max_abs}
        # Alternate process order to reduce thermal/load bias between the
        # unfused and fused version-2 graphs. Version 3 is measured separately.
        samples = {label: [] for label in artifacts}
        for label in ("unfused_v2", "fused_v2", "fused_v2", "unfused_v2"):
            samples[label].append(_bench(args.leaf_bench.resolve(),
                                         artifact_paths[label], input_path))
        samples["fused_v3"].append(_bench(args.leaf_bench.resolve(),
                                          artifact_paths["fused_v3"], input_path))
        for label, values in samples.items():
            measurements[label]["p50_ms"] = statistics.median(values)
            measurements[label]["p50_samples_ms"] = values
        no_slowdown = (measurements["fused_v2"]["p50_ms"] <=
                       measurements["unfused_v2"]["p50_ms"] * 1.02)
        result = {
            "benchmark": "leaf-native-swiglu-ffn",
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.platform(),
            "shape": [1, 16, 128],
            "intermediate_width": 256,
            "warmup": 10,
            "runs": 100,
            "pytorch_cpu_p50_ms": statistics.median(torch_timings),
            "measurements": measurements,
            "fusion_no_slowdown_with_2_percent_tolerance": no_slowdown,
        }
        print(json.dumps(result, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if args.enforce_no_slowdown and not no_slowdown:
            raise AssertionError("native SwiGLU fusion exceeded unfused latency by over 2%")


if __name__ == "__main__":
    main()
