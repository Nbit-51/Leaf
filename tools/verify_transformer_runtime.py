"""Native Transformer-op parity and whole-graph CPU latency checks."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.graph_opt.executor import run_graph
from tools.graph_opt.export_binary import export_graph
from tools.graph_opt.ir import Graph, Node, TensorInfo
from tools.graph_opt.memory_planner import plan_memory


def _infer(executable: Path, artifact: Path, input_path: Path,
           shape: tuple[int, ...], output_path: Path) -> np.ndarray:
    result = subprocess.run(
        [str(executable), str(artifact), str(input_path),
         ",".join(map(str, shape)), str(output_path)],
        capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return np.fromfile(output_path, dtype=np.float32).reshape(shape)


def _bench(executable: Path, artifact: Path, input_path: Path,
           shape: tuple[int, ...]) -> float:
    result = subprocess.run(
        [str(executable), str(artifact), str(input_path),
         ",".join(map(str, shape)), "10", "100"],
        capture_output=True, text=True, check=True)
    match = re.search(r"p50=([0-9.]+)", result.stdout)
    if match is None:
        raise RuntimeError("native graph benchmark did not report p50")
    return float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf-infer", type=Path, required=True)
    parser.add_argument("--leaf-bench", type=Path, required=True)
    parser.add_argument("--portable-infer", type=Path)
    parser.add_argument("--portable-bench", type=Path)
    parser.add_argument("--enforce-no-slowdown", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.portable_infer) != bool(args.portable_bench):
        parser.error("portable infer and bench executables must be supplied together")
    if args.enforce_no_slowdown and not args.portable_bench:
        parser.error("--enforce-no-slowdown requires both portable executable paths")
    torch.set_num_threads(1)
    rng = np.random.default_rng(914)
    shape = (1, 64, 896)
    value = rng.normal(0, 0.8, shape).astype(np.float32)
    weight = rng.normal(1, 0.1, (shape[-1],)).astype(np.float32)
    epsilon = 1e-6
    graph = Graph()
    graph.inputs, graph.outputs = ["x"], ["y"]
    graph.initializers = {"weight": weight}
    graph.nodes = [Node("rmsnorm", "RMSNorm", ["x", "weight"], ["y"],
                        {"eps": epsilon})]
    graph.value_info = {
        name: TensorInfo(name, shape, "float32") for name in ("x", "y")
    }
    with torch.inference_mode():
        torch_input = torch.from_numpy(value)
        torch_weight = torch.from_numpy(weight)

        def torch_call():
            return torch_input * torch.rsqrt(torch_input.square().mean(dim=-1, keepdim=True)
                                             + epsilon) * torch_weight

        expected = torch_call().numpy()
        for _ in range(10):
            torch_call()
        torch_timings = []
        for _ in range(100):
            start = time.perf_counter_ns()
            torch_call()
            torch_timings.append((time.perf_counter_ns() - start) / 1e6)

    python_reference = run_graph(graph, {"x": value})["y"]
    if not np.allclose(python_reference, expected, rtol=2e-5, atol=2e-5):
        raise AssertionError("Python RMSNorm reference differs from PyTorch")

    with tempfile.TemporaryDirectory(prefix="leaf_transformer_op_") as temporary:
        directory = Path(temporary)
        v2 = directory / "rmsnorm.v2.leaf"
        v3 = directory / "rmsnorm.v3.leaf"
        input_path = directory / "input.bin"
        value.tofile(input_path)
        export_graph(graph, str(v2))
        export_graph(graph, str(v3), memory_plan=plan_memory(graph))
        measurements = {}
        for label, artifact, infer, bench in (
            ("avx2_v2", v2, args.leaf_infer, args.leaf_bench),
            ("avx2_v3", v3, args.leaf_infer, args.leaf_bench),
        ):
            actual = _infer(infer.resolve(), artifact, input_path, shape,
                            directory / f"{label}.bin")
            max_abs = float(np.max(np.abs(actual - expected)))
            if not np.allclose(actual, expected, rtol=2e-5, atol=2e-5):
                raise AssertionError(f"{label} differs from PyTorch: {max_abs}")
            measurements[label] = {"max_abs_vs_pytorch": max_abs,
                                   "p50_ms": _bench(bench.resolve(), artifact, input_path, shape)}
        if args.portable_infer:
            actual = _infer(args.portable_infer.resolve(), v2, input_path, shape,
                            directory / "portable.bin")
            max_abs = float(np.max(np.abs(actual - expected)))
            if not np.allclose(actual, expected, rtol=2e-5, atol=2e-5):
                raise AssertionError(f"portable RMSNorm differs from PyTorch: {max_abs}")
            measurements["portable_v2"] = {
                "max_abs_vs_pytorch": max_abs,
                "p50_ms": _bench(args.portable_bench.resolve(), v2, input_path, shape),
            }
        no_slowdown = (measurements["avx2_v2"]["p50_ms"] <=
                       measurements["portable_v2"]["p50_ms"] * 1.02) if args.portable_bench else None
        result = {
            "benchmark": "leaf-native-rmsnorm",
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.platform(),
            "shape": list(shape),
            "warmup": 10,
            "runs": 100,
            "pytorch_cpu_p50_ms": statistics.median(torch_timings),
            "measurements": measurements,
            "avx2_no_slowdown_with_2_percent_tolerance": no_slowdown,
        }
        print(json.dumps(result, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if args.enforce_no_slowdown and not no_slowdown:
            raise AssertionError("AVX2 RMSNorm exceeded portable latency by over 2%")


if __name__ == "__main__":
    main()
