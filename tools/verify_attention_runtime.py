"""Check native masked attention against PyTorch for prefill and decode shapes."""

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
           shape: str, output_path: Path, output_shape: tuple[int, ...]) -> np.ndarray:
    result = subprocess.run([str(executable), str(artifact), str(input_path),
                             shape, str(output_path)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return np.fromfile(output_path, dtype=np.float32).reshape(output_shape)


def _bench(executable: Path, artifact: Path, input_path: Path, shape: str) -> float:
    result = subprocess.run([str(executable), str(artifact), str(input_path),
                             shape, "10", "100"], capture_output=True, text=True, check=True)
    match = re.search(r"p50=([0-9.]+)", result.stdout)
    if match is None:
        raise RuntimeError("native benchmark did not report p50")
    return float(match.group(1))


def _case(name: str, query_tokens: int, key_tokens: int, rng: np.random.Generator,
          args: argparse.Namespace, directory: Path) -> dict:
    batch, heads, dim = 1, 8, 64
    query = rng.normal(0, 0.3, (batch, heads, query_tokens, dim)).astype(np.float32)
    key = rng.normal(0, 0.3, (batch, heads, key_tokens, dim)).astype(np.float32)
    value = rng.normal(0, 0.3, (batch, heads, key_tokens, dim)).astype(np.float32)
    mask = np.ones((1, 1, query_tokens, key_tokens), dtype=np.float32)
    if query_tokens > 1:
        mask[0, 0] = np.tril(np.ones((query_tokens, key_tokens), dtype=np.float32))
    else:
        mask[0, 0, 0, 0] = 0.0  # Exercise nontrivial decode masking.
    scale = dim ** -0.25
    graph = Graph()
    graph.inputs, graph.outputs = ["query"], ["output"]
    graph.initializers = {"key": key, "value": value, "mask": mask}
    graph.nodes = [Node("attention", "Attention", ["query", "key", "value", "mask"],
                        ["output"], attributes={"scale": scale, "mask_nonzero_is_valid": 1})]
    output_shape = (batch, query_tokens, heads, dim)
    graph.value_info = {
        "query": TensorInfo("query", query.shape, "float32"),
        "output": TensorInfo("output", output_shape, "float32"),
    }

    tq, tk, tv, tm = (torch.from_numpy(array) for array in (query, key, value, mask))

    def torch_call():
        scores = (tq * scale) @ (tk * scale).transpose(-1, -2)
        scores = scores.masked_fill(tm == 0, float("-inf"))
        probability = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        return (probability @ tv).permute(0, 2, 1, 3)

    with torch.inference_mode():
        expected = torch_call().numpy()
        for _ in range(10):
            torch_call()
        timings = []
        for _ in range(100):
            start = time.perf_counter_ns()
            torch_call()
            timings.append((time.perf_counter_ns() - start) / 1e6)

    reference = run_graph(graph, {"query": query})["output"]
    if not np.allclose(reference, expected, rtol=1e-4, atol=1e-4):
        raise AssertionError(f"Python {name} attention differs from PyTorch")

    input_path = directory / f"{name}_query.bin"
    query.tofile(input_path)
    shape = ",".join(str(value) for value in query.shape)
    candidates = {"v2": None, "v3": plan_memory(graph)}
    measurements = {}
    for label, plan in candidates.items():
        artifact = directory / f"{name}_{label}.leaf"
        export_graph(graph, str(artifact), memory_plan=plan)
        actual = _infer(args.leaf_infer.resolve(), artifact, input_path, shape,
                        directory / f"{name}_{label}.bin", output_shape)
        max_abs = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=1e-4, atol=1e-4):
            raise AssertionError(f"native {name} {label} differs from PyTorch: {max_abs}")
        measurements[label] = {"max_abs_vs_pytorch": max_abs,
                               "p50_ms": _bench(args.leaf_bench.resolve(), artifact,
                                                input_path, shape)}
    if args.portable_infer and args.portable_bench:
        artifact = directory / f"{name}_v2.leaf"
        actual = _infer(args.portable_infer.resolve(), artifact, input_path, shape,
                        directory / f"{name}_portable.bin", output_shape)
        max_abs = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=1e-4, atol=1e-4):
            raise AssertionError(f"portable {name} attention differs from PyTorch")
        measurements["portable_v2"] = {
            "max_abs_vs_pytorch": max_abs,
            "p50_ms": _bench(args.portable_bench.resolve(), artifact, input_path, shape),
        }
    return {"query_shape": list(query.shape), "key_shape": list(key.shape),
            "pytorch_cpu_p50_ms": statistics.median(timings), "measurements": measurements}


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
        parser.error("pass both portable executable paths")
    if args.enforce_no_slowdown and not args.portable_bench:
        parser.error("--enforce-no-slowdown requires both portable executable paths")
    torch.set_num_threads(1)
    rng = np.random.default_rng(411)
    with tempfile.TemporaryDirectory(prefix="leaf_attention_") as temporary:
        directory = Path(temporary)
        cases = {
            "prefill": _case("prefill", 32, 32, rng, args, directory),
            "decode": _case("decode", 1, 64, rng, args, directory),
        }
    no_slowdown = all(
        case["measurements"]["v2"]["p50_ms"] <=
        case["measurements"]["portable_v2"]["p50_ms"] * 1.02
        for case in cases.values()
    ) if args.portable_bench else None
    result = {"benchmark": "leaf-native-attention", "measured_at_utc":
              datetime.now(timezone.utc).isoformat(), "host": platform.platform(),
              "warmup": 10, "runs": 100, "cases": cases,
              "avx2_no_slowdown_with_2_percent_tolerance": no_slowdown}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.enforce_no_slowdown and no_slowdown is False:
        raise AssertionError("optimized native attention exceeded portable latency by over 2%")


if __name__ == "__main__":
    main()
