"""End-to-end checks for typed INT8 .leaf artifacts and native graph execution."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "graph_opt"))

from benchmark.workloads import cnn_workload, transformer_workload
from export_binary import export_graph
from tools.graph_opt.executor import run_graph
from tools.graph_opt.pipeline import optimize_graph
from tools.graph_opt.constant_folding import fold_constants
from tools.graph_opt.fusion import run_fusion_passes
from tools.graph_opt.transformer import run_transformer_rewrites
from tools.graph_opt.read_binary import read_graph


def verify_case(name: str, graph, torch_reference, shape: tuple[int, ...],
                executable: Path, directory: Path, seed: int,
                benchmark_executable: Path | None) -> dict:
    rng = np.random.default_rng(seed)
    calibration = [{"x": rng.normal(0, 1, shape).astype(np.float32)} for _ in range(8)]
    sample = calibration[0]["x"]
    optimization = optimize_graph(graph, calibration)
    optimized = optimization.graph
    artifact = directory / f"{name}.leaf"
    input_path = directory / f"{name}.input.bin"
    output_path = directory / f"{name}.output.bin"
    export_graph(optimized, str(artifact))
    decoded = read_graph(str(artifact))
    assert decoded["version"] == 2
    assert any(value.dtype == np.int8 for value in decoded["initializers"].values())
    assert any("quantization" in node["attributes"] for node in decoded["nodes"])
    sample.tofile(input_path)
    process = subprocess.run([str(executable), str(artifact), str(input_path),
                              ",".join(map(str, shape)), str(output_path)],
                             capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"{name} native inference failed: {process.stderr.strip()}")

    expected = run_graph(optimized, {"x": sample})["y"]
    actual = np.fromfile(output_path, dtype=np.float32).reshape(expected.shape)
    planned_artifact = directory / f"{name}.planned.leaf"
    planned_output_path = directory / f"{name}.planned.output.bin"
    export_graph(optimized, str(planned_artifact), memory_plan=optimization.memory_plan)
    planned_process = subprocess.run(
        [str(executable), str(planned_artifact), str(input_path),
         ",".join(map(str, shape)), str(planned_output_path)],
        capture_output=True, text=True)
    if planned_process.returncode:
        raise RuntimeError(f"{name} planned INT8 inference failed: {planned_process.stderr.strip()}")
    planned_actual = np.fromfile(planned_output_path, dtype=np.float32).reshape(expected.shape)
    if not np.array_equal(planned_actual, actual):
        raise AssertionError(f"{name} planned and unplanned INT8 outputs differ")
    absolute_error = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, rtol=2e-3, atol=2e-3):
        raise AssertionError(f"{name} native INT8/reference mismatch: max abs {absolute_error:.6g}")
    with torch.no_grad():
        baseline = torch_reference(torch.from_numpy(sample)).numpy()
    relative_rmse = float(np.sqrt(np.mean((actual - baseline) ** 2)) /
                          max(np.sqrt(np.mean(baseline ** 2)), 1e-12))
    if relative_rmse > 0.08:
        raise AssertionError(f"{name} INT8/PyTorch quality regression: {relative_rmse:.4%}")
    print(f"PASS {name}: native/reference max abs {absolute_error:.6g}; "
          f"PyTorch relative RMSE {relative_rmse:.3%}")
    record = {"shape": list(shape), "native_reference_max_abs": absolute_error,
              "pytorch_relative_rmse": relative_rmse}
    if benchmark_executable is not None:
        fp32 = run_transformer_rewrites(run_fusion_passes(fold_constants(graph)))
        fp32_artifact = directory / f"{name}.fp32.leaf"
        export_graph(fp32, str(fp32_artifact))
        for label, path in (("FP32", fp32_artifact), ("INT8", artifact)):
            result = subprocess.run([str(benchmark_executable), str(path), str(input_path),
                                     ",".join(map(str, shape)), "10", "50"],
                                    check=True, capture_output=True, text=True)
            latency = next(line.strip() for line in result.stdout.splitlines()
                           if "latency ms:" in line)
            median = re.search(r"p50=([0-9.]+)", latency)
            rss = re.search(r"peak process RSS bytes: (\d+)", result.stdout)
            record[label.lower()] = {
                "p50_ms": float(median.group(1)) if median else None,
                "peak_process_rss_bytes": int(rss.group(1)) if rss else None,
            }
            print(f"  {label}: {latency}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf-infer", type=Path, required=True)
    parser.add_argument("--leaf-bench", type=Path)
    parser.add_argument("--output", type=Path,
                        help="write parity, latency, and peak-process-RSS measurements")
    args = parser.parse_args()
    executable = args.leaf_infer.resolve()
    with tempfile.TemporaryDirectory(prefix="leaf_int8_parity_") as temporary:
        directory = Path(temporary)
        benchmark_executable = args.leaf_bench.resolve() if args.leaf_bench else None
        results = {
            "cnn": verify_case("cnn", *cnn_workload(), (1, 3, 16, 16),
                               executable, directory, 701, benchmark_executable),
            "transformer_ffn": verify_case(
                "transformer_ffn", *transformer_workload(), (1, 8, 16),
                executable, directory, 702, benchmark_executable),
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({
                "benchmark": "leaf-native-int8-graph-integration",
                "measured_at_utc": datetime.now(timezone.utc).isoformat(),
                "host": platform.platform(),
                "warmup": 10 if benchmark_executable else None,
                "runs": 50 if benchmark_executable else None,
                "results": results,
            }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
