"""Compare embedded v3 arena execution with the v2 buffer-pool baseline."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
import platform
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.graph_opt.export_binary import export_graph
from tools.graph_opt.ir import Graph
from tools.graph_opt.memory_planner import plan_memory
from tools.verify_cpp_runtime import export_resnet18


def benchmark(executable: Path, artifact: Path, input_path: Path) -> dict:
    result = subprocess.run([str(executable), str(artifact), str(input_path),
                             "1,3,32,32", "5", "20"],
                            check=True, capture_output=True, text=True)
    median = re.search(r"p50=([0-9.]+)", result.stdout)
    rss = re.search(r"peak process RSS bytes: (\d+)", result.stdout)
    if median is None or rss is None:
        raise RuntimeError("native benchmark did not report p50 and peak RSS")
    return {"p50_ms": float(median.group(1)),
            "peak_process_rss_bytes": int(rss.group(1))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf-infer", required=True, type=Path)
    parser.add_argument("--leaf-bench", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enforce-no-slowdown", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="leaf_memory_plan_") as temporary:
        directory = Path(temporary)
        baseline = directory / "resnet18.v2.leaf"
        planned = directory / "resnet18.v3.leaf"
        input_path = directory / "input.bin"
        expected_path = directory / "expected.bin"
        export_resnet18(baseline, input_path, expected_path)
        graph = Graph.from_onnx(str(baseline.with_suffix(".onnx")))
        plan = plan_memory(graph)
        export_graph(graph, str(planned), memory_plan=plan)

        malformed = deepcopy(plan)
        corrupted = False
        for index, target in enumerate(malformed.allocations):
            for other in malformed.allocations[:index]:
                live_overlap = (target.first_node <= other.last_node and
                                other.first_node <= target.last_node)
                if (live_overlap and target.offset != other.offset and
                        other.offset + target.size <= malformed.arena_size):
                    malformed.allocations[index] = replace(target, offset=other.offset)
                    corrupted = True
                    break
            if corrupted:
                break
        if not corrupted:
            raise AssertionError("test graph has no overlapping-live plan candidate")
        malformed_artifact = directory / "resnet18.invalid-plan.leaf"
        export_graph(graph, str(malformed_artifact), memory_plan=malformed)
        rejected = subprocess.run(
            [str(args.leaf_infer.resolve()), str(malformed_artifact),
             str(input_path), "1,3,32,32", str(directory / "invalid-output.bin")],
            capture_output=True, text=True)
        if rejected.returncode == 0 or "overlapping live arena allocations" not in rejected.stderr:
            raise AssertionError("native parser did not reject overlapping live allocations")

        expected = np.fromfile(expected_path, dtype=np.float32)
        measurements = {"buffer_pool": {}, "arena_plan": {}}
        for label, artifact in (("buffer_pool", baseline), ("arena_plan", planned)):
            actual_path = directory / f"{label}.output.bin"
            subprocess.run([str(args.leaf_infer.resolve()), str(artifact),
                            str(input_path), "1,3,32,32", str(actual_path)],
                           check=True, capture_output=True, text=True)
            actual = np.fromfile(actual_path, dtype=np.float32)
            max_abs = float(np.max(np.abs(actual - expected)))
            if not np.allclose(actual, expected, rtol=1e-3, atol=1e-3):
                raise AssertionError(f"{label} differs from PyTorch: {max_abs}")
            measurements[label]["max_abs_vs_pytorch"] = max_abs

        timings = {"buffer_pool": [], "arena_plan": []}
        rss_values = {"buffer_pool": [], "arena_plan": []}
        paths = {"buffer_pool": baseline, "arena_plan": planned}
        for label in ("buffer_pool", "arena_plan", "arena_plan", "buffer_pool") * 2:
            sample = benchmark(args.leaf_bench.resolve(), paths[label], input_path)
            timings[label].append(sample["p50_ms"])
            rss_values[label].append(sample["peak_process_rss_bytes"])
        for label in measurements:
            measurements[label]["p50_ms"] = statistics.median(timings[label])
            measurements[label]["p50_samples_ms"] = timings[label]
            measurements[label]["peak_process_rss_bytes"] = statistics.median(rss_values[label])
            measurements[label]["peak_rss_samples_bytes"] = rss_values[label]

        no_slowdown = (measurements["arena_plan"]["p50_ms"] <=
                       measurements["buffer_pool"]["p50_ms"] * 1.02)

        result = {
            "benchmark": "leaf-memory-plan-resnet18",
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.platform(),
            "model": "seeded ResNet-18, input 1x3x32x32",
            "warmup": 5,
            "runs": 20,
            "plan": {"alignment": plan.alignment, "arena_size": plan.arena_size,
                     "naive_size": plan.naive_size,
                     "allocations": len(plan.allocations),
                     "unplanned_tensors": plan.unplanned_tensors},
            "measurements": measurements,
            "arena_no_slowdown_with_2_percent_tolerance": no_slowdown,
        }
        print(json.dumps(result, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if args.enforce_no_slowdown and not no_slowdown:
            raise AssertionError("embedded arena plan exceeded buffer-pool latency by over 2%")


if __name__ == "__main__":
    main()
