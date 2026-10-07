"""Run serial packing-inclusive full-K / candidate GEMM ABBA microbenchmarks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.affinity import pin_cpu
from leaf.power import keep_awake
from tools.decoder_validation import latency_stability, MAX_LATENCY_P90_P10_RATIO
from tools.validate_decoder import digest, utc_now, write_record


def summarize(passes: list[dict], candidate: str = "kblocked") -> dict:
    result = {}
    for stage in ("full_k", candidate):
        groups = [p["samples_ms"] for p in passes if p["stage"] == stage]
        checks = [latency_stability(values) for values in groups]
        samples = [value for values in groups for value in values]
        aggregate = latency_stability(samples)
        medians = [statistics.median(values) for values in groups]
        drift = max(medians) / min(medians)
        result[stage] = {"median_ms": statistics.median(samples), "pass_medians_ms": medians,
                         "pass_stability": checks, "aggregate_stability": aggregate,
                         "between_pass_ratio": drift,
                         "stable": len(groups) == 2 and all(c["passed"] for c in checks)
                         and aggregate["passed"] and drift <= MAX_LATENCY_P90_P10_RATIO}
    result[f"ratio_{candidate}_full_k"] = result[candidate]["median_ms"] / result["full_k"]["median_ms"]
    result["stable"] = all(result[stage]["stable"] for stage in ("full_k", candidate))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--runs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate", choices=("kblocked", "weight_panel", "wide_token", "unrolled", "row_reuse", "mlp_panel", "mlp_pack", "wide_row_reuse"), default="kblocked")
    parser.add_argument("--windows-above-normal", action="store_true")
    parser.add_argument("--weight-copies", type=int, default=1, help="Rotate resident replicas; samples are per-matrix means")
    args = parser.parse_args()
    if args.windows_above_normal and sys.platform != "win32":
        parser.error("Above-normal priority requires Windows")
    if not 1 <= args.weight_copies <= 32:
        parser.error("weight-copies must be 1..32")
    if args.output.exists():
        parser.error("Require a new output path; preserve previous measurements")
    if not 5 <= args.runs <= 10000 or not 0 <= args.warmup <= 10000:
        parser.error("runs must be 5..10000 and warmup 0..10000")
    paths = {"executable": args.executable, "source": ROOT / "benchmark/token_panel_benchmark.cpp",
             "kernel": ROOT / "engine/include/leaf/kernels/token_panel.h",
             "weight_panel_kernel": ROOT / "engine/include/leaf/kernels/weight_panel.h",
             "wide_token_kernel": ROOT / "engine/include/leaf/kernels/wide_token.h",
             "row_reuse_kernel": ROOT / "engine/include/leaf/kernels/row_reuse.h",
             "unrolled_kernel": ROOT / "engine/include/leaf/kernels/unrolled_token.h"}
    if args.candidate == "mlp_panel":
        paths["mlp_panel_kernel"] = ROOT / "engine/include/leaf/kernels/mlp_panel.h"
    hashes = {name: digest(path) for name, path in paths.items()}
    pin_cpu(args.cpu)
    with keep_awake():
        completed = subprocess.run([str(args.executable.resolve()), str(args.runs), str(args.warmup), args.candidate, str(args.weight_copies)],
                                   capture_output=True, text=True, check=True,
                                   creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS if args.windows_above_normal else 0)
    record = json.loads(completed.stdout)
    if record.get("candidate") != args.candidate or record.get("weight_copies") != args.weight_copies:
        raise ValueError("Benchmark binary did not execute the requested candidate")
    if hashes != {name: digest(path) for name, path in paths.items()}:
        raise ValueError("Benchmark binary/source changed during execution")
    record.update(benchmark="token-panel-shape-abba-v1", measured_at_utc=utc_now(),
                  baseline_kernel="row_reuse" if args.candidate in ("mlp_panel", "mlp_pack", "wide_row_reuse") else "full_k",
                  windows_above_normal=args.windows_above_normal,
                  platform=platform.platform(), cpu=platform.processor(), pinned_cpu=args.cpu,
                  threads=1, runs_per_pass=args.runs, warmup_per_pass=args.warmup, sha256=hashes,
                  synthetic_inputs=True, model_latency_measured=False,
                  sample_unit="milliseconds per matrix, averaged over resident weight replicas",
                  weight_replica_values="identical synthetic values at distinct addresses",
                  automatic_selection_authorized=False, maximum_stability_ratio=MAX_LATENCY_P90_P10_RATIO)
    for shape in record["shapes"]:
        shape["summary"] = summarize(shape["passes"], args.candidate)
        print(shape["m"], shape["n"], shape["k"], "ratio", shape["summary"][f"ratio_{args.candidate}_full_k"],
              "stable", shape["summary"]["stable"])
    write_record(args.output, record)


if __name__ == "__main__":
    main()
