"""Measure isolated PyTorch FP32 linear layers on Leaf's GPT-2 prefill shapes.

Matched to benchmark/token_panel_benchmark.cpp: M=63 tokens, FP32 weights and
bias, one intra-op thread, rotating resident layer weights so successive calls
do not reuse a hot cache-resident matrix. Samples are milliseconds per matrix,
averaged over the replicas. This is an operator comparison only; it does not
measure whole-model latency.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.affinity import pin_cpu  # noqa: E402
from tools.decoder_validation import latency_stability  # noqa: E402
from tools.validate_decoder import utc_now, write_record  # noqa: E402

SHAPES = ((768, 768), (3072, 768), (768, 3072), (2304, 768))  # (N output rows, K)


def above_normal() -> None:
    if sys.platform != "win32":
        raise SystemExit("Above-normal priority requires Windows")
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00008000):
        raise OSError(ctypes.get_last_error(), "Unable to set Above Normal priority")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--weight-copies", type=int, default=12)
    parser.add_argument("--windows-above-normal", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Require a new output path; preserve previous measurements")
    pin_cpu(args.cpu)
    if args.windows_above_normal:
        above_normal()
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(0)
    record = {"benchmark": "torch-linear-shapes-v1", "measured_at_utc": utc_now(),
              "torch": torch.__version__, "platform": platform.platform(), "pinned_cpu": args.cpu,
              "threads": torch.get_num_threads(), "weight_copies": args.weight_copies,
              "runs_per_pass": args.runs, "warmup_per_pass": args.warmup,
              "windows_above_normal": args.windows_above_normal, "m": 63, "dtype": "float32",
              "operator": "torch.nn.functional.linear(x[1,63,K], W[N,K], b[N]) under inference_mode",
              "sample_unit": "milliseconds per matrix, averaged over resident weight replicas",
              "shapes": []}
    with torch.inference_mode():
        for rows, columns in SHAPES:
            x = torch.randn(1, 63, columns)
            weights = [torch.randn(rows, columns) / columns ** 0.5 for _ in range(args.weight_copies)]
            biases = [torch.randn(rows) for _ in range(args.weight_copies)]
            passes = []
            for _ in range(args.passes):
                samples = []
                for iteration in range(args.warmup + args.runs):
                    start = time.perf_counter_ns()
                    for w, b in zip(weights, biases):
                        y = torch.nn.functional.linear(x, w, b)
                    elapsed = (time.perf_counter_ns() - start) / 1e6 / args.weight_copies
                    float(y[0, -1, 0])
                    if iteration >= args.warmup:
                        samples.append(elapsed)
                passes.append({"samples_ms": samples, "median_ms": statistics.median(samples),
                               "stability": latency_stability(samples)})
            pooled = [v for p in passes for v in p["samples_ms"]]
            shape = {"n": rows, "k": columns, "passes": passes, "median_ms": statistics.median(pooled),
                     "aggregate_stability": latency_stability(pooled)}
            record["shapes"].append(shape)
            print(rows, columns, "median_ms", round(shape["median_ms"], 5),
                  "pass_medians", [round(p["median_ms"], 5) for p in passes])
    write_record(args.output, record)


if __name__ == "__main__":
    main()
