"""Serial balanced AB/BA whole-model timing of two default-policy Windows builds.

Predeclared incremental gate (unchanged policy): pooled-median prefill ratio
<= 0.98, pooled-median decode ratio <= 1.02, every pass and both pooled sample
sets meet the existing stability checks. Also reports median paired ratios.
Both builds must report optimized_fp32_active. Every raw sample is retained.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.affinity import pin_cpu  # noqa: E402
from leaf.power import keep_awake  # noqa: E402
from tools.benchmark_decoder_comparison import validate_timing_request  # noqa: E402
from tools.decoder_validation import benchmark_stability, latency_stability, run_native  # noqa: E402
from tools.validate_decoder import digest, utc_now, write_record  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "workdir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if os.name != "nt" or args.output.exists():
        parser.error("Require Windows and a fresh output")
    if args.pairs < 6 or args.pairs % 2 or args.runs < 31 or args.warmup < 10:
        parser.error("Require an even number of >=6 pairs, >=31 samples and >=10 warmups")
    if any(v for k, v in os.environ.items() if k.startswith("LEAF_EXPERIMENTAL_") or
           k in ("LEAF_DISABLE_VNNI", "LEAF_DECODER_PROFILE")):
        parser.error("Clear inherited native experimental/profiling policies")
    pin_cpu(args.cpu)
    artifact = args.workdir / "decoder-32.leaf"
    paths = [args.before, args.after, artifact, args.workdir / "tokens.json", Path(__file__),
             ROOT / "tools/decoder_validation.py", ROOT / "tools/benchmark_decoder_comparison.py"]
    hashes = {str(p): digest(p) for p in paths}
    data = json.loads((args.workdir / "tokens.json").read_text())
    record = dict(format="leaf-default-abba-v1", measured_at_utc=utc_now(), hashes=hashes, cpu=args.cpu,
                  threads=1, runs_per_pass=args.runs, warmup_per_pass=args.warmup,
                  windows_above_normal=True, criteria=__doc__, pairs=[], complete=False)
    write_record(args.output, record)
    with keep_awake():
        for index in range(args.pairs):
            pair = {"order": ["before", "after"] if index % 2 == 0 else ["after", "before"]}
            record["pairs"].append(pair)
            for stage in pair["order"]:
                binary = args.before if stage == "before" else args.after
                _, metrics = run_native(binary, artifact, [data["benchmark_ids"]], mode="bench", threads=1,
                                        runs=args.runs, warmup=args.warmup, windows_above_normal=True)
                validate_timing_request(metrics, runs=args.runs, threads=1, activation_bits=32)
                if metrics.get("optimized_fp32_active") is not True:
                    raise AssertionError("Both builds must use the shipped default FP32 dispatch")
                metrics["stability"] = benchmark_stability(metrics)
                pair[stage] = metrics
                write_record(args.output, record)
                print(index + 1, stage, metrics["prefill_p50_ms"], metrics["decode_p50_ms"],
                      metrics["stability"]["passed"], flush=True)
    if any(digest(Path(p)) != v for p, v in hashes.items()):
        raise ValueError("Inputs changed during measurement")
    summary = {}
    for phase, limit in (("prefill", 0.98), ("decode", 1.02)):
        pooled = {s: [v for p in record["pairs"] for v in p[s][phase + "_samples_ms"]] for s in ("before", "after")}
        paired = [p["after"][phase + "_p50_ms"] / p["before"][phase + "_p50_ms"] for p in record["pairs"]]
        medians = {s: statistics.median(v) for s, v in pooled.items()}
        stability = {s: latency_stability(v) for s, v in pooled.items()}
        ratio = medians["after"] / medians["before"]
        summary[phase] = dict(pooled_median_ms=medians, pooled_ratio=ratio, limit=limit,
                              median_paired_ratio=statistics.median(paired), paired_ratios=paired,
                              pooled_stability=stability, ratio_passed=ratio <= limit,
                              pooled_stable=all(x["passed"] for x in stability.values()))
    every_pass_stable = all(p[s]["stability"]["passed"] for p in record["pairs"] for s in ("before", "after"))
    summary["every_pass_stable"] = every_pass_stable
    summary["passed"] = bool(every_pass_stable and all(summary[ph]["ratio_passed"] and summary[ph]["pooled_stable"]
                                                        for ph in ("prefill", "decode")))
    record["gate"] = summary
    record["complete"] = True
    write_record(args.output, record)
    print(json.dumps({ph: {k: summary[ph][k] for k in ("pooled_median_ms", "pooled_ratio", "median_paired_ratio",
                                                        "ratio_passed", "pooled_stable")}
                      for ph in ("prefill", "decode")}, indent=1))
    print("every_pass_stable", every_pass_stable, "PASSED" if summary["passed"] else "NOT PASSED")


if __name__ == "__main__":
    main()
