"""Validate MLP worker-grain changes at matched Windows core budgets.

The candidate changes only row scheduling, so exact native parity is required.
Each core budget has its own serial ABBA and unchanged 2%/1.25 speed/spread gates.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil

from leaf.power import keep_awake
from tools.benchmark_core_scaling import validate_cores
from tools.benchmark_decoder_comparison import (native_experiment_gate, stage_summary,
                                               validate_timing_request, _quality_gate)
from tools.decoder_validation import quality, run_native
from tools.diagnose_windows_benchmark import cpu_sets
from tools.validate_decoder import digest, utc_now, write_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "workdir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cores", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if sys.platform != "win32" or args.output.exists() or args.runs < 5 or args.warmup < 1:
        parser.error("Require Windows, fresh output, >=5 samples and >=1 warmup")
    if (not args.counts or len(set(args.counts)) != len(args.counts) or
            any(count < 1 or count > len(args.cores) for count in args.counts)):
        parser.error("Core counts must be distinct and fit selected CPUs")
    topology = validate_cores(args.cores, cpu_sets())
    if any(value for key, value in os.environ.items() if key.startswith("LEAF_EXPERIMENTAL_") or
           key in ("LEAF_DISABLE_VNNI", "LEAF_DECODER_PROFILE")):
        parser.error("Clear inherited native experimental/profiling policies")
    artifact = args.workdir / "decoder-32.leaf"
    paths = [args.before, args.after, artifact, args.workdir / "tokens.json", args.workdir / "reference.npy",
             args.workdir / "pytorch.json", args.workdir / "decoder-32.provenance.json",
             Path(__file__), ROOT / "engine/src/decoder.cpp"]
    hashes = {str(path): digest(path) for path in paths}
    frozen = json.loads((args.workdir / "pytorch.json").read_text())
    provenance = json.loads((args.workdir / "decoder-32.provenance.json").read_text())
    if any(frozen["quality_cache_sha256"][name] != digest(args.workdir / name)
           for name in ("tokens.json", "reference.npy")):
        raise ValueError("Frozen reference/token hashes differ")
    if (provenance["artifact_sha256"] != digest(artifact) or provenance["request"]["weight_bits"] != 32 or
            any(provenance["request"][key] != frozen[key]
                for key in ("model_config_sha256", "source_weight_sha256"))):
        raise ValueError("Native artifact does not match frozen PyTorch weights")
    data = json.loads((args.workdir / "tokens.json").read_text())
    reference = np.load(args.workdir / "reference.npy", mmap_mode="r", allow_pickle=False)
    record = dict(format="leaf-worker-grain-v1", measured_at_utc=utc_now(), hashes=hashes,
                  topology=topology, runs=args.runs, warmup=args.warmup, cases={}, complete=False,
                  automatic_selection_authorized=False)
    write_record(args.output, record)
    process = psutil.Process()
    old_affinity = process.cpu_affinity()
    os.environ["LEAF_EXPERIMENTAL_FLOAT_TILES"] = "1"
    try:
        with keep_awake():
            for count in args.counts:
                cpus = args.cores[:count]
                process.cpu_affinity(cpus)
                if sorted(process.cpu_affinity()) != sorted(cpus):
                    raise RuntimeError("CPU affinity was not applied")
                case = dict(threads=count, affinity=cpus, passes=[])
                record["cases"][str(count)] = case
                # All held-out logits, cached chunks and generated tokens must
                # agree exactly; no tolerance is allowed for this scheduling edit.
                before, _ = run_native(args.before, artifact, data["sequences"], threads=count)
                after, _ = run_native(args.after, artifact, data["sequences"], threads=count)
                exact = np.array_equal(before, after)
                measured = quality(after.reshape(reference.shape), reference, data["sequences"])
                fp32_close = bool(np.allclose(after.reshape(reference.shape), reference, atol=2e-3, rtol=2e-3))
                del before, after
                parity = {}
                for mode, sequences, options in (
                    ("chunked", [data["sequences"][0]], {"chunk": 16}),
                    ("generate", [data["generation_ids"]], {"generate": len(data["generated_tokens"])})):
                    left, left_metrics = run_native(args.before, artifact, sequences, mode=mode, threads=count, **options)
                    right, right_metrics = run_native(args.after, artifact, sequences, mode=mode, threads=count, **options)
                    parity[mode] = bool(np.array_equal(left, right) and
                        left_metrics["generated_tokens"] == right_metrics["generated_tokens"] and
                        (mode != "generate" or right_metrics["generated_tokens"] == data["generated_tokens"]))
                passed = exact and fp32_close and _quality_gate(measured, 32) and all(parity.values())
                case["quality"] = dict(measured=measured, logits_bit_exact=exact, fp32_allclose=fp32_close,
                                       parity=parity, passed=passed)
                write_record(args.output, record)
                if not passed:
                    raise AssertionError("Worker scheduling changed model results")
                for stage in ("before", "after", "after", "before"):
                    binary = args.before if stage == "before" else args.after
                    _, metrics = run_native(binary, artifact, [data["benchmark_ids"]], mode="bench",
                        threads=count, runs=args.runs, warmup=args.warmup, windows_above_normal=True)
                    validate_timing_request(metrics, runs=args.runs, threads=count, activation_bits=32)
                    case["passes"].append(dict(stage=stage, metrics=metrics))
                    write_record(args.output, record)
                    print(count, stage, metrics["prefill_p50_ms"], metrics["decode_p50_ms"], flush=True)
                for stage in ("before", "after"):
                    case[stage] = stage_summary([item["metrics"] for item in case["passes"] if item["stage"] == stage])
                case["gate"] = native_experiment_gate(case["before"], case["after"], passed)
                write_record(args.output, record)
                print(count, json.dumps(case["gate"]), flush=True)
    finally:
        process.cpu_affinity(old_affinity)
        os.environ.pop("LEAF_EXPERIMENTAL_FLOAT_TILES", None)
    if any(digest(Path(path)) != value for path, value in hashes.items()):
        raise ValueError("Inputs changed during measurement")
    record["complete"] = True
    write_record(args.output, record)


if __name__ == "__main__":
    main()
