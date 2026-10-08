"""Qualify the established Windows FP32 stack as a release default.

Predeclared release criteria: exact equivalence to the previously validated
stack, frozen trained quality/cache/generation parity, ten alternating AB/BA
pairs, >=8/10 prefill pairs improving by >=2%, and a paired-bootstrap 95% upper
bound <=0.98 for prefill and <=1.02 for decode. Report all existing spread
checks separately. This does not change automatic precision-selection gates.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from leaf.affinity import pin_cpu
from leaf.power import keep_awake
from tools.benchmark_decoder_comparison import _quality_gate, validate_timing_request
from tools.decoder_validation import benchmark_stability, quality, run_native
from tools.validate_decoder import digest, utc_now, write_record


def paired_gate(pairs, quality_passed):
    if len(pairs) != 10:
        raise ValueError("Release qualification requires exactly ten complete pairs")
    rng = np.random.default_rng(707)
    indices = rng.integers(0, 10, size=(20000, 10))
    phases = {}
    for phase, limit in (("prefill", 0.98), ("decode", 1.02)):
        ratios = np.array([pair["after"][phase + "_p50_ms"] / pair["before"][phase + "_p50_ms"] for pair in pairs])
        if not np.isfinite(ratios).all() or np.any(ratios <= 0):
            raise ValueError("Invalid timing ratio")
        interval = np.percentile(np.median(ratios[indices], axis=1), [2.5, 97.5])
        phases[phase] = dict(paired_ratios=ratios.tolist(), median_paired_ratio=float(np.median(ratios)),
            paired_bootstrap_95_percent_interval=interval.tolist(), maximum_upper_bound=limit,
            upper_bound_passed=bool(interval[1] <= limit), pairs_improving_at_least_two_percent=int(np.sum(ratios <= .98)))
    passed = bool(quality_passed and phases["prefill"]["pairs_improving_at_least_two_percent"] >= 8 and
                  all(phase["upper_bound_passed"] for phase in phases.values()))
    return dict(passed=passed, quality_passed=quality_passed, **phases,
        bootstrap_seed=707, bootstrap_resamples=20000, automatic_precision_selection_authorized=False,
        limitation="One host/workload; pair bootstrap does not establish cross-device performance.")


@contextmanager
def legacy_stack():
    os.environ["LEAF_EXPERIMENTAL_FLOAT_TILES"] = "1"
    try:
        yield
    finally:
        os.environ.pop("LEAF_EXPERIMENTAL_FLOAT_TILES", None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "stack", "workdir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=2)
    args = parser.parse_args()
    if os.name != "nt" or args.output.exists():
        parser.error("Require Windows and a fresh output")
    if any(value for key, value in os.environ.items() if key.startswith("LEAF_EXPERIMENTAL_") or
           key in ("LEAF_DISABLE_VNNI", "LEAF_DECODER_PROFILE")):
        parser.error("Clear inherited native experimental/profiling policies")
    pin_cpu(args.cpu)
    paths = [args.before, args.after, args.stack, Path(__file__), args.workdir / "decoder-32.leaf",
             args.workdir / "tokens.json", args.workdir / "reference.npy", args.workdir / "pytorch.json",
             args.workdir / "decoder-32.provenance.json", ROOT / "engine/src/decoder.cpp",
             ROOT / "engine/include/leaf/runtime/decoder_config.h"]
    hashes = {str(path): digest(path) for path in paths}
    frozen = json.loads((args.workdir / "pytorch.json").read_text())
    provenance = json.loads((args.workdir / "decoder-32.provenance.json").read_text())
    artifact = args.workdir / "decoder-32.leaf"
    if (provenance["artifact_sha256"] != digest(artifact) or provenance["request"]["weight_bits"] != 32 or
            any(provenance["request"][key] != frozen[key] for key in ("model_config_sha256", "source_weight_sha256")) or
            any(frozen["quality_cache_sha256"][name] != digest(args.workdir / name) for name in ("tokens.json", "reference.npy"))):
        raise ValueError("Frozen trained artifact/reference provenance differs")
    data = json.loads((args.workdir / "tokens.json").read_text())
    reference = np.load(args.workdir / "reference.npy", mmap_mode="r", allow_pickle=False)
    record = dict(format="leaf-fp32-release-v1", measured_at_utc=utc_now(), hashes=hashes, cpu=args.cpu,
                  threads=1, runs_per_pass=31, warmup_per_pass=10, criteria=__doc__, quality={}, pairs=[], complete=False)
    write_record(args.output, record)
    with keep_awake():
        for threads in (1, 2):
            actual, execution = run_native(args.after, artifact, data["sequences"], threads=threads)
            with legacy_stack():
                expected, _ = run_native(args.stack, artifact, data["sequences"], threads=threads)
            exact = bool(np.array_equal(actual, expected))
            measured = quality(actual.reshape(reference.shape), reference, data["sequences"])
            close = bool(np.allclose(actual.reshape(reference.shape), reference, atol=2e-3, rtol=2e-3))
            cached, _ = run_native(args.after, artifact, [data["sequences"][0]], mode="chunked", chunk=16, threads=threads)
            cache_parity = bool(np.allclose(cached, actual[:cached.size], atol=2e-3, rtol=2e-3))
            _, generation = run_native(args.after, artifact, [data["generation_ids"]], mode="generate",
                                       generate=len(data["generated_tokens"]), threads=threads)
            generated = generation["generated_tokens"] == data["generated_tokens"]
            passed = bool(exact and close and cache_parity and generated and _quality_gate(measured, 32) and
                          execution.get("optimized_fp32_active") is True)
            record["quality"][str(threads)] = dict(passed=passed, measured=measured, stack_logits_bit_exact=exact,
                fp32_allclose=close, chunked_matches_full=cache_parity, generation_matches_pytorch=generated,
                optimized_fp32_active=execution.get("optimized_fp32_active"))
            del actual, expected, cached
            write_record(args.output, record)
            if not passed:
                raise AssertionError("Release quality failed")
        for pair_index in range(10):
            pair = {"order": ["before", "after"] if pair_index % 2 == 0 else ["after", "before"]}
            record["pairs"].append(pair)
            for stage in pair["order"]:
                binary = args.before if stage == "before" else args.after
                _, metrics = run_native(binary, artifact, [data["benchmark_ids"]], mode="bench", threads=1,
                                        runs=31, warmup=10, windows_above_normal=True)
                validate_timing_request(metrics, runs=31, threads=1, activation_bits=32)
                if metrics.get("optimized_fp32_active") is not (stage == "after"):
                    raise AssertionError("Unexpected default dispatch")
                metrics["stability"] = benchmark_stability(metrics)
                pair[stage] = metrics
                write_record(args.output, record)
                print(pair_index + 1, stage, metrics["prefill_p50_ms"], metrics["decode_p50_ms"],
                      metrics["stability"]["passed"], flush=True)
    if any(digest(Path(path)) != value for path, value in hashes.items()):
        raise ValueError("Release inputs changed during measurement")
    record["gate"] = paired_gate(record["pairs"], all(item["passed"] for item in record["quality"].values()))
    record["all_passes_stable"] = all(pair[stage]["stability"]["passed"] for pair in record["pairs"] for stage in ("before", "after"))
    record["complete"] = True
    write_record(args.output, record)
    print(json.dumps(record["gate"]), flush=True)


if __name__ == "__main__":
    main()
