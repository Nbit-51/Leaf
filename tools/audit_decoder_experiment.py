"""Replay saved ABBA verdicts and independently audit paired inference error.

No latency is measured here. Historical records remain immutable. Fresh logits
use the exact binaries, artifact and token cache from one saved comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.affinity import pin_cpu
from tools.benchmark_decoder_comparison import (native_experiment_gate, profiling_disabled,
    stage_summary, validate_inputs, validate_timing_request)
from tools.decoder_validation import benchmark_stability, quality, run_native
from tools.validate_decoder import digest, native_policy, utc_now, write_record


def replay(record: dict) -> dict:
    """Recompute every saved pass/stage/gate from unfiltered raw samples."""
    checks = {}
    for key, case in record["native"].items():
        timing = case["timing"]
        passes = timing["passes"]
        order = [p["stage"] for p in passes]
        if order != ["before", "after", "after", "before"] or timing["order"] != order:
            raise ValueError("Saved comparison is not ABBA")
        for p in passes:
            validate_timing_request(p["latency"], runs=record["runs_per_pass"],
                                    threads=record["threads"], activation_bits=case["activation_bits"])
            if p["native_policy"] != record["native_policy_by_stage"][p["stage"]]:
                raise ValueError("Saved pass policy differs from stage policy")
            if benchmark_stability(p["latency"]) != p["stability"]:
                raise ValueError("Saved pass stability differs from raw samples")
        stages = {name: stage_summary([p["latency"] for p in passes if p["stage"] == name])
                  for name in ("before", "after")}
        if any(stages[name] != timing[name] for name in stages):
            raise ValueError("Saved stage summary differs from raw samples")
        gate = native_experiment_gate(stages["before"], stages["after"], case["quality_gate_passed"])
        if gate != case["gate"]:
            raise ValueError("Saved verdict differs from recomputed gate")
        checks[key] = {"raw_sample_replay_passed": True, "gate": gate}
    return checks


def independent_loss(logits: np.ndarray, sequences: list[list[int]]) -> dict:
    """Separate PyTorch float64 CE oracle; shift within each block, never across."""
    import torch
    import torch.nn.functional as functional
    losses, offset = [], 0
    for ids in sequences:
        rows = torch.from_numpy(np.array(logits[offset:offset + len(ids) - 1], dtype=np.float64))
        targets = torch.tensor(ids[1:], dtype=torch.long)
        losses.append(functional.cross_entropy(rows, targets, reduction="none").numpy())
        offset += len(ids)
    values = np.concatenate(losses)
    return {"evaluated_tokens": len(values), "mean_nll": float(values.mean()),
            "perplexity": float(np.exp(values.mean())),
            "block_mean_nll": [float(x.mean()) for x in losses]}


def error_summary(actual: np.ndarray, reference: np.ndarray) -> dict:
    error = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    tolerance = 2e-3 + 2e-3 * np.abs(reference.astype(np.float64))
    return {"max_absolute_error": float(error.max()), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "max_fraction_of_allclose_tolerance": float(np.max(error / tolerance)),
            "allclose_passed": bool(np.all(error <= tolerance))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons", nargs="+", type=Path, required=True)
    parser.add_argument("--quality-comparison", type=Path, required=True)
    parser.add_argument("--frozen-record", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [*args.comparisons, args.quality_comparison, args.frozen_record,
              args.before, args.after, args.workdir / "reference.npy", args.workdir / "tokens.json",
              Path(__file__), ROOT / "tools/decoder_validation.py", ROOT / "tools/benchmark_decoder_comparison.py"]
    if args.output.exists() or args.output.resolve() in [p.resolve() for p in inputs]:
        raise ValueError("Audit output must be a new file")
    snapshots = {str(p): digest(p) for p in inputs}
    saved = json.loads(args.quality_comparison.read_text())
    if (saved["before_executable_sha256"] != digest(args.before) or
            saved["after_executable_sha256"] != digest(args.after) or
            saved["frozen_record_sha256"] != digest(args.frozen_record) or
            any(policy != native_policy() for policy in saved["native_policy_by_stage"].values())):
        raise ValueError("Paired audit requires the saved binaries and identical BEFORE/AFTER environment policy")
    context = validate_inputs(SimpleNamespace(before=args.before, after=args.after, workdir=args.workdir,
        record=args.frozen_record, keys=["32"], cpu=saved["pinned_cpu"]))
    artifact = context["artifacts"]["32"]
    if args.output.resolve() == artifact.resolve():
        raise ValueError("Audit output cannot overwrite artifact")
    snapshots[str(artifact)] = digest(artifact)
    replayed = {str(p): replay(json.loads(p.read_text())) for p in args.comparisons}
    replay(saved)
    pin_cpu(context["cpu"])
    sequences, reference = context["data"]["sequences"], context["reference"]
    outputs, metrics = {}, {}
    with profiling_disabled():
        for stage, binary in (("before", args.before), ("after", args.after)):
            print(f"Fresh {stage} held-out logits", flush=True)
            values, execution = run_native(binary, artifact, sequences, threads=context["threads"], activation_bits=32)
            outputs[stage] = values.reshape(reference.shape)
            metrics[stage] = {"quality": quality(outputs[stage], reference, sequences), "execution": execution,
                              "error_vs_pytorch": error_summary(outputs[stage], reference)}
    import torch
    torch.set_num_threads(context["threads"])
    oracle = {name: independent_loss(values, sequences)
              for name, values in {"pytorch": reference, **outputs}.items()}
    for stage in outputs:
        if not np.isclose(oracle[stage]["perplexity"], metrics[stage]["quality"]["perplexity"], rtol=1e-12, atol=1e-12):
            raise AssertionError("Independent CE disagrees with quality metric")
        if not np.isclose(oracle["pytorch"]["perplexity"], metrics[stage]["quality"]["pytorch_perplexity"], rtol=1e-12, atol=1e-12):
            raise AssertionError("Independent reference CE disagrees with quality metric")
    paired = error_summary(outputs["after"], outputs["before"])
    paired["mean_nll_delta_after_before"] = oracle["after"]["mean_nll"] - oracle["before"]["mean_nll"]
    paired["perplexity_ratio_after_before"] = oracle["after"]["perplexity"] / oracle["before"]["perplexity"]
    paired["block_mean_nll_deltas"] = [a - b for a, b in zip(oracle["after"]["block_mean_nll"], oracle["before"]["block_mean_nll"])]
    if any(digest(Path(p)) != value for p, value in snapshots.items()):
        raise ValueError("Audit inputs changed during execution")
    write_record(args.output, {"audit": "decoder-inference-error-and-benchmark-replay-v1", "measured_at_utc": utc_now(),
        "input_sha256": snapshots, "native_policy": native_policy(), "timing_replay": replayed,
        "fresh_quality": metrics, "independent_float64_cross_entropy": oracle,
        "independent_loss_agreement_passed": True, "paired_gelu_error": paired,
        "fresh_latency_measured": False, "automatic_selection_authorized": False,
        "scope": "Inference on frozen trained GPT-2 subset, not training loss or gradient validation. No new timing qualification."})
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
