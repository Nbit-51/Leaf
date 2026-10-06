"""Validate a native decoder experiment and measure serial whole-model ABBA.

This is a same-machine native experiment, not fresh PyTorch benchmarking or
permission to enable automatic quantization. Quality uses the frozen trained
reference; unchanged artifacts and the before executable are checked by hash.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.affinity import pin_cpu
from leaf.power import keep_awake
from tools.decoder_validation import (MAX_LATENCY_P90_P10_RATIO, MIN_LATENCY_SAMPLES,
                                      benchmark_stability, latency_median,
                                      latency_stability, quality, run_native)
from tools.validate_decoder import GATES, digest, native_policy, utc_now, write_record


NATIVE_POLICY_NAMES = ("LEAF_DISABLE_VNNI", "LEAF_EXPERIMENTAL_FLOAT_TILES", "LEAF_EXPERIMENTAL_FLOAT_GEMV")
DEFAULT_NATIVE_POLICY = {name: False for name in NATIVE_POLICY_NAMES}


@contextmanager
def profiling_disabled():
    """Profiling changes hot-loop cost; never include it in timed child runs."""
    previous = os.environ.pop("LEAF_DECODER_PROFILE", None)
    try:
        yield
    finally:
        if previous is not None:
            os.environ["LEAF_DECODER_PROFILE"] = previous


@contextmanager
def native_stage_policy(stage: str, after_float_tiles: bool = False):
    """Apply an experiment only to its owned stage, restoring the caller exactly."""
    if stage not in ("before", "after"):
        raise ValueError("Native experiment stage must be before or after")
    if not after_float_tiles:
        yield native_policy()
        return
    previous = {name: os.environ.get(name) for name in NATIVE_POLICY_NAMES}
    try:
        for name in NATIVE_POLICY_NAMES:
            os.environ.pop(name, None)
        if stage == "after":
            os.environ["LEAF_EXPERIMENTAL_FLOAT_TILES"] = "1"
        yield native_policy()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _tokens(values, *, nonempty=True):
    return (isinstance(values, list) and (bool(values) or not nonempty) and
            all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values))


def validate_inputs(args) -> dict:
    """Preflight all frozen quality and artifact provenance before execution."""
    if getattr(args, "windows_high_qos", False) and os.name != "nt":
        raise ValueError("--windows-high-qos requires Windows")
    scoped_tiles = bool(getattr(args, "after_float_tiles", False))
    if scoped_tiles:
        conflicts = sorted(name for name, value in os.environ.items() if value and
                           (name == "LEAF_DISABLE_VNNI" or name.startswith("LEAF_EXPERIMENTAL_")))
        if conflicts:
            raise ValueError("--after-float-tiles conflicts with parent native policy: " + ", ".join(conflicts))
    record = json.loads(args.record.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("pytorch"), dict):
        raise ValueError("Frozen record has no PyTorch quality provenance")
    if os.environ.get("LEAF_DISABLE_VNNI"):
        raise ValueError("Unset LEAF_DISABLE_VNNI; comparison cannot reuse quality under a changed ISA policy")
    if scoped_tiles:
        frozen_policy = record.get("native_policy")
        if (not isinstance(frozen_policy, dict) or set(frozen_policy) != set(DEFAULT_NATIVE_POLICY) or
                any(not isinstance(value, bool) for value in frozen_policy.values()) or
                frozen_policy != DEFAULT_NATIVE_POLICY):
            raise ValueError("--after-float-tiles requires an explicitly default frozen BEFORE native policy")
    else:
        frozen_policy = record.get("native_policy")
        if (not isinstance(frozen_policy, dict) or set(frozen_policy) != set(DEFAULT_NATIVE_POLICY) or
                any(not isinstance(value, bool) for value in frozen_policy.values()) or
                frozen_policy != native_policy()):
            raise ValueError("Current native policy differs from explicit frozen BEFORE quality policy")
    baseline = record["pytorch"]
    if (baseline.get("platform") != platform.platform() or
            baseline.get("cpu") != platform.processor()):
        raise ValueError("Current OS/CPU description differs from the frozen baseline")
    if baseline.get("quality_gate_thresholds") != GATES:
        raise ValueError("Frozen quality thresholds differ; rerun trained validation")
    threads, cpu = baseline.get("threads"), baseline.get("pinned_cpu")
    if not isinstance(threads, int) or isinstance(threads, bool) or threads <= 0:
        raise ValueError("Frozen record has invalid threads")
    if cpu is not None and (not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0):
        raise ValueError("Frozen record has invalid CPU pin")
    requested_cpu = getattr(args, "cpu", None)
    if requested_cpu is not None:
        if not isinstance(requested_cpu, int) or isinstance(requested_cpu, bool) or requested_cpu < 0:
            raise ValueError("CPU override must be a nonnegative integer")
        cpu = requested_cpu
    for name, minimum in (("blocks", 1), ("sequence_length", 2)):
        value = baseline.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"Frozen record has invalid {name}")
    before_hash, after_hash = digest(args.before), digest(args.after)
    if record.get("native_executable_sha256") != before_hash:
        raise ValueError("Before executable does not match the frozen quality record")
    expected_hashes = baseline.get("quality_cache_sha256")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != {"reference.npy", "tokens.json"}:
        raise ValueError("Frozen record must contain reference.npy and tokens.json quality-cache hashes")
    cache_hashes = {name: digest(args.workdir / name) for name in expected_hashes}
    if expected_hashes != cache_hashes:
        raise ValueError("Quality cache hashes do not match the frozen record")
    data = json.loads((args.workdir / "tokens.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Frozen token cache must be an object")
    sequences = data.get("sequences")
    if (not isinstance(sequences, list) or not sequences or
            len(sequences) != baseline.get("blocks") or
            any(not _tokens(ids) or len(ids) != baseline.get("sequence_length") for ids in sequences) or
            not _tokens(data.get("benchmark_ids")) or len(data["benchmark_ids"]) < 2 or
            not _tokens(data.get("generation_ids")) or
            not _tokens(data.get("generated_tokens"), nonempty=False) or
            data["generated_tokens"] != baseline.get("generated_tokens")):
        raise ValueError("Frozen tokens do not match the baseline evaluation dimensions/generation")
    reference = np.load(args.workdir / "reference.npy", mmap_mode="r", allow_pickle=False)
    if (reference.ndim != 2 or reference.shape[0] != sum(map(len, sequences)) or
            reference.shape[1] <= 0 or reference.dtype.kind != "f"):
        raise ValueError("Frozen reference logits have invalid dimensions/dtype")
    cases = record.get("native")
    if not isinstance(cases, dict) or not cases:
        raise ValueError("Frozen record has no native candidates")
    keys = list(cases) if args.keys is None else args.keys
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Select at least one unique candidate key")
    artifacts = {}
    evaluated_tokens = sum(len(ids) - 1 for ids in sequences)
    for key in keys:
        case = cases.get(key)
        if (not isinstance(case, dict) or case.get("weight_bits") not in (32, 8, 4) or
                case.get("activation_bits") not in (32, 8) or
                not isinstance(case.get("quality"), dict) or
                case["quality"].get("evaluated_tokens") != evaluated_tokens or
                not isinstance(case.get("quality_gate_passed"), bool) or
                not _tokens(case.get("generated_tokens"), nonempty=False) or
                not isinstance(case.get("artifact"), str)):
            raise ValueError(f"Frozen candidate {key} has incomplete trained-quality provenance")
        artifact = args.workdir / case["artifact"]
        if (artifact.resolve().parent != args.workdir.resolve() or not artifact.is_file() or
                case.get("artifact_sha256") != digest(artifact)):
            raise ValueError(f"Artifact {key} is external, missing or differs from frozen quality")
        artifacts[key] = artifact
    return {"record": record, "record_sha256": digest(args.record), "data": data,
            "reference": reference, "keys": keys, "artifacts": artifacts,
            "before_sha256": before_hash, "after_sha256": after_hash,
            "quality_cache_sha256": cache_hashes, "threads": threads, "cpu": cpu,
            "high_qos": bool(getattr(args, "windows_high_qos", False)),
            "after_float_tiles": scoped_tiles}


def stage_summary(passes: list[dict]) -> dict:
    """Keep all timing samples, check within-pass noise and between-pass drift."""
    aggregate, pass_medians, between_pass = {}, {}, {}
    for phase in ("prefill", "decode"):
        parts = [item.get(f"{phase}_samples_ms") if isinstance(item, dict) else None for item in passes]
        samples = [value for part in parts for value in part] if all(isinstance(part, list) for part in parts) else []
        stable = latency_stability(samples)
        aggregate[f"{phase}_samples_ms"] = samples
        aggregate[f"{phase}_p50_ms"] = float(statistics.median(samples)) if stable["p10_ms"] is not None else None
        medians = [latency_median(item, phase) for item in passes]
        pass_medians[phase] = medians
        valid = bool(medians) and all(value is not None for value in medians)
        ratio = max(medians) / min(medians) if valid else None
        if ratio is not None and not np.isfinite(ratio):
            ratio, valid = None, False
        between_pass[phase] = {"passed": bool(valid and ratio <= MAX_LATENCY_P90_P10_RATIO),
                               "max_min_pass_median_ratio": ratio,
                               "maximum_ratio": MAX_LATENCY_P90_P10_RATIO}
    pass_checks = [benchmark_stability(item) for item in passes]
    stability = benchmark_stability(aggregate)
    aggregate.update(stability=stability, pass_stability=pass_checks,
                     pass_medians_ms=pass_medians, between_pass_stability=between_pass,
                     stability_passed=bool(len(passes) == 2 and stability["passed"] and
                         all(item["passed"] for item in pass_checks) and
                         all(item["passed"] for item in between_pass.values())))
    return aggregate


def native_experiment_gate(before: dict, after: dict, quality_passed: bool, objective: str = "prefill") -> dict:
    if objective not in ("prefill", "decode"):
        raise ValueError("Unknown performance objective")
    values = {phase: (latency_median(before, phase), latency_median(after, phase))
              for phase in ("prefill", "decode")}
    ratios = {phase: candidate / baseline if baseline is not None and candidate is not None else None
              for phase, (baseline, candidate) in values.items()}
    ratios = {phase: value if value is not None and np.isfinite(value) else None
              for phase, value in ratios.items()}
    stable = before.get("stability_passed") is True and after.get("stability_passed") is True
    no_slowdown = all(value is not None and value <= 1.02 for value in ratios.values())
    improved = ratios[objective] is not None and ratios[objective] <= 0.98
    return {"quality_passed": quality_passed is True, "timing_stability_passed": stable,
            "prefill_ratio_after_before": ratios["prefill"], "decode_ratio_after_before": ratios["decode"],
            "no_slowdown_passed": no_slowdown, f"{objective}_improvement_passed": improved,
            "maximum_slowdown_ratio": 1.02, f"required_{objective}_ratio": 0.98,
            "accepted_native_experiment": bool(quality_passed is True and stable and no_slowdown and improved),
            "automatic_selection_authorized": False, "fresh_pytorch_comparison": False}


def validate_timing_request(metrics: dict, *, runs: int, threads: int, activation_bits: int) -> None:
    """Reject a child result that does not match the requested measurement."""
    if not isinstance(metrics, dict):
        raise ValueError("Native timing metrics must be an object")
    for name, expected in (("threads", threads), ("activation_bits", activation_bits)):
        if type(metrics.get(name)) is not int or metrics[name] != expected:
            raise ValueError(f"Native timing {name} differs from requested workload")
    for phase in ("prefill", "decode"):
        samples = metrics.get(f"{phase}_samples_ms")
        if not isinstance(samples, list) or len(samples) != runs:
            raise ValueError(f"Native timing {phase} sample count differs from requested runs")


def _quality_gate(measured: dict, bits: int) -> bool:
    gate = GATES[str(bits)]
    return bool(measured["next_token_agreement"] >= gate["min_next_token_agreement"] and
                measured["perplexity_ratio"] <= gate["max_perplexity_ratio"])


def verify_after(args, context: dict, key: str) -> dict:
    with native_stage_policy("after", bool(getattr(args, "after_float_tiles", False))) as policy:
        result = _verify_after(args, context, key)
        result["native_policy"] = policy
        return result


def _verify_after(args, context: dict, key: str) -> dict:
    case = context["record"]["native"][key]
    bits, activations = case["weight_bits"], case["activation_bits"]
    artifact, data, reference = context["artifacts"][key], context["data"], context["reference"]
    actual, execution = run_native(args.after, artifact, data["sequences"], threads=context["threads"],
                                    activation_bits=activations, high_qos=context["high_qos"])
    actual = actual.reshape(reference.shape)
    measured = quality(actual, reference, data["sequences"])
    first_ids = data["sequences"][0]
    chunk = min(16, max(1, len(first_ids) // 2))
    chunked, _ = run_native(args.after, artifact, [first_ids], mode="chunked", chunk=chunk,
                            threads=context["threads"], activation_bits=activations, high_qos=context["high_qos"])
    chunked = chunked.reshape(len(first_ids), reference.shape[-1])
    chunked_quality = quality(chunked, reference[:len(first_ids)], [first_ids])
    chunked_parity = bool(np.allclose(chunked, actual[:len(first_ids)], atol=2e-3, rtol=2e-3))
    chunked_fp32_parity = bool(np.allclose(chunked, reference[:len(first_ids)], atol=2e-3, rtol=2e-3)) if bits == 32 else None
    generation_count = max(1, len(data["generated_tokens"]), len(case["generated_tokens"]))
    _, generation = run_native(args.after, artifact, [data["generation_ids"]], mode="generate",
                                threads=context["threads"], generate=generation_count, activation_bits=activations,
                                high_qos=context["high_qos"])
    generated = generation["generated_tokens"]
    match_pytorch = generated == data["generated_tokens"]
    match_before = generated == case["generated_tokens"]
    fp32_parity = bool(np.allclose(actual, reference, atol=2e-3, rtol=2e-3)) if bits == 32 else None
    # The fixed trained thresholds qualify the complete held-out evaluation,
    # not a smaller individual block whose sampling variance can be larger.
    # Cached execution must still reproduce the same full-mode logits.
    passed = _quality_gate(measured, bits) and chunked_parity
    if bits == 32:
        passed = passed and fp32_parity and chunked_parity and chunked_fp32_parity and match_pytorch and match_before
    return {"weight_bits": bits, "activation_bits": activations,
            "artifact": artifact.name, "artifact_sha256": case["artifact_sha256"],
            "quality": measured, "trained_quality_gate_passed": _quality_gate(measured, bits),
            "quality_gate_passed": bool(passed), "fp32_allclose": fp32_parity,
            "chunked": {"sequence_index": 0, "chunk_tokens": chunk, "quality": chunked_quality,
                        "quality_gate_passed": _quality_gate(chunked_quality, bits),
                        "parity_vs_full": chunked_parity, "fp32_allclose_pytorch": chunked_fp32_parity},
            "generated_tokens": generated, "generation_exact_match_pytorch": match_pytorch,
            "generation_exact_match_before": match_before,
            "frozen_before_quality_gate_passed": case["quality_gate_passed"],
            "execution": execution, "quality_measured_at_utc": utc_now()}


def compare(args) -> dict:
    if args.runs < MIN_LATENCY_SAMPLES or args.warmup < 0:
        raise ValueError(f"Require at least {MIN_LATENCY_SAMPLES} runs and nonnegative warmup")
    if args.output.resolve() == args.record.resolve():
        raise ValueError("Output must not overwrite the frozen quality record")
    if args.output.exists():
        raise ValueError("Output exists; preserve previous measurements")
    context = validate_inputs(args)
    pin_cpu(context["cpu"])
    cases = {}
    stage_policies = {stage: dict(native_policy()) for stage in ("before", "after")}
    if context["after_float_tiles"]:
        stage_policies = {"before": dict(DEFAULT_NATIVE_POLICY),
                          "after": {**DEFAULT_NATIVE_POLICY, "LEAF_EXPERIMENTAL_FLOAT_TILES": True}}
    with keep_awake(), profiling_disabled():
        for key in context["keys"]:
            print(f"Candidate {key}: AFTER trained quality/generation/chunked parity", flush=True)
            cases[key] = verify_after(args, context, key)
        for key in context["keys"]:
            records = []
            for index, stage in enumerate(("before", "after", "after", "before")):
                print(f"Candidate {key}: ABBA {index + 1}/4 ({stage})", flush=True)
                executable = args.before if stage == "before" else args.after
                with native_stage_policy(stage, context["after_float_tiles"]) as policy:
                    _, metrics = run_native(executable, context["artifacts"][key], [context["data"]["benchmark_ids"]],
                                             mode="bench", threads=context["threads"], runs=args.runs,
                                             warmup=args.warmup, activation_bits=cases[key]["activation_bits"],
                                             high_qos=context["high_qos"])
                validate_timing_request(metrics, runs=args.runs, threads=context["threads"],
                                        activation_bits=cases[key]["activation_bits"])
                records.append({"index": index, "stage": stage, "measured_at_utc": utc_now(),
                                "latency": metrics, "stability": benchmark_stability(metrics),
                                "native_policy": policy})
            before = stage_summary([item["latency"] for item in records if item["stage"] == "before"])
            after = stage_summary([item["latency"] for item in records if item["stage"] == "after"])
            cases[key]["timing"] = {"order": [item["stage"] for item in records], "passes": records,
                                    "before": before, "after": after}
            cases[key]["gate"] = native_experiment_gate(before, after, cases[key]["quality_gate_passed"], getattr(args, "objective", "prefill"))
    # Files must remain frozen for the entire comparison, not just its start.
    if (digest(args.before) != context["before_sha256"] or digest(args.after) != context["after_sha256"] or
            digest(args.record) != context["record_sha256"] or
            any(digest(args.workdir / name) != value for name, value in context["quality_cache_sha256"].items()) or
            any(digest(path) != cases[key]["artifact_sha256"] for key, path in context["artifacts"].items())):
        raise ValueError("Frozen inputs or executables changed during comparison")
    result = {"benchmark": "trained-decoder-native-experiment-abba", "objective": getattr(args, "objective", "prefill"), "measured_at_utc": utc_now(),
              "before_executable_sha256": context["before_sha256"], "after_executable_sha256": context["after_sha256"],
              "frozen_record_sha256": context["record_sha256"], "quality_cache_sha256": context["quality_cache_sha256"],
              "frozen_quality_measured_at_utc": context["record"].get("quality_measured_at_utc", context["record"].get("measured_at_utc")),
              "platform": platform.platform(), "cpu": platform.processor(), "machine": platform.machine(),
              "threads": context["threads"], "pinned_cpu": context["cpu"], "runs_per_pass": args.runs,
              "frozen_reference_pinned_cpu": context["record"]["pytorch"].get("pinned_cpu"),
              "warmup_per_pass": args.warmup, "profiling_enabled": False,
              "windows_high_qos": context["high_qos"],
              "after_float_tiles": context["after_float_tiles"], "native_policy_by_stage": stage_policies,
              "isa_policy": {"LEAF_DISABLE_VNNI": os.environ.get("LEAF_DISABLE_VNNI")},
              "experimental_policy": {name: os.environ.get(name) for name in
                  ("LEAF_EXPERIMENTAL_FLOAT_TILES", "LEAF_EXPERIMENTAL_FLOAT_GEMV")},
              "stability_thresholds": {"minimum_samples_per_pass": MIN_LATENCY_SAMPLES,
                                         "maximum_p90_p10_ratio": MAX_LATENCY_P90_P10_RATIO,
                                         "maximum_between_pass_median_ratio": MAX_LATENCY_P90_P10_RATIO},
              "frozen_pytorch_context": context["record"]["pytorch"],
              "native": cases, "automatic_selection_authorized": False,
              "scope": "Complete trained decoder against frozen held-out subset logits. Serial same-machine native ABBA only; PyTorch timings are historical context, not a fresh promotion baseline. Timed forward excludes artifact preparation/process startup. Quantized greedy equality and individual chunk quality are diagnostic; full held-out quality and chunked-vs-full parity qualify all precisions. FP32 also requires exact generation and reference allclose."}
    write_record(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True, help="immutable full trained-quality record for BEFORE")
    parser.add_argument("--keys", nargs="+", help="candidate keys in frozen record; default all")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cpu", type=int, help="pin BOTH native stages to this CPU; frozen PyTorch timing is not reused as a gate")
    parser.add_argument("--windows-high-qos", action="store_true",
                        help="opt in both owned native child processes to Windows HighQoS; no global power changes")
    parser.add_argument("--after-float-tiles", action="store_true",
                        help="scope experimental FP32 tiles to AFTER quality/timing only; BEFORE stays default")
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/decoder_comparison.json"))
    args = parser.parse_args()
    result = compare(args)
    print(json.dumps({key: case["gate"] for key, case in result["native"].items()}, indent=2))


if __name__ == "__main__":
    main()
