"""Shared native protocol and numerical/model-quality measurements."""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np


MIN_LATENCY_SAMPLES = 5
MAX_LATENCY_P90_P10_RATIO = 1.25
# Native JSON currently uses six significant digits. Even-count medians can
# therefore differ slightly from the median of individually rounded samples.
LATENCY_MEDIAN_REL_TOL = 1e-5
LATENCY_MEDIAN_ABS_TOL_MS = 1e-6


def latency_stability(samples) -> dict:
    """Conservative repeatability check; never discard timing outliers.

    Percentiles use NumPy's linear interpolation. Missing, short or invalid
    measurements cannot qualify a faster configuration for automatic use.
    Raw samples remain in the caller's latency record, unchanged.
    """
    record = {"passed": False, "sample_count": 0,
              "required_samples": MIN_LATENCY_SAMPLES,
              "max_p90_p10_ratio": MAX_LATENCY_P90_P10_RATIO,
              "p10_ms": None, "p90_ms": None, "p90_p10_ratio": None}
    if not isinstance(samples, (list, tuple, np.ndarray)):
        return dict(record, reason="missing_or_invalid_samples")
    # Do not ask NumPy to infer the shape of a Python sequence first: ragged
    # input can raise during conversion rather than producing an invalid record.
    if isinstance(samples, np.ndarray) and samples.ndim != 1:
        return dict(record, reason="samples_must_be_finite_positive_numbers")
    record["sample_count"] = len(samples)
    if any(isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, float, np.integer, np.floating)) for value in samples):
        return dict(record, reason="samples_must_be_finite_positive_numbers")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(samples, dtype=np.float64)
    except (OverflowError, TypeError, ValueError):
        return dict(record, reason="samples_must_be_finite_positive_numbers")
    if not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        return dict(record, reason="samples_must_be_finite_positive_numbers")
    p10, p90 = np.percentile(values, [10, 90], method="linear")
    with np.errstate(over="ignore"):
        ratio = float(p90 / p10)
    record.update(p10_ms=float(p10), p90_ms=float(p90),
                  p90_p10_ratio=ratio if np.isfinite(ratio) else None)
    if len(values) < MIN_LATENCY_SAMPLES:
        return dict(record, reason="insufficient_samples")
    if not np.isfinite(ratio) or ratio > MAX_LATENCY_P90_P10_RATIO:
        return dict(record, reason="timing_spread_exceeds_threshold")
    return dict(record, passed=True, reason="passed")


def benchmark_stability(latency: dict) -> dict:
    """Check both phases of one whole-model benchmark."""
    latency = latency if isinstance(latency, dict) else {}
    phases = {phase: latency_phase_stability(latency, phase)
              for phase in ("prefill", "decode")}
    return {"passed": all(result["passed"] for result in phases.values()), **phases}


def latency_median(latency: dict | None, phase: str) -> float | None:
    """Read a reported median without allowing booleans or non-finite values."""
    value = latency.get(f"{phase}_p50_ms") if isinstance(latency, dict) else None
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, float, np.integer, np.floating))):
        return None
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    return result if np.isfinite(result) and result > 0 else None


def latency_phase_stability(latency: dict | None, phase: str) -> dict:
    """Check spread and agreement between a reported median and retained data."""
    latency = latency if isinstance(latency, dict) else {}
    samples = latency.get(f"{phase}_samples_ms")
    result = latency_stability(samples)
    declared = latency_median(latency, phase)
    raw_median = None
    if result["p10_ms"] is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            measured = float(np.median(np.asarray(samples, dtype=np.float64)))
        if np.isfinite(measured) and measured > 0:
            raw_median = measured
    consistent = bool(declared is not None and raw_median is not None and math.isclose(
        declared, raw_median, rel_tol=LATENCY_MEDIAN_REL_TOL, abs_tol=LATENCY_MEDIAN_ABS_TOL_MS))
    result.update(reported_median_ms=declared, samples_median_ms=raw_median, median_consistent=consistent,
                  median_relative_tolerance=LATENCY_MEDIAN_REL_TOL,
                  median_absolute_tolerance_ms=LATENCY_MEDIAN_ABS_TOL_MS)
    if not consistent:
        result["passed"] = False
        if result["reason"] == "passed":
            result["reason"] = "reported_median_does_not_match_samples"
    return result


def comparison_stability(candidate: dict, leaf_fp32: dict | None, pytorch: dict) -> dict:
    """Require stable candidates and stable baselines used by the speed gate.

    PyTorch may win prefill with one implementation and decode with another.
    Tied fastest paths are all checked for the phase in which they are fastest.
    """
    candidate_check = benchmark_stability(candidate)
    fp32_check = benchmark_stability(leaf_fp32) if isinstance(leaf_fp32, dict) else None
    pytorch = pytorch if isinstance(pytorch, dict) else {}
    fastest = {}
    for phase in ("prefill", "decode"):
        checks = {name: latency_phase_stability(latency, phase) for name, latency in pytorch.items()}
        valid = {name: value for name, latency in pytorch.items()
                 if (value := latency_median(latency, phase)) is not None}
        if (len(valid) != len(pytorch) or not valid or
                any(item["median_consistent"] is not True for item in checks.values())):
            fastest[phase] = {"passed": False, "implementations": {},
                              "reason": "missing_invalid_or_inconsistent_pytorch_medians"}
            continue
        minimum = min(valid.values())
        implementations = {name: checks[name]
                           for name, value in valid.items() if value == minimum}
        fastest[phase] = {"passed": all(result["passed"] for result in implementations.values()),
                          "median_ms": minimum, "implementations": implementations}
    passed = (candidate_check["passed"] and bool(fp32_check and fp32_check["passed"]) and
              all(phase["passed"] for phase in fastest.values()))
    return {"passed": bool(passed), "minimum_samples": MIN_LATENCY_SAMPLES,
            "max_p90_p10_ratio": MAX_LATENCY_P90_P10_RATIO,
            "candidate": candidate_check, "leaf_fp32": fp32_check,
            "fastest_pytorch": fastest}


def run_native(executable: Path, artifact: Path, sequences: list[list[int]],
               mode: str = "verify", threads: int = 1, runs: int = 3,
               warmup: int = 1, generate: int = 16, scalar: bool = False,
               chunk: int = 1, activation_bits: int | None = None,
               high_qos: bool = False, capture_profile: bool = False,
               windows_above_normal: bool = False) -> tuple[np.ndarray, dict]:
    if windows_above_normal and sys.platform != "win32":
        raise ValueError("Above-normal child priority requires Windows")
    with tempfile.TemporaryDirectory(prefix="leaf_decoder_") as temp:
        root = Path(temp)
        request, output, metrics = root / "tokens.bin", root / "logits.bin", root / "metrics.json"
        with request.open("wb") as stream:
            stream.write(struct.pack("<I", len(sequences)))
            for ids in sequences:
                stream.write(struct.pack("<I", len(ids)))
                stream.write(np.asarray(ids, dtype="<u4").tobytes())
        command = [str(executable.resolve()), str(artifact.resolve()), str(request),
                   str(output), str(metrics), mode, str(threads), str(runs),
                   str(warmup), str(generate), str(int(scalar)), str(chunk)]
        if activation_bits is not None:
            command.append(str(activation_bits))
        policy = None
        if high_qos or windows_above_normal:
            from leaf.power import set_child_high_qos
            flags = subprocess.ABOVE_NORMAL_PRIORITY_CLASS if windows_above_normal else 0
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                       creationflags=flags)
            try:
                if high_qos:
                    policy = set_child_high_qos(process)
                if windows_above_normal:
                    import psutil
                    if psutil.Process(process.pid).nice() != psutil.ABOVE_NORMAL_PRIORITY_CLASS:
                        raise RuntimeError("Child priority does not match requested above-normal class")
                stdout, stderr = process.communicate()
            except BaseException:
                process.kill()
                process.communicate()
                raise
            completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        else:
            completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"Native decoder failed ({completed.returncode}): {completed.stderr.strip()}")
        measured = json.loads(metrics.read_text())
        if capture_profile:
            profiles = [json.loads(line) for line in completed.stderr.splitlines()
                        if line.startswith('{"leaf_decoder_profile":')]
            if len(profiles) != 1 or profiles[0].get("leaf_decoder_profile") != 1:
                raise ValueError("Expected exactly one native decoder phase profile")
            measured["diagnostic_profile"] = profiles[0]
        if policy is not None:
            measured["windows_process_qos"] = policy
        if windows_above_normal:
            measured["windows_process_priority"] = "above_normal"
        return np.fromfile(output, dtype="<f4"), measured


def quality(logits: np.ndarray, baseline: np.ndarray, sequences: list[list[int]]) -> dict:
    if (logits.shape != baseline.shape or logits.ndim != 2 or
            logits.shape[0] != sum(map(len, sequences)) or
            not np.isfinite(logits).all() or not np.isfinite(baseline).all()):
        raise AssertionError("Invalid native logits")
    predictions = np.argmax(logits, axis=-1)
    reference = np.argmax(baseline, axis=-1)
    nll, reference_nll, evaluated = 0.0, 0.0, 0
    agreement = 0
    position = 0
    for ids in sequences:
        for row, target in enumerate(ids[1:]):
            if not 0 <= target < logits.shape[-1]:
                raise ValueError("Quality target is outside the vocabulary")
            agreement += predictions[position + row] == reference[position + row]
            for values, is_reference in ((logits[position + row], False), (baseline[position + row], True)):
                maximum = float(np.max(values))
                loss = maximum + float(np.log(np.exp(values.astype(np.float64) - maximum).sum())) - float(values[target])
                if is_reference:
                    reference_nll += loss
                else:
                    nll += loss
            evaluated += 1
        position += len(ids)
    if not evaluated:
        raise ValueError("Quality evaluation requires at least one next-token target")
    perplexity, reference_perplexity = float(np.exp(nll / evaluated)), float(np.exp(reference_nll / evaluated))
    return {"evaluated_tokens": evaluated, "max_abs_logit_error": float(np.max(np.abs(logits - baseline))),
            "relative_logit_rmse": float(np.sqrt(np.mean((logits - baseline) ** 2)) / max(np.sqrt(np.mean(baseline ** 2)), 1e-12)),
            "next_token_agreement": float(agreement / evaluated),
            "perplexity": perplexity, "pytorch_perplexity": reference_perplexity,
            "perplexity_ratio": perplexity / reference_perplexity}
