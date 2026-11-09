"""Validate optional native Windows accounting without reinterpreting it as Hz.

Process CPU accounting includes every worker and bounded-spin work, so its sum
can exceed elapsed wall time. Windows CPU-time accounting can be coarse and a
short sample can legitimately report zero. Cycle counters are retained as raw
counts; they do not establish clock frequency, instructions retired or waiting
time. Page faults include both soft and hard faults. These summaries are
diagnostics, never acceptance criteria or reasons to discard slow samples.
"""
from __future__ import annotations

import math
from statistics import median


_PHASES = ("prefill", "decode")
_FLOAT_FIELDS = ("interval_ms", "process_cpu_ms", "caller_cpu_ms")
_INTEGER_LIMITS = {
    "process_cycles": (1 << 64) - 1,
    "caller_cycles": (1 << 64) - 1,
    "process_page_faults": (1 << 32) - 1,
    "caller_cpu_start": 255,
    "caller_cpu_end": 255,
    "caller_group_start": (1 << 16) - 1,
    "caller_group_end": (1 << 16) - 1,
}
# Native metrics use ten significant digits; tolerate independent decimal
# rounding of a forward wall sample and its (slightly larger) counter bracket.
_INTERVAL_REL_TOL = 1e-8
_INTERVAL_ABS_TOL_MS = 1e-6


def _array(record: dict, key: str, count: int, location: str) -> list:
    values = record.get(key)
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{location}.{key} must contain exactly {count} samples")
    return values


def _number(value, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and (value > 0 if positive else value >= 0)
    except (OverflowError, ValueError):
        return False


def validate_benchmark_counters(metrics: dict, expected_samples: int) -> None:
    """Require complete version-one accounting aligned with both phase arrays.

    ``expected_samples`` counts measured samples, excluding all warmups, across
    every request. An absent or malformed requested counter must fail rather
    than becoming a zero in the report. No relation is enforced between process
    CPU sums, caller CPU time and wall time: they use different accounting and
    query boundaries, and worker time is additive.
    """
    if (isinstance(expected_samples, bool) or not isinstance(expected_samples, int)
            or expected_samples < 1):
        raise ValueError("Expected counter sample count must be a positive integer")
    if not isinstance(metrics, dict):
        raise ValueError("Benchmark metrics must be an object")
    counters = metrics.get("benchmark_counters")
    if not isinstance(counters, dict):
        raise ValueError("Requested benchmark_counters object is missing")
    version = counters.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("Unsupported benchmark_counters version; require integer version 1")
    for field in ("scope", "interval"):
        if not isinstance(counters.get(field), str) or not counters[field].strip():
            raise ValueError(f"benchmark_counters.{field} must describe the counter scope")
    for phase in _PHASES:
        location = f"benchmark_counters.{phase}"
        samples = counters.get(phase)
        if not isinstance(samples, dict):
            raise ValueError(f"{location} object is missing")
        wall = _array(metrics, phase + "_samples_ms", expected_samples, "metrics")
        if any(not _number(value, positive=True) for value in wall):
            raise ValueError(f"metrics.{phase}_samples_ms must be finite positive numbers")
        for field in _FLOAT_FIELDS:
            values = _array(samples, field, expected_samples, location)
            if any(not _number(value, positive=field == "interval_ms") for value in values):
                raise ValueError(f"{location}.{field} has invalid nonnegative numeric samples")
        for field, maximum in _INTEGER_LIMITS.items():
            values = _array(samples, field, expected_samples, location)
            if any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 0 or value > maximum for value in values):
                raise ValueError(f"{location}.{field} must contain integers in [0, {maximum}]")
        for forward, interval in zip(wall, samples["interval_ms"]):
            if interval < forward and not math.isclose(
                    interval, forward, rel_tol=_INTERVAL_REL_TOL,
                    abs_tol=_INTERVAL_ABS_TOL_MS):
                raise ValueError(f"{location}.interval_ms cannot be shorter than the forward wall sample")


def summarize_benchmark_counters(metrics: dict) -> dict:
    """Summarize every retained sample; no filtering or cause classification.

    Forward latency and the larger counter bracket remain separate. The
    brackets overlap around the shared middle stamp, so adding prefill and
    decode bracket intervals does not produce a combined elapsed duration.
    """
    if not isinstance(metrics, dict) or not isinstance(metrics.get("prefill_samples_ms"), list):
        raise ValueError("Benchmark metrics require prefill sample arrays")
    count = len(metrics["prefill_samples_ms"])
    validate_benchmark_counters(metrics, count)
    result = {}
    for phase in _PHASES:
        samples = metrics["benchmark_counters"][phase]
        endpoints = sorted({(group, cpu)
            for boundary in ("start", "end")
            for group, cpu in zip(samples[f"caller_group_{boundary}"], samples[f"caller_cpu_{boundary}"])})
        result[phase] = dict(
            sample_count=count,
            wall_median_ms=median(metrics[phase + "_samples_ms"]),
            interval_median_ms=median(samples["interval_ms"]),
            process_cpu_median_ms=median(samples["process_cpu_ms"]),
            caller_cpu_median_ms=median(samples["caller_cpu_ms"]),
            process_cycles_median=median(samples["process_cycles"]),
            caller_cycles_median=median(samples["caller_cycles"]),
            process_page_faults_total=sum(samples["process_page_faults"]),
            caller_endpoints=[dict(group=group, cpu=cpu) for group, cpu in endpoints],
        )
    return result
