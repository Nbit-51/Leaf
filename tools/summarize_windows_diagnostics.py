"""Publish aggregate diagnostics without a user's process inventory or PIDs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_decoder_comparison import stage_summary
from tools.validate_decoder import digest, write_record


def distribution(values):
    values = [x for x in values if x is not None]
    return {"count": len(values), "minimum": min(values), "median": statistics.median(values),
            "maximum": max(values)} if values else {"count": 0}


def summarize(record):
    cpu = record["cpu"]
    own = next(c for c in record["cpu_sets"] if c["group"] == 0 and c["logical_cpu"] == cpu)
    siblings = [c["logical_cpu"] for c in record["cpu_sets"]
                if c["group"] == own["group"] and c["core"] == own["core"] and c["logical_cpu"] != cpu]
    passes = []
    for p in record["passes"]:
        samples = p["telemetry"]
        active = [s for s in samples if s.get("child_one_core_percent", 0) > 80]
        totals = defaultdict(float)
        for sample in samples:
            items = sample.get("process_activity", sample["counters"].get("process_activity", []))
            for item in items:
                name = item["name"].lower().split('#')[0].removesuffix('.exe')
                seconds = item.get("cpu_seconds", item["one_core_percent"] * sample["interval_seconds"] / 100)
                category = ("vs_code" if name == "code" else
                            "cpp_tools" if name in ("g++", "gcc", "cc1plus", "clang", "clang++", "clangd", "cpptools", "cmake", "ninja", "msbuild", "cl") else
                            "native_benchmark" if name == "leaf_decoder" else
                            "python_processes" if name == "python" else "other_processes")
                totals[category] += seconds
        passes.append({"index": p["index"], "high_qos": p["high_qos"], "latency": p["latency"],
            "stability": p["stability"], "telemetry_samples": len(samples),
            "collection_seconds": distribution([s.get("collection_seconds") for s in samples]),
            "child_one_core_percent": distribution([s.get("child_one_core_percent") for s in samples]),
            "performance_percent_while_child_busy": distribution([s["counters"].get("cpu_performance_percent") for s in active]),
            "sibling_busy_percent": {str(c): distribution([s["cpu_busy_percent"][c] for s in samples]) for c in siblings},
            "page_reads_per_second": distribution([s["counters"].get("page_reads_per_second") for s in samples]),
            "available_memory_bytes": distribution([s["available_memory_bytes"] for s in samples]),
            "power_plugged_every_sample": all(s["power_plugged"] is True for s in samples),
            "estimated_process_cpu_seconds_by_category": dict(totals)})
    stages = {}
    for high_qos in (False, True):
        values = [p["latency"] for p in passes if p["high_qos"] == high_qos]
        if len(values) == 2:
            stages["high_qos" if high_qos else "normal"] = stage_summary(values)
    return {"complete": record["complete"], "inputs_unchanged": record["inputs_unchanged"],
            "measured_at_utc": record["measured_at_utc"], "cpu": cpu, "monitor_cpu": record["monitor_cpu"],
            "cpu_sets": record["cpu_sets"], "runs": record["runs"], "warmup": record["warmup"],
            "native_policy": record["native_policy"], "process_sampler": record.get("process_sampler", "initial psutil enumeration (high overhead)"),
            "active_power_scheme": record["active_power_scheme"], "processor_power_settings": record["processor_power_settings"],
            "counter_unavailable": record["counter_unavailable"], "passes": passes, "stages": stages,
            "automatic_selection_authorized": False, "acceptance_timing": False,
            "scope": record["scope"] + " Process-category CPU seconds are approximate and include loading and warmups."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    result = {p.name: {"raw_record_sha256": digest(p), "summary": summarize(json.loads(p.read_text()))} for p in args.inputs}
    write_record(args.output, result)


if __name__ == "__main__":
    main()
