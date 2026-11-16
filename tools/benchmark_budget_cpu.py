"""Measure cached generation from an installed Leaf wheel on the current CPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

import psutil

from leaf.evaluation import write_record
from leaf.power import keep_awake
from tools.decoder_validation import latency_stability
from tools.verify_package_runtime import digest, run_process


DEFAULT_PROMPT = "Once upon a time, a small robot found"
TIMINGS = (
    "external_first_visible_text_ms", "external_wall_ms", "load_ms",
    "prefill_p50_ms", "decode_p50_ms", "time_to_first_token_ms",
)
PROBE = """
import json, os, sys
from pathlib import Path
from leaf import cli

model, revision, prompt = json.loads(sys.argv[1])
binary = Path(cli.__file__).parent / 'bin' / ('leaf_decoder.exe' if os.name == 'nt' else 'leaf_decoder')
if not binary.is_file():
    raise SystemExit('Install a Leaf platform wheel with a bundled decoder')
cache = cli.cache_root()
snapshot = cli.resolve_model(model, cache, offline=True, revision=revision)
artifact = cli.model_directory(snapshot, cache) / 'decoder-32.leaf'
if not artifact.is_file():
    raise SystemExit('Prepare the cached FP32 artifact with leaf run before benchmarking')
_, ids = cli.encode_prompt(snapshot, prompt, raw=True)
print(json.dumps(dict(binary=str(binary), artifact=str(artifact), input_ids=ids)))
"""


def summarize(rows):
    summary = {}
    for field in TIMINGS:
        samples = [row[field] for row in rows]
        stability = latency_stability(samples)
        summary[field] = dict(p50_ms=statistics.median(samples), stability=stability)
    summary["max_decoder_peak_rss_bytes"] = max(row["peak_rss_bytes"] for row in rows)
    return summary


def plugged_in():
    battery = psutil.sensors_battery()
    return None if battery is None else battery.power_plugged


def cpu_name():
    if os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="cached Hub model ID or local snapshot")
    parser.add_argument("--revision", help="cached Hub revision")
    parser.add_argument("--prompt-file", type=Path, help="UTF-8 text, default is a short raw continuation prompt")
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2], choices=(1, 2, 4))
    parser.add_argument("--allowed-cpus", nargs="+", type=int, help="optional process affinity for a constrained run")
    parser.add_argument("--runs", type=int, default=6, help="even number of recorded requests per setting, at least six")
    parser.add_argument("--output", required=True, type=Path, help="fresh output directory")
    args = parser.parse_args()
    if args.runs < 6 or args.runs % 2 or len(set(args.threads)) != len(args.threads):
        parser.error("Choose unique worker counts and an even number of runs of at least six")
    if args.allowed_cpus is not None and len(set(args.allowed_cpus)) < max(args.threads):
        parser.error("Provide at least one distinct allowed logical CPU per worker")

    prompt = args.prompt_file.read_text(encoding="utf-8-sig") if args.prompt_file else DEFAULT_PROMPT
    if not prompt.strip():
        parser.error("The prompt must contain text")
    environment = {key: value for key, value in os.environ.items()
                   if key.upper() != "LEAF_DECODER_BIN" and not key.upper().startswith("LEAF_EXPERIMENTAL_")}
    if "LEAF_CACHE_DIR" in environment:
        environment["LEAF_CACHE_DIR"] = str(Path(environment["LEAF_CACHE_DIR"]).resolve())
    model = str(Path(args.model).resolve()) if Path(args.model).is_dir() else args.model
    child = [sys.executable, "-I", "-u"]
    probe = subprocess.check_output(
        [*child, "-c", PROBE, json.dumps([model, args.revision, prompt])],
        env=environment, encoding="utf-8",
    )
    config = json.loads(probe)
    binary, artifact = (Path(config[key]) for key in ("binary", "artifact"))
    fingerprints = {key: digest(path) for key, path in (("binary", binary), ("artifact", artifact))}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    process = psutil.Process()
    has_affinity = hasattr(process, "cpu_affinity")
    original = process.cpu_affinity() if has_affinity else None
    if args.allowed_cpus is not None and (original is None or not set(args.allowed_cpus).issubset(original)):
        parser.error("Allowed CPUs must belong to the current process affinity mask")
    report = dict(
        complete=False, model=args.model, revision=args.revision, platform=platform.platform(),
        cpu=cpu_name(),
        system_physical_cores=psutil.cpu_count(logical=False), system_memory_bytes=psutil.virtual_memory().total,
        initial_available_memory_bytes=psutil.virtual_memory().available, requested_affinity=args.allowed_cpus,
        fingerprints=fingerprints, artifact_bytes=artifact.stat().st_size,
        prompt=prompt, prompt_token_ids=config["input_ids"], requested_new_tokens=16, rows=[],
    )
    command = [*child, "-m", "leaf.cli", "run", model, "--offline", "--bits", "32", "--fp32-math",
               "accurate", "--raw", "--prompt", prompt, "--max-tokens", "16"]
    if args.revision:
        command.extend(["--revision", args.revision])

    def run(workers, name):
        metrics = output / f"{name}.json"
        on_ac = plugged_in()
        measured = run_process([*command, "--threads", str(workers), "--metrics", str(metrics)],
                               environment, output, require_package_check=False)
        native = json.loads(metrics.read_bytes())
        if ("native artifact..." in measured["stderr"] or native["threads"] != workers or
                native["weight_bits"] != 32 or native["activation_bits"] != 32 or native["fp32_math"] != "accurate" or
                native["artifact_bytes"] != report["artifact_bytes"] or not native["generated_tokens"]):
            raise RuntimeError("The measured request did not use the prepared FP32 configuration")
        if on_ac != plugged_in():
            raise RuntimeError("Laptop AC state changed during the request")
        row = {field: measured[field] if field in measured else native[field] for field in TIMINGS}
        if any(value is None or value <= 0 for value in row.values()) or native["peak_rss_bytes"] <= 0:
            raise RuntimeError("A timing or decoder memory measurement is missing")
        return dict(row, peak_rss_bytes=native["peak_rss_bytes"], generated_tokens=native["generated_tokens"],
                    avx2=native["avx2"], threads=workers, plugged_in=on_ac)

    try:
        if args.allowed_cpus is not None:
            process.cpu_affinity(args.allowed_cpus)
        report["effective_affinity"] = process.cpu_affinity() if has_affinity else None
        with keep_awake():
            warmups = [run(workers, f"warmup-{workers}") for workers in args.threads]
            expected = warmups[0]["generated_tokens"]
            states = {row["plugged_in"] for row in warmups}
            for round_number in range(args.runs):
                order = args.threads if round_number % 2 == 0 else reversed(args.threads)
                for workers in order:
                    row = run(workers, f"round-{round_number + 1}-threads-{workers}")
                    report["rows"].append(dict(row, round=round_number + 1))
                    write_record(report_path, report)
            if any(row["generated_tokens"] != expected for row in [*warmups, *report["rows"]]):
                raise RuntimeError("Greedy generation differed between requests")
            states.update(row["plugged_in"] for row in report["rows"])
            if len(states) != 1:
                raise RuntimeError("Laptop AC state changed between requests")
        report["summary"] = {str(workers): summarize([row for row in report["rows"] if row["threads"] == workers])
                             for workers in args.threads}
        if fingerprints != {key: digest(path) for key, path in (("binary", binary), ("artifact", artifact))}:
            raise RuntimeError("Decoder or artifact changed during the benchmark")
        report["complete"] = True
        print(json.dumps(report["summary"], indent=2))
    finally:
        if args.allowed_cpus is not None:
            process.cpu_affinity(original)
        write_record(report_path, report)


if __name__ == "__main__":
    main()
