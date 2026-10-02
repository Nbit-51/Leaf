"""Check an installed Leaf wheel without compilers or heavy model libraries.

Runs an offline local model twice against existing trained PyTorch generation
references. Fresh-artifact-cache preparation and repeat execution are separate
observations, not statistically qualified latency benchmarks. External timing
includes Python startup; CLI timing starts inside the already running process.
"""
from __future__ import annotations

import argparse
import codecs
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time


MARKER = "LEAF_PACKAGE_CHECK "
CHILD_PROGRAM = r'''
import hashlib, json, os, pathlib, sys
payload = json.loads(sys.argv[1])
installed = pathlib.Path(payload["installed_dir"]).resolve()
sys.path.insert(0, str(installed))
from leaf import cli

def relative_module(name):
    module = sys.modules.get(name)
    if module is None:
        return None
    return str(pathlib.Path(module.__file__).resolve().relative_to(installed))

relative_module("leaf.cli")
cache = pathlib.Path(os.environ["LEAF_CACHE_DIR"])
runtime = cli.executable(cache).resolve()
bundled = (installed / "leaf" / "bin" / ("leaf_decoder.exe" if os.name == "nt" else "leaf_decoder")).resolve()
assert runtime == bundled, "Installed wheel did not select its bundled runtime"
runtime_sha = hashlib.sha256(runtime.read_bytes()).hexdigest()
snapshot = pathlib.Path(payload["model"]).resolve()
directory = cli.model_directory(snapshot, cache)
kernel_policy = {name: os.environ.get(name) for name in
                 ("LEAF_DISABLE_VNNI", "LEAF_DECODER_PROFILE", "LEAF_EXPERIMENTAL_FLOAT_TILES", "LEAF_EXPERIMENTAL_FLOAT_GEMV")}
assert all(value is None for value in kernel_policy.values()), "Installed smoke inherited a kernel/profiling override"
assert not any(name.startswith("LEAF_EXPERIMENTAL_") for name in os.environ), "Installed smoke inherited an experimental flag"
metadata = {"module": relative_module("leaf.cli"), "bundled_runtime_selected": True,
            "native_executable_sha256": runtime_sha, "artifact_preexisting": (directory / "decoder-32.leaf").is_file(),
            "runtime_environment_policy": kernel_policy}
if payload["phase"] == "probe":
    profile = json.loads(pathlib.Path(payload["baseline"]).read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "validation.json").write_text(json.dumps(profile), encoding="utf-8")
    selected = cli.select_configuration(directory, runtime, "auto", payload["threads"], "auto", payload["cpu"])
    assert selected == (32, 32, None), "Unvalidated installed runtime selected lower precision"
    metadata.update(profile_native_binary_mismatch=profile.get("native_executable_sha256") != runtime_sha,
                    automatic_weight_bits=selected[0], automatic_activation_bits=selected[1])
    code = 0
else:
    sys.argv = ["leaf", "run", str(snapshot), "--offline", "--prompt", payload["prompt"],
                "--max-tokens", str(payload["generate"]), "--threads", str(payload["threads"]),
                "--metrics", payload["metrics"]]
    if payload["cpu"] is not None:
        sys.argv += ["--cpu", str(payload["cpu"])]
    code = cli.main()
    metadata["export_module"] = relative_module("tools.export_decoder")
heavy = [name for name in ("torch", "transformers", "onnx", "pyarrow", "safetensors") if name in sys.modules]
assert not heavy, "Heavy model libraries imported: " + ", ".join(heavy)
metadata["heavy_model_libraries_loaded"] = heavy
print("LEAF_PACKAGE_CHECK " + json.dumps(metadata), file=sys.stderr, flush=True)
raise SystemExit(code)
'''


def digest(filename: Path) -> str:
    result = hashlib.sha256()
    with filename.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def isolated_environment(cache: Path) -> dict:
    environment = os.environ.copy()
    removed = {"LEAF_DECODER_BIN", "LEAF_DISABLE_VNNI", "LEAF_DECODER_PROFILE", "PYTHONPATH", "PYTHONHOME", "CXX"}
    # Smoke the shipped default path, independent of prior developer experiments.
    # Clear only the child copy; never mutate the caller's environment.
    for name in tuple(environment):
        if name in removed or name.startswith("LEAF_EXPERIMENTAL_"):
            environment.pop(name)
    environment.update(LEAF_CACHE_DIR=str(cache.resolve()), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    if os.name == "nt":
        system_root = environment.get("SystemRoot") or environment.get("SYSTEMROOT")
        if not system_root:
            raise ValueError("SystemRoot is required to isolate the Windows runtime PATH")
        environment["PATH"] = str(Path(system_root) / "System32")
    else:
        # Native launch uses an absolute path; no external command is needed.
        environment["PATH"] = ""
    return environment


def run_process(command: list[str], environment: dict, cwd: Path) -> dict:
    """Observe unbuffered output while draining stderr without pipe deadlocks."""
    start = time.perf_counter()
    process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, bufsize=0)
    error_chunks = []
    error_reader = threading.Thread(target=lambda: error_chunks.append(process.stderr.read()), daemon=True)
    error_reader.start()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    output_chunks, first_output_ms, first_visible_ms = [], None, None
    while chunk := process.stdout.read(1):
        observed_ms = (time.perf_counter() - start) * 1000
        if first_output_ms is None:
            first_output_ms = observed_ms
        decoded = decoder.decode(chunk)
        if first_visible_ms is None and any(not character.isspace() for character in decoded):
            first_visible_ms = observed_ms
        output_chunks.append(chunk)
    returncode = process.wait()
    error_reader.join()
    wall_ms = (time.perf_counter() - start) * 1000
    stderr = b"".join(error_chunks).decode("utf-8", errors="replace")
    if returncode:
        raise RuntimeError(f"Installed runtime process failed ({returncode}): {stderr.strip()}")
    probe_lines = [line[len(MARKER):] for line in stderr.splitlines() if line.startswith(MARKER)]
    if len(probe_lines) != 1:
        raise RuntimeError("Installed runtime did not produce exactly one package provenance check")
    return {"external_wall_ms": wall_ms, "external_first_stdout_byte_ms": first_output_ms,
            "external_first_visible_text_ms": first_visible_ms,
            "stdout": b"".join(output_chunks).decode("utf-8", errors="replace"),
            "preparation_message_seen": "Preparing 32-bit native artifact..." in stderr,
            "package_check": json.loads(probe_lines[0])}


def validate_reference(model: Path, record: dict) -> dict:
    baseline = record.get("pytorch", record)
    if not isinstance(baseline, dict):
        raise ValueError("Trained baseline must contain a PyTorch reference object")
    weights = {filename.name: digest(filename) for filename in sorted(model.glob("*.safetensors"))}
    if (not weights or baseline.get("source_weight_sha256") != weights or
            baseline.get("model_config_sha256") != digest(model / "config.json")):
        raise ValueError("Trained baseline provenance does not match the local model")
    tokens = baseline.get("generated_tokens")
    if (not isinstance(tokens, list) or not tokens or
            any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in tokens) or
            not isinstance(baseline.get("generation_prompt"), str) or not baseline["generation_prompt"]):
        raise ValueError("Trained baseline requires a prompt and nonempty generated token IDs")
    threads, cpu = baseline.get("threads"), baseline.get("pinned_cpu")
    if not isinstance(threads, int) or isinstance(threads, bool) or not 1 <= threads <= 64:
        raise ValueError("Trained baseline threads must be 1..64")
    if cpu is not None and (not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0):
        raise ValueError("Trained baseline CPU pin must be a nonnegative integer or null")
    return baseline


def child_command(payload: dict) -> list[str]:
    return [sys.executable, "-I", "-u", "-c", CHILD_PROGRAM, json.dumps(payload)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-dir", type=Path, default=Path("build/wheel-install"))
    parser.add_argument("--model", type=Path, default=Path("build/models/gpt2"))
    parser.add_argument("--baseline", type=Path, default=Path("benchmark/results/gpt2_full_decoder.json"))
    parser.add_argument("--wheel", type=Path, help="optional wheel file to fingerprint")
    parser.add_argument("--workdir", type=Path, default=Path("build/package-runtime"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/package_runtime.json"))
    args = parser.parse_args()
    installed, model, baseline_path = args.installed_dir.resolve(), args.model.resolve(), args.baseline.resolve()
    binary = installed / "leaf" / "bin" / ("leaf_decoder.exe" if os.name == "nt" else "leaf_decoder")
    if not (installed / "leaf" / "cli.py").is_file() or not binary.is_file():
        parser.error("--installed-dir must contain the installed Leaf package and bundled native decoder")
    if not model.is_dir() or not baseline_path.is_file():
        parser.error("A local model directory and previously measured trained baseline are required")
    source_record = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline = validate_reference(model, source_record)
    args.workdir.mkdir(parents=True, exist_ok=True)
    # A new, retained task directory gives the first run a genuinely empty
    # artifact cache without deleting user caches or old benchmark artifacts.
    run_directory = Path(tempfile.mkdtemp(prefix="installed-", dir=args.workdir.resolve()))
    working_directory = run_directory / "workdir"
    working_directory.mkdir()
    cache = run_directory / "cache"
    environment = isolated_environment(cache)
    payload = {"installed_dir": str(installed), "model": str(model), "baseline": str(baseline_path),
               "threads": baseline["threads"], "cpu": baseline.get("pinned_cpu"),
               "prompt": baseline["generation_prompt"], "generate": len(baseline["generated_tokens"]),
               "phase": "probe"}
    checked = run_process(child_command(payload), environment, working_directory)["package_check"]
    if checked["artifact_preexisting"] or checked["heavy_model_libraries_loaded"]:
        raise AssertionError("Installed package probe was not isolated")
    runs = []
    for index, label in enumerate(("first_preparation", "cached_repeat")):
        metrics_path = run_directory / f"{label}.json"
        payload.update(phase="run", metrics=str(metrics_path))
        measured = run_process(child_command(payload), environment, working_directory)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("generated_tokens") != baseline["generated_tokens"]:
            raise AssertionError(f"{label}: installed native greedy tokens differ from the trained PyTorch reference")
        if metrics.get("weight_bits") != 32 or metrics.get("activation_bits") != 32:
            raise AssertionError("Unvalidated package runtime did not remain FP32")
        if measured["preparation_message_seen"] != (index == 0):
            raise AssertionError("First preparation and cached repeat were not distinguished")
        if measured["package_check"]["artifact_preexisting"] != (index == 1):
            raise AssertionError("Artifact cache state did not match the expected execution stage")
        measured.update(stage=label, generated_tokens_match_pytorch=True, cli_metrics=metrics)
        runs.append(measured)
        print(f"Installed package {label}: exact greedy tokens, FP32 fallback, no Torch", flush=True)
    result = {"benchmark": "installed-package-offline-runtime-smoke",
              "measured_at_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "cpu": platform.processor(), "threads": baseline["threads"], "pinned_cpu": baseline.get("pinned_cpu"),
              "model": model.name, "model_config_sha256": baseline["model_config_sha256"],
              "source_weight_sha256": baseline["source_weight_sha256"],
              "reference_record_sha256": digest(baseline_path), "package_check": checked,
              "compiler_path_available": False, "runtime_path_scope": "Windows System32 only" if os.name == "nt" else "empty PATH",
              "fresh_artifact_cache": True, "operating_system_file_cache_flushed": False,
              "timing_scope": "One observation per stage, not a speed gate. External timers include Python startup; CLI timers do not.",
              "runs": runs, "passed": True}
    if args.wheel:
        result.update(wheel=args.wheel.name, wheel_sha256=digest(args.wheel))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Installed package runtime record: {args.output}", flush=True)


if __name__ == "__main__":
    main()
