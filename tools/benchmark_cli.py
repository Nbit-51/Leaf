"""Measure cached source-CLI startup and streaming against frozen native records.

No model export, native build, download, or profile qualification is performed.
The original validation profile and hashed artifacts are staged in a fresh,
retained build cache. Every measured command starts a new Python/native process.
Identity verification warms the OS file cache; it is never described as a cold
disk-cache benchmark. Installed wheels require their own binary qualification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.verify_package_runtime import digest, isolated_environment, run_process, validate_reference


CHILD_PROGRAM = r'''
import json, pathlib, sys
payload = json.loads(sys.argv[1])
root = pathlib.Path(payload["source_root"]).resolve()
sys.path.insert(0, str(root))
from leaf import cli
assert pathlib.Path(cli.__file__).resolve() == root / "leaf" / "cli.py", "Unexpected CLI package"
sys.argv = ["leaf", *payload["arguments"]]
code = cli.main()
heavy = [name for name in ("torch", "transformers", "onnx", "pyarrow", "safetensors") if name in sys.modules]
assert not heavy, "Heavy model libraries imported: " + ", ".join(heavy)
print("LEAF_PACKAGE_CHECK " + json.dumps({"source_cli": True, "heavy_model_libraries_loaded": heavy}),
      file=sys.stderr, flush=True)
raise SystemExit(code)
'''


def token_list(values) -> bool:
    return (isinstance(values, list) and bool(values) and
            all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values))


def artifact_name(case: dict, key: str) -> str:
    name = case.get("artifact", f"decoder-{key}.leaf")
    if not isinstance(name, str) or not name or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Validation artifacts must be plain filenames inside the artifact directory")
    return name


def stage_cache(model: Path, artifacts: Path, profile_path: Path, executable: Path,
                cache: Path, requested: list[str], raw: bool = False) -> tuple[dict, dict, dict]:
    """Verify original identities and stage only existing, frozen artifacts."""
    from leaf import cli

    if any(os.environ.get(name) for name in
           ("LEAF_DISABLE_VNNI", "LEAF_EXPERIMENTAL_FLOAT_TILES", "LEAF_EXPERIMENTAL_FLOAT_GEMV")):
        raise ValueError("Unset native ISA policy and experimental overrides: a changed policy cannot reuse this validation profile")
    profile_bytes = profile_path.read_bytes()
    profile = json.loads(profile_bytes)
    baseline = validate_reference(model, profile)
    if (baseline.get("platform") != platform.platform() or baseline.get("cpu") != platform.processor()):
        raise ValueError("Validation profile belongs to a different platform or CPU description")
    if profile.get("native_executable_sha256") != digest(executable):
        raise ValueError("Native executable does not match the frozen validation profile")
    cli.validate_model_generation(model)
    reference_path = artifacts / "tokens.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (not token_list(reference.get("generation_ids")) or
            reference.get("generated_tokens") != baseline["generated_tokens"]):
        raise ValueError("Generation token cache does not match the original trained reference")
    sealed_hash = baseline.get("quality_cache_sha256", {}).get("tokens.json")
    if sealed_hash is not None and sealed_hash != digest(reference_path):
        raise ValueError("Generation token cache SHA256 differs from the trained reference")
    _, prompt_ids = cli.encode_prompt(model, baseline["generation_prompt"], raw)
    if prompt_ids != reference["generation_ids"]:
        raise ValueError("CLI chat/raw prompt token IDs differ from the original generation reference")
    native = profile.get("native")
    if not isinstance(native, dict) or "32" not in native:
        raise ValueError("Validation profile requires a native FP32 reference")
    if artifact_name(native["32"], "32") != "decoder-32.leaf":
        raise ValueError("Cached explicit FP32 requires the CLI artifact name decoder-32.leaf")
    directory = cli.model_directory(model, cache)
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(profile_path, directory / "validation.json")
    if (directory / "validation.json").read_bytes() != profile_bytes:
        raise ValueError("Validation profile changed while staging the cache")
    needed = {"32": native["32"]}
    if "auto" in requested:
        needed.update({key: case for key, case in native.items()
                       if case.get("eligible_for_automatic_selection") is True})
    staged = {}
    for key, case in needed.items():
        name = artifact_name(case, key)
        source = artifacts / name
        if (source.resolve().parent != artifacts.resolve() or not source.is_file() or
                case.get("artifact_sha256") != digest(source)):
            raise ValueError(f"Missing, external, or changed frozen artifact: {name}")
        destination = directory / name
        try:
            os.link(source, destination)
            method = "hardlink"
        except OSError:
            shutil.copyfile(source, destination)
            method = "copy"
            if digest(destination) != case["artifact_sha256"]:
                raise ValueError(f"Staged artifact SHA256 mismatch: {name}")
        staged[name] = {"method": method, "bytes": destination.stat().st_size,
                        "sha256": case["artifact_sha256"]}
    cases = {}
    for requested_bits in requested:
        bits, activations, selected = cli.select_configuration(
            directory, executable, requested_bits, baseline["threads"], "auto", baseline.get("pinned_cpu"))
        if requested_bits == "auto" and selected is None:
            raise ValueError("No genuinely eligible automatic candidate matches the staged profile/runtime/settings")
        selected = selected or directory / "decoder-32.leaf"
        matches = [(key, case) for key, case in native.items()
                   if artifact_name(case, key) == selected.name and case.get("weight_bits", int(key.split("-")[0])) == bits
                   and case.get("latency", {}).get("activation_bits", 32) == activations]
        if len(matches) != 1:
            raise ValueError("Selected artifact has no unambiguous native validation case")
        key, case = matches[0]
        if (case.get("quality_gate_passed") is not True or not token_list(case.get("generated_tokens")) or
                len(case["generated_tokens"]) != len(baseline["generated_tokens"])):
            raise ValueError(f"Selected native case has no passing quality and exact generation reference: {key}")
        cases[requested_bits] = {"native_case": key, "weight_bits": bits, "activation_bits": activations,
                                 "artifact": selected.name, "artifact_sha256": case["artifact_sha256"],
                                 "expected_tokens": case["generated_tokens"]}
    provenance = {"profile_sha256": hashlib.sha256(profile_bytes).hexdigest(), "generation_reference_sha256": digest(reference_path),
                  "native_executable_sha256": profile["native_executable_sha256"],
                  "staged_artifacts": staged, "prompt_token_ids": prompt_ids,
                  "generation_config_sha256": digest(model / "generation_config.json")
                  if (model / "generation_config.json").is_file() else None}
    return baseline, cases, provenance


def child_command(model: Path, baseline: dict, case: str, metrics: Path, raw: bool = False) -> list[str]:
    arguments = ["run", str(model), "--offline", "--prompt", baseline["generation_prompt"],
                 "--max-tokens", str(len(baseline["generated_tokens"])), "--threads", str(baseline["threads"]),
                 "--bits", case, "--metrics", str(metrics)]
    if baseline.get("pinned_cpu") is not None:
        arguments.extend(["--cpu", str(baseline["pinned_cpu"])])
    if raw:
        arguments.append("--raw")
    payload = {"source_root": str(ROOT), "arguments": arguments}
    return [sys.executable, "-I", "-u", "-c", CHILD_PROGRAM, json.dumps(payload)]


def check_run(measured: dict, metrics: dict, case: dict, baseline: dict) -> dict:
    if measured["preparation_message_seen"]:
        raise AssertionError("A cached benchmark unexpectedly exported an artifact")
    if not measured["package_check"].get("source_cli") or measured["package_check"].get("heavy_model_libraries_loaded"):
        raise AssertionError("CLI execution did not use the isolated lightweight source package")
    if (metrics.get("weight_bits") != case["weight_bits"] or
            metrics.get("activation_bits") != case["activation_bits"] or
            metrics.get("threads") != baseline["threads"] or
            metrics.get("generated_tokens") != case["expected_tokens"]):
        raise AssertionError("CLI precision, settings, or generated tokens differ from the selected native reference")
    for name in ("load_ms", "prefill_p50_ms", "decode_p50_ms", "time_to_first_token_ms", "end_to_end_ms"):
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise AssertionError(f"Missing or invalid CLI/native timing: {name}")
    for name in ("external_wall_ms", "external_first_stdout_byte_ms", "external_first_visible_text_ms"):
        value = measured.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise AssertionError(f"Missing or invalid external timing: {name}")
    if not (measured["external_first_stdout_byte_ms"] <= measured["external_first_visible_text_ms"] <= measured["external_wall_ms"]):
        raise AssertionError("Invalid streaming timer ordering")
    measured.update(generated_tokens_match_selected_native=True,
                    generated_tokens_match_pytorch=metrics["generated_tokens"] == baseline["generated_tokens"],
                    cli_metrics=metrics,
                    frontend_and_startup_remainder_until_first_visible_ms=(measured["external_first_visible_text_ms"]
                        - metrics["load_ms"] - metrics["prefill_p50_ms"]))
    return measured


def summarize(runs: list[dict]) -> dict:
    from tools.decoder_validation import latency_stability

    fields = ("external_wall_ms", "external_first_stdout_byte_ms", "external_first_visible_text_ms")
    internal = ("load_ms", "prefill_p50_ms", "decode_p50_ms", "time_to_first_token_ms", "end_to_end_ms")
    summary = {}
    for name in fields + internal:
        samples = [run[name] if name in fields else run["cli_metrics"][name] for run in runs]
        summary[name] = {"p50_ms": statistics.median(samples), "samples_ms": samples,
                         "stability": latency_stability(samples)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="complete locally cached model snapshot")
    parser.add_argument("--profile", type=Path, required=True, help="unchanged trained native validation JSON")
    parser.add_argument("--artifacts-dir", type=Path, required=True, help="existing .leaf artifacts and tokens.json")
    parser.add_argument("--executable", type=Path, required=True, help="frozen binary named by the validation SHA256")
    parser.add_argument("--cases", nargs="+", choices=("32", "auto"), default=["32", "auto"])
    parser.add_argument("--runs", type=int, default=5, help="fresh process observations per case, minimum five")
    parser.add_argument("--raw", action="store_true", help="must reproduce the saved reference prompt token IDs")
    parser.add_argument("--workdir", type=Path, default=Path("build/cli-benchmark"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/cli_runtime.json"))
    args = parser.parse_args()
    if args.runs < 5 or len(set(args.cases)) != len(args.cases):
        parser.error("Require at least five runs and unique cases")
    model, profile, artifacts, executable = (path.resolve() for path in
        (args.model, args.profile, args.artifacts_dir, args.executable))
    if not model.is_dir() or not artifacts.is_dir() or not profile.is_file() or not executable.is_file():
        parser.error("Local model, artifact directory, original profile, and frozen executable are required")
    args.workdir.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="cached-", dir=args.workdir.resolve()))
    cache, cwd = folder / "cache", folder / "workdir"
    cwd.mkdir()
    baseline, cases, provenance = stage_cache(model, artifacts, profile, executable, cache, args.cases, args.raw)
    environment = isolated_environment(cache)
    environment["LEAF_DECODER_BIN"] = str(executable)
    observations = {name: [] for name in cases}
    execution_order = []
    from leaf.power import keep_awake
    with keep_awake():
        for repetition in range(args.runs):
            # Reverse the order on alternating repetitions; retain every sample.
            order = args.cases if repetition % 2 == 0 else list(reversed(args.cases))
            for name in order:
                metrics_path = folder / f"{name}-{repetition + 1}.json"
                measured = run_process(child_command(model, baseline, name, metrics_path, args.raw), environment, cwd)
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                measured = check_run(measured, metrics, cases[name], baseline)
                measured.update(repetition=repetition + 1, native_case=cases[name]["native_case"])
                observations[name].append(measured)
                execution_order.append({"case": name, "repetition": repetition + 1})
                print(f"Cached CLI {name} #{repetition + 1}: first visible {measured['external_first_visible_text_ms']:.1f} ms, "
                      f"total {measured['external_wall_ms']:.1f} ms, native reference tokens match", flush=True)
    summaries = {name: summarize(runs) for name, runs in observations.items()}
    result = {"benchmark": "cached-source-cli-full-command-streaming", "measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "cpu": platform.processor(), "model": model.name,
              "threads": baseline["threads"], "pinned_cpu": baseline.get("pinned_cpu"),
              "generation_prompt": baseline["generation_prompt"], "max_tokens": len(baseline["generated_tokens"]),
              "raw_prompt": args.raw, "model_config_sha256": baseline["model_config_sha256"],
              "source_weight_sha256": baseline["source_weight_sha256"], **provenance,
              "source_sha256": {name: digest(ROOT / name) for name in
                  ("leaf/cli.py", "leaf/affinity.py", "tools/decoder_plan.py", "tools/validate_decoder.py", "tools/benchmark_cli.py")},
              "runtime_scope": "Source CLI with explicit frozen validated executable, not an installed-wheel speed qualification",
              "cached_artifacts": True, "artifact_export_performed": False, "native_build_performed": False,
              "operating_system_file_cache_flushed": False, "identity_preflight_reads_weights_and_artifacts": True,
              "retained_run_directory": str(folder), "case_selection": cases, "execution_order": execution_order,
              "timing_scope": "External timers include Python startup and streaming. CLI TTFT excludes interpreter startup. "
                  "Native load_ms includes constructor/packing, not CLI profile/artifact hashing. Native prefill/decode are prompt execution. "
                  "The first-visible remainder combines startup, hashing, tokenizer, IPC, and rendering; it is not an isolated Python timer.",
              "runs": observations, "summary": summaries, "passed": True}
    if "32" in summaries and "auto" in summaries:
        result["auto_vs_fp32"] = {
            name: {"speedup": summaries["32"][name]["p50_ms"] / summaries["auto"][name]["p50_ms"],
                   "both_stable": all(summaries[case][name]["stability"]["passed"] for case in ("32", "auto"))}
            for name in ("external_first_visible_text_ms", "external_wall_ms")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Cached CLI record: {args.output}", flush=True)


if __name__ == "__main__":
    main()
