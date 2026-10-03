"""Trained decoder parity, held-out text quality, generation and CPU latency.

Baseline loading and native evaluation run in separate processes so mapped
native weights do not compete with a resident multi-gigabyte PyTorch model.
Quality thresholds are fixed before measurement; failing formats stay opt-in.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.decoder_validation import (MAX_LATENCY_P90_P10_RATIO, MIN_LATENCY_SAMPLES,
                                      benchmark_stability, comparison_stability,
                                      latency_median, quality, run_native)
from tools.datasets import token_prefix
from leaf.affinity import pin_cpu
from leaf.power import keep_awake

GATES = {
    "32": {"min_next_token_agreement": 0.999, "max_perplexity_ratio": 1.001},
    "8": {"min_next_token_agreement": 0.95, "max_perplexity_ratio": 1.02},
    "4": {"min_next_token_agreement": 0.90, "max_perplexity_ratio": 1.05},
}


LATENCY_WORKLOAD = {
    "format": "leaf-decoder-latency-workload-v1",
    "batch_size": 1,
    "prefill": "all_prefix_tokens",
    "decode": "one_token_with_prefill_kv_cache",
    "logits": "last_token_only",
    "use_cache": True,
}


def latency_workload_matches(record) -> bool:
    """Strict canonical comparison; bool/int lookalikes are not equivalent."""
    if not isinstance(record, dict):
        return False
    try:
        return (json.dumps(record.get("latency_workload"), sort_keys=True, separators=(",", ":")) ==
                json.dumps(LATENCY_WORKLOAD, sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError):
        return False


def native_policy() -> dict:
    return {name: bool(os.environ.get(name)) for name in
            ("LEAF_DISABLE_VNNI", "LEAF_EXPERIMENTAL_FLOAT_TILES", "LEAF_EXPERIMENTAL_FLOAT_GEMV")}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def validate_baseline_provenance(args, record: dict) -> None:
    """Reject stale baseline quality before loading a model or running native code."""
    if not isinstance(record, dict):
        raise ValueError("Baseline record must be an object")
    if (record.get("model_config_sha256") != digest(args.model / "config.json") or
            record.get("source_weight_sha256") != {p.name: digest(p) for p in sorted(args.model.glob("*.safetensors"))}):
        raise ValueError("Baseline provenance does not match the requested model")
    if record.get("dataset_sha256") != digest(args.dataset):
        raise ValueError("Baseline provenance does not match the requested held-out dataset")
    for name, value in (("threads", args.threads), ("pinned_cpu", args.cpu),
                        ("sequence_length", args.sequence_length), ("blocks", args.blocks),
                        ("text_column", args.text_column)):
        if record.get(name) != value:
            raise ValueError(f"Native comparison settings differ from baseline: {name}")


def quality_cache_hashes(workdir: Path) -> dict:
    return {name: digest(workdir / name) for name in ("reference.npy", "tokens.json")}


def load_baseline_cache(args) -> tuple[dict, dict]:
    if not (args.workdir / "pytorch.json").is_file():
        raise ValueError("Baseline cache missing pytorch.json; run --phase baseline first")
    record = json.loads((args.workdir / "pytorch.json").read_text(encoding="utf-8"))
    validate_baseline_provenance(args, record)
    for name in ("reference.npy", "tokens.json"):
        if not (args.workdir / name).is_file():
            raise ValueError(f"Baseline cache missing {name}; run --phase baseline first")
    data = json.loads((args.workdir / "tokens.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Baseline tokens must be an object")
    sequences = data.get("sequences")

    def token_list(values):
        return (isinstance(values, list) and bool(values) and
                all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values))

    if (not isinstance(sequences, list) or len(sequences) != args.blocks or
            any(not token_list(ids) or len(ids) != args.sequence_length for ids in sequences) or
            not token_list(data.get("benchmark_ids")) or len(data["benchmark_ids"]) < 2 or
            not token_list(data.get("generation_ids")) or
            data.get("generated_tokens") != record.get("generated_tokens")):
        raise ValueError("Baseline token cache does not match its quality record")
    hashes = quality_cache_hashes(args.workdir)
    if "quality_cache_sha256" in record and record["quality_cache_sha256"] != hashes:
        raise ValueError("Baseline quality cache hashes changed; rerun full validation")
    # Older completed runs predate cache hashes. Seal them on their first timing
    # refresh; model/dataset/settings and embedded quality identity still match.
    record["quality_cache_sha256"] = hashes
    record.setdefault("quality_measured_at_utc", record.get("measured_at_utc"))
    return data, record


def baseline_quality_identity(record: dict) -> dict:
    """Fields that a timing-only phase must never replace or reinterpret."""
    names = ("model_config_sha256", "source_weight_sha256", "dataset_sha256", "text_column",
             "sequence_length", "blocks", "threads", "pinned_cpu", "generation_prompt",
             "generated_tokens", "generated_text", "quality_gate_thresholds")
    return {**{name: record.get(name) for name in names},
            "quality_measured_at_utc": record.get("quality_measured_at_utc", record.get("measured_at_utc"))}


def decoder_plan_identity(args) -> tuple[dict, str]:
    from tools.decoder_plan import make_plan, validate_plan, validate_snapshot_generation
    filename = getattr(args, "plan", None)
    plan = json.loads(filename.read_text(encoding="utf-8")) if filename else make_plan(
        json.loads((args.model / "config.json").read_text(encoding="utf-8")))
    validate_plan(plan)
    validate_snapshot_generation(args.model, plan)
    identity = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return plan, identity


def calibration_identity(filename: Path, pytorch: dict) -> tuple[dict, str]:
    """Validate model/split provenance even when a quantized artifact exists."""
    try:
        with np.load(filename, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["__metadata__"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Calibration has invalid provenance metadata") from error
    def sha256(value):
        return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)

    weights = metadata.get("source_weight_sha256") if isinstance(metadata, dict) else None
    if (not isinstance(metadata, dict) or metadata.get("format") != "leaf-activation-calibration-v1" or
            not sha256(metadata.get("model_config_sha256")) or not isinstance(weights, dict) or not weights or
            any(not isinstance(name, str) or not name or not sha256(value) for name, value in weights.items()) or
            metadata.get("model_config_sha256") != pytorch.get("model_config_sha256") or
            metadata.get("source_weight_sha256") != pytorch.get("source_weight_sha256")):
        raise ValueError("Calibration provenance does not match source model")
    split = metadata.get("dataset_sha256")
    if not sha256(split):
        raise ValueError("Calibration provenance is missing its dataset hash")
    if split == pytorch.get("dataset_sha256"):
        raise ValueError("Calibration and held-out datasets must be separate splits")
    return metadata, digest(filename)


def export_request(pytorch: dict, plan: dict, plan_sha256: str, bits: int, activation_bits: int,
                   kept_fp32, int8_group_size: int, calibration_sha256: str | None = None) -> dict:
    from tools.export_decoder import validate_fp32_tensors
    kept = sorted(validate_fp32_tensors(plan, kept_fp32))
    if (bits not in (32, 8, 4) or activation_bits not in (32, 8) or
            not isinstance(int8_group_size, int) or isinstance(int8_group_size, bool) or
            int8_group_size < 0 or int8_group_size % 2):
        raise ValueError("Invalid native export precision policy")
    return {"format": "leaf-decoder-export-request-v1",
            "model_config_sha256": pytorch["model_config_sha256"],
            "source_weight_sha256": pytorch["source_weight_sha256"],
            "plan_sha256": plan_sha256, "weight_bits": bits, "activation_bits": activation_bits,
            "keep_fp32_tensors": kept, "int8_group_size": int8_group_size if bits == 8 else None,
            "int4_group_size": 64 if bits == 4 else None,
            "calibration_sha256": calibration_sha256,
            "smoothing_alpha": 0.5 if calibration_sha256 else None}


def cached_export_matches(artifact: Path, request: dict) -> bool:
    sidecar = artifact.with_suffix(".provenance.json")
    if not artifact.is_file() or not sidecar.is_file():
        return False
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        return (isinstance(record, dict) and record.get("request") == request and
                record.get("artifact_sha256") == digest(artifact))
    except (OSError, ValueError, TypeError):
        return False


def write_export_provenance(artifact: Path, request: dict) -> None:
    sidecar = artifact.with_suffix(".provenance.json")
    temporary = sidecar.with_suffix(sidecar.suffix + ".partial")
    try:
        write_record(temporary, {"request": request, "artifact_sha256": digest(artifact)})
        temporary.replace(sidecar)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def measure_pytorch_latency(model, benchmark_ids: list[int], args) -> dict:
    try:
        parameter = inspect.signature(model.forward).parameters.get("logits_to_keep")
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Cannot verify PyTorch last-token logits support") from error
    if parameter is None or parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                    inspect.Parameter.KEYWORD_ONLY):
        raise ValueError("PyTorch forward must explicitly support logits_to_keep for matched latency")
    if len(benchmark_ids) < 2:
        raise ValueError("Latency requires a prefix and a cached final token")
    import torch

    def verify_logits(output):
        try:
            shape = tuple(output.logits.shape)
        except (AttributeError, TypeError) as error:
            raise ValueError("PyTorch latency must return last-token logits [1, 1, vocabulary]") from error
        configured_vocab = getattr(getattr(model, "config", None), "vocab_size", None)
        if (len(shape) != 3 or shape[0] != 1 or shape[1] != 1 or shape[2] <= 0 or
                (configured_vocab is not None and shape[2] != configured_vocab)):
            raise ValueError("PyTorch latency must return last-token logits [1, 1, vocabulary]")

    benchmarks = {}
    for implementation in ("eager", "sdpa"):
        model.set_attn_implementation(implementation)
        prefix = torch.tensor([benchmark_ids[:-1]])
        final = torch.tensor([[benchmark_ids[-1]]])
        prefill_times, decode_times = [], []
        with torch.inference_mode():
            for iteration in range(args.warmup + args.runs):
                start = time.perf_counter(); state = model(prefix, use_cache=True, logits_to_keep=1)
                prefill = (time.perf_counter() - start) * 1000
                verify_logits(state)
                if getattr(state, "past_key_values", None) is None:
                    raise ValueError("PyTorch latency must return a KV cache for cached decode")
                start = time.perf_counter(); decoded = model(final, past_key_values=state.past_key_values,
                                                             use_cache=True, logits_to_keep=1)
                decode = (time.perf_counter() - start) * 1000
                verify_logits(decoded)
                if iteration >= args.warmup:
                    prefill_times.append(prefill); decode_times.append(decode)
                # A new prefill must not retain a previous request's cache or
                # logits. Session reset is outside the timed native call too.
                del decoded, state
        benchmarks[implementation] = {"prefill_p50_ms": statistics.median(prefill_times),
                                      "decode_p50_ms": statistics.median(decode_times),
                                      "prefill_samples_ms": prefill_times, "decode_samples_ms": decode_times,
                                      "latency_workload": dict(LATENCY_WORKLOAD)}
        benchmarks[implementation]["stability"] = benchmark_stability(benchmarks[implementation])
    return benchmarks


def load_pytorch_model(args):
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                                dtype=torch.float32, attn_implementation="eager").eval()
    return model, time.perf_counter() - start


def apply_speed_gates(cases: dict, pytorch: dict) -> None:
    """Annotate measurements without hiding raw samples or unstable runs."""
    baseline_latencies = pytorch.get("latency", {}) if isinstance(pytorch, dict) else {}
    baseline_latencies = baseline_latencies if isinstance(baseline_latencies, dict) else {}
    fp32 = cases.get("32")

    def fastest(phase):
        values = [value for latency in baseline_latencies.values() if (value := latency_median(latency, phase)) is not None]
        return min(values) if values else None

    fastest_prefill, fastest_decode = fastest("prefill"), fastest("decode")
    fp32_latency = fp32.get("latency") if isinstance(fp32, dict) else None
    fp32_prefill, fp32_decode = latency_median(fp32_latency, "prefill"), latency_median(fp32_latency, "decode")
    sdpa_decode = latency_median(baseline_latencies.get("sdpa"), "decode")
    for latency in baseline_latencies.values():
        if isinstance(latency, dict):
            latency["stability"] = benchmark_stability(latency)
    for case in cases.values():
        latency = case.get("latency")
        if isinstance(latency, dict):
            latency["stability"] = benchmark_stability(latency)
        stability = comparison_stability(latency, fp32_latency, baseline_latencies)
        case["latency_stability"] = stability
        workload_match = (latency_workload_matches(pytorch) and bool(baseline_latencies) and
                          all(latency_workload_matches(item) for item in baseline_latencies.values()) and
                          latency_workload_matches(fp32_latency) and latency_workload_matches(latency))
        case["latency_workload_match"] = {"passed": workload_match, "expected": dict(LATENCY_WORKLOAD)}
        prefill, decode = latency_median(latency, "prefill"), latency_median(latency, "decode")
        case["decode_speedup_vs_pytorch_sdpa"] = sdpa_decode / decode if sdpa_decode and decode else None
        case["decode_speedup_vs_fastest_pytorch"] = fastest_decode / decode if fastest_decode and decode else None
        case["prefill_speedup_vs_fastest_pytorch"] = fastest_prefill / prefill if fastest_prefill and prefill else None
        case["decode_speedup_vs_leaf_fp32"] = fp32_decode / decode if fp32_decode and decode else None
        case["prefill_speedup_vs_leaf_fp32"] = fp32_prefill / prefill if fp32_prefill and prefill else None
        case["eligible_for_automatic_selection"] = bool(
            case.get("quality_gate_passed") is True and workload_match and stability["passed"] and
            prefill and decode and fp32_prefill and fp32_decode and fastest_prefill and fastest_decode and
            decode < fp32_decode and prefill <= fp32_prefill * 1.02 and
            decode < fastest_decode * 0.98 and prefill <= fastest_prefill * 1.02)


def baseline(args):
    import psutil
    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    needed = args.blocks * args.sequence_length
    ids = token_prefix(tokenizer, args.dataset, needed, args.text_column)
    sequences = [ids[i:i + args.sequence_length] for i in range(0, needed, args.sequence_length)]
    prompt = "Explain in one sentence why careful measurement matters when optimizing software."
    if tokenizer.chat_template:
        generation_ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                        tokenize=True, add_generation_prompt=True)
    else:
        generation_ids = tokenizer.encode(prompt)
    if hasattr(generation_ids, "input_ids"):
        generation_ids = generation_ids.input_ids
    if generation_ids and isinstance(generation_ids[0], list):
        generation_ids = generation_ids[0]
    model, load_seconds = load_pytorch_model(args)
    references = []
    with torch.inference_mode():
        for index, sequence in enumerate(sequences):
            references.append(model(torch.tensor([sequence]), use_cache=False).logits[0].numpy().copy())
            print(f"PyTorch quality block {index + 1}/{len(sequences)}", flush=True)
        generation = model.generate(torch.tensor([generation_ids]), max_new_tokens=args.generate,
                                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated = generation[0, len(generation_ids):].tolist()
    reference = np.concatenate(references)
    np.save(args.workdir / "reference.npy", reference)
    data = {"sequences": sequences, "generation_ids": generation_ids, "generated_tokens": generated,
            "benchmark_ids": ids[:64]}
    (args.workdir / "tokens.json").write_text(json.dumps(data), encoding="utf-8")
    quality_timestamp = utc_now()
    benchmarks = measure_pytorch_latency(model, data["benchmark_ids"], args)
    latency_timestamp = utc_now()
    record = {"measured_at_utc": latency_timestamp, "quality_measured_at_utc": quality_timestamp,
              "latency_measured_at_utc": latency_timestamp,
              "latency_workload": dict(LATENCY_WORKLOAD),
              "quality_cache_sha256": quality_cache_hashes(args.workdir), "model": args.model.name,
              "model_config_sha256": digest(args.model / "config.json"),
              "source_weight_sha256": {p.name: digest(p) for p in sorted(args.model.glob("*.safetensors"))},
              "dataset": args.dataset.name + ", contiguous blocks from first nonempty text",
              "dataset_source": args.dataset_source or "User-supplied local dataset",
              "dataset_sha256": digest(args.dataset), "text_column": args.text_column,
              "sequence_length": args.sequence_length,
              "blocks": args.blocks, "threads": args.threads, "warmup": args.warmup, "runs": args.runs,
              "pinned_cpu": args.cpu,
              "platform": platform.platform(), "machine": platform.machine(), "cpu": platform.processor(),
              "logical_cpus": psutil.cpu_count(), "physical_memory_bytes": psutil.virtual_memory().total,
              "torch": torch.__version__, "load_seconds": load_seconds,
              "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
              "resident_bytes": psutil.Process().memory_info().rss,
              "generation_prompt": prompt, "generated_tokens": generated,
              "generated_text": tokenizer.decode(generated), "latency": benchmarks,
              "quality_gate_thresholds": GATES}
    write_record(args.workdir / "pytorch.json", record)


def baseline_latency(args):
    """Retime PyTorch without regenerating reference logits or token quality."""
    data, record = load_baseline_cache(args)
    model, load_seconds = load_pytorch_model(args)
    record["latency"] = measure_pytorch_latency(model, data["benchmark_ids"], args)
    record.update(latency_measured_at_utc=utc_now(), warmup=args.warmup, runs=args.runs,
                  latency_load_seconds=load_seconds, latency_workload=dict(LATENCY_WORKLOAD))
    write_record(args.workdir / "pytorch.json", record)


def native(args):
    data, pytorch = load_baseline_cache(args)
    executable_sha256 = digest(args.executable)
    reference = np.load(args.workdir / "reference.npy")
    cases = {}
    candidates = [(str(bits), bits, 32, None, [], 0) for bits in args.bits]
    plan, plan_sha256 = decoder_plan_identity(args)
    metadata = None
    calibration_sha256 = None
    if args.calibration:
        metadata, calibration_sha256 = calibration_identity(args.calibration, pytorch)
        candidates.append(("8-smooth", 8, 8, args.calibration, [], 0))
    if getattr(args, "protected_int8", False):
        from tools.export_decoder import protected_int8_tensors
        candidates.append(("8-protected", 8, 32, None, protected_int8_tensors(plan), 0))
    if getattr(args, "grouped_int8", False):
        candidates.append(("8-grouped", 8, 8, None, [], 64))
    if getattr(args, "grouped_int8_smooth", False):
        if not args.calibration:
            raise ValueError("Grouped smoothed INT8 requires --calibration")
        candidates.append(("8-grouped-smooth", 8, 8, args.calibration, [], 64))
    for key, bits, activation_bits, calibration, kept_fp32, int8_group_size in candidates:
        artifact = args.workdir / f"decoder-{key}.leaf"
        request = export_request(pytorch, plan, plan_sha256, bits, activation_bits, kept_fp32,
                                 int8_group_size, calibration_sha256 if calibration else None)
        if not cached_export_matches(artifact, request):
            command = [sys.executable, str(ROOT / "tools/export_decoder.py"), "--model", str(args.model),
                       "--output", str(artifact), "--bits", str(bits)]
            if args.plan:
                command += ["--plan", str(args.plan)]
            if calibration:
                command += ["--calibration", str(calibration)]
            if int8_group_size:
                command += ["--int8-group-size", str(int8_group_size)]
            if kept_fp32:
                command += ["--keep-fp32-tensors", *kept_fp32]
            subprocess.run(command, check=True)
            write_export_provenance(artifact, request)
        artifact_sha256 = digest(artifact)
        print(f"Native {key} held-out quality", flush=True)
        actual, execution = run_native(args.executable, artifact, data["sequences"], threads=args.threads,
                                        activation_bits=activation_bits)
        actual = actual.reshape(reference.shape)
        measured = quality(actual, reference, data["sequences"])
        if bits == 32:
            np.testing.assert_allclose(actual, reference, atol=2e-3, rtol=2e-3)
            chunked, _ = run_native(args.executable, artifact, [data["sequences"][0]],
                                    mode="chunked", threads=args.threads, chunk=16)
            np.testing.assert_allclose(chunked.reshape(-1, reference.shape[-1]),
                                       reference[:args.sequence_length], atol=2e-3, rtol=2e-3)
        _, generation = run_native(args.executable, artifact, [data["generation_ids"]], mode="generate",
                                    threads=args.threads, generate=args.generate, activation_bits=activation_bits)
        generation_match = generation["generated_tokens"] == data["generated_tokens"]
        if bits == 32 and not generation_match:
            raise AssertionError("FP32 greedy generation differs from PyTorch")
        quality_timestamp = utc_now()
        _, latency = run_native(args.executable, artifact, [data["benchmark_ids"]], mode="bench",
                                 threads=args.threads, runs=args.runs, warmup=args.warmup, activation_bits=activation_bits)
        latency_timestamp = utc_now()
        latency.update(warmup=args.warmup, runs=args.runs, latency_workload=dict(LATENCY_WORKLOAD))
        gate = GATES[str(bits)]
        passed = (measured["next_token_agreement"] >= gate["min_next_token_agreement"] and
                  measured["perplexity_ratio"] <= gate["max_perplexity_ratio"])
        cases[key] = {"weight_bits": bits, "activation_bits": activation_bits,
                            "export_provenance": request,
                            "keep_fp32_tensors": sorted(kept_fp32),
                            "int8_group_size": int8_group_size if bits == 8 else None,
                            "artifact": artifact.name, "artifact_sha256": artifact_sha256,
                            "quality": measured, "quality_gate_passed": passed,
                            "generation_exact_match": generation_match,
                            "generated_tokens": generation["generated_tokens"],
                            "quality_measured_at_utc": quality_timestamp,
                            "latency_measured_at_utc": latency_timestamp,
                            "execution": execution, "latency": latency}
        print(json.dumps({"candidate": key, "quality": measured, "quality_gate_passed": passed,
                          "prefill_ms": latency["prefill_p50_ms"], "decode_ms": latency["decode_p50_ms"]}), flush=True)
        del actual
    if (digest(args.executable) != executable_sha256 or any(
            digest(args.workdir / case["artifact"]) != case["artifact_sha256"] for case in cases.values())):
        raise ValueError("Native executable or artifacts changed during validation; rerun full validation")
    apply_speed_gates(cases, pytorch)
    result = {"benchmark": "trained-full-decoder-quality-and-latency", "pytorch": pytorch,
              "measured_at_utc": utc_now(),
              "quality_measured_at_utc": max(case["quality_measured_at_utc"] for case in cases.values()),
              "latency_measured_at_utc": max(case["latency_measured_at_utc"] for case in cases.values()),
              "native_executable_sha256": executable_sha256,
              "native_policy": native_policy(),
              "latency_workload": dict(LATENCY_WORKLOAD),
              "latency_stability_gate": {"minimum_samples": MIN_LATENCY_SAMPLES,
                                          "max_p90_p10_ratio": MAX_LATENCY_P90_P10_RATIO},
              "native": cases, "scope": "Complete trained model; held-out subset perplexity is not whole-corpus perplexity."}
    if metadata:
        result["calibration"] = dict(metadata, smoothing_alpha=0.5)
    write_record(args.output, result)
    if not cases.get("32", {}).get("quality_gate_passed", True):
        raise AssertionError("FP32 quality gate failed")


def native_latency(args):
    """Retime unchanged validated artifacts; quality is never rerun or relabeled."""
    data, pytorch = load_baseline_cache(args)
    if not args.output.is_file():
        raise ValueError("Native validation record missing; run full validation first")
    result = json.loads(args.output.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or not isinstance(result.get("pytorch"), dict):
        raise ValueError("Native validation record has no baseline quality provenance")
    prior_baseline = result["pytorch"]
    if baseline_quality_identity(prior_baseline) != baseline_quality_identity(pytorch):
        raise ValueError("Native record baseline quality provenance differs from cached baseline")
    if (prior_baseline.get("quality_cache_sha256", pytorch["quality_cache_sha256"]) !=
            pytorch["quality_cache_sha256"]):
        raise ValueError("Native record baseline quality cache hashes differ")
    if prior_baseline.get("quality_gate_thresholds") != GATES:
        raise ValueError("Quality gate thresholds changed; rerun full validation")
    if result.get("native_executable_sha256") != digest(args.executable):
        raise ValueError("Native executable changed; rerun full validation")
    default_policy = {name: False for name in native_policy()}
    prior_policy = result.get("native_policy", default_policy)
    if (not isinstance(prior_policy, dict) or set(prior_policy) != set(default_policy) or
            any(not isinstance(value, bool) for value in prior_policy.values()) or prior_policy != native_policy()):
        raise ValueError("Native ISA/experimental policy changed; rerun full validation")
    cases = result.get("native")
    if not isinstance(cases, dict) or not cases:
        raise ValueError("Native validation record has no measured candidates")
    artifacts = {}
    plan, plan_sha256 = decoder_plan_identity(args)
    calibration_sha256 = None
    if getattr(args, "calibration", None):
        _, calibration_sha256 = calibration_identity(args.calibration, pytorch)
    # Validate every candidate before benchmarking any candidate. A timing-only
    # refresh cannot export weights, introduce a new configuration or bless a
    # changed executable/artifact using quality measured for another binary.
    for key, case in cases.items():
        if (not isinstance(case, dict) or not isinstance(case.get("quality"), dict) or
                not isinstance(case.get("quality_gate_passed"), bool) or
                case.get("weight_bits") not in (32, 8, 4) or
                case.get("activation_bits") not in (32, 8) or
                not isinstance(case.get("artifact"), str)):
            raise ValueError(f"Native candidate {key} has incomplete quality provenance")
        artifact = args.workdir / case["artifact"]
        if (artifact.resolve().parent != args.workdir.resolve() or not artifact.is_file() or
                case.get("artifact_sha256") != digest(artifact)):
            raise ValueError(f"Native artifact {key} changed; rerun full validation")
        prior_request = case.get("export_provenance")
        if not isinstance(prior_request, dict):
            raise ValueError(f"Native export provenance for {key} is missing; rerun --phase native")
        uses_calibration = prior_request.get("calibration_sha256") is not None
        if uses_calibration and calibration_sha256 is None:
            raise ValueError(f"Native candidate {key} requires its recorded --calibration for a timing refresh")
        request = export_request(pytorch, plan, plan_sha256, case["weight_bits"], case["activation_bits"],
                                 case.get("keep_fp32_tensors", []), case.get("int8_group_size") or 0,
                                 calibration_sha256 if uses_calibration else None)
        if prior_request != request:
            raise ValueError(f"Native export provenance for {key} changed; rerun --phase native")
        artifacts[key] = artifact
    original_quality_timestamp = result.get("quality_measured_at_utc", result.get("measured_at_utc"))
    for key, case in cases.items():
        print(f"Native {key} latency only", flush=True)
        _, latency = run_native(args.executable, artifacts[key], [data["benchmark_ids"]], mode="bench",
                                 threads=args.threads, runs=args.runs, warmup=args.warmup,
                                 activation_bits=case["activation_bits"])
        latency.update(warmup=args.warmup, runs=args.runs, latency_workload=dict(LATENCY_WORKLOAD))
        case["latency"] = latency
        case.setdefault("quality_measured_at_utc", original_quality_timestamp)
        case["latency_measured_at_utc"] = utc_now()
    if (digest(args.executable) != result["native_executable_sha256"] or any(
            digest(path) != cases[key]["artifact_sha256"] for key, path in artifacts.items())):
        raise ValueError("Native executable or artifacts changed during timing refresh; rerun full validation")
    apply_speed_gates(cases, pytorch)
    result["pytorch"] = pytorch
    result.update(quality_measured_at_utc=original_quality_timestamp,
                  latency_measured_at_utc=utc_now(),
                  latency_workload=dict(LATENCY_WORKLOAD),
                  latency_stability_gate={"minimum_samples": MIN_LATENCY_SAMPLES,
                                          "max_p90_p10_ratio": MAX_LATENCY_P90_P10_RATIO})
    result["measured_at_utc"] = result["latency_measured_at_utc"]
    write_record(args.output, result)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("benchmark/data/wikitext2-test.parquet"))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--dataset-source",
                        help="provenance URL or description for the supplied held-out dataset")
    parser.add_argument("--workdir", type=Path, default=Path("build/trained_decoder"))
    parser.add_argument("--executable", type=Path, default=Path("build/leaf_decoder.exe"))
    parser.add_argument("--plan", type=Path, help="custom import tensor mapping and block semantics")
    parser.add_argument("--calibration", type=Path, help="independent channel calibration NPZ; adds W8A8 smoothing candidate")
    parser.add_argument("--protected-int8", action="store_true",
                        help="add W8A32 with canonical embedding, learned-position and output-head weights kept FP32")
    parser.add_argument("--grouped-int8", action="store_true",
                        help="add W8A8 with symmetric INT8 scales per 64 input channels")
    parser.add_argument("--grouped-int8-smooth", action="store_true",
                        help="add per-64-channel W8A8 with independent smoothing; requires --calibration")
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/trained_decoder.json"))
    parser.add_argument("--phase", choices=("all", "baseline", "native", "latency", "baseline-latency", "native-latency"),
                        default="all", help="latency reuses unchanged validated quality; runs baseline/native in separate processes")
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpu", type=int, help="optional logical CPU pin; comparisons use the same inherited affinity")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--generate", type=int, default=16)
    parser.add_argument("--bits", type=int, nargs="+", choices=(32, 8, 4), default=[32, 8, 4])
    args = parser.parse_args()
    pin_cpu(args.cpu)
    if min(args.blocks, args.sequence_length - 1, args.threads, args.runs, args.generate) <= 0 or args.warmup < 0:
        parser.error("invalid evaluation dimensions")
    if args.grouped_int8_smooth and not args.calibration:
        parser.error("grouped-int8-smooth requires calibration")
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.phase in ("all", "latency"):
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        phases = ("baseline", "native") if args.phase == "all" else ("baseline-latency", "native-latency")
        for phase in phases:
            subprocess.run(command + ["--phase", phase], check=True)
    elif args.phase == "baseline":
        baseline(args)
    elif args.phase == "baseline-latency":
        baseline_latency(args)
    elif args.phase == "native-latency":
        native_latency(args)
    else:
        native(args)


def main():
    with keep_awake():
        _main()


if __name__ == "__main__":
    main()
