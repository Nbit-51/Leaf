"""One-command model preparation, native generation and gated optimization."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time

FORMAT = "leaf-decoder-v2"


def cache_root() -> Path:
    return Path(os.environ.get("LEAF_CACHE_DIR", Path.home() / ".cache" / "leaf"))


def source_engine() -> Path:
    checkout = Path(__file__).resolve().parents[1] / "engine"
    if (checkout / "src/decoder.cpp").is_file():
        return checkout
    installed = Path(sys.prefix) / "share/leaf/engine"
    if (installed / "src/decoder.cpp").is_file():
        return installed
    raise RuntimeError("Native source bundle is missing; reinstall leaf-cpu")


def executable(cache: Path) -> Path:
    configured = os.environ.get("LEAF_DECODER_BIN")
    if configured:
        candidate = Path(configured)
        if not candidate.is_file():
            raise FileNotFoundError("LEAF_DECODER_BIN does not point to an executable")
        return candidate.resolve()
    bundled = Path(__file__).resolve().parent / "bin" / ("leaf_decoder.exe" if os.name == "nt" else "leaf_decoder")
    if bundled.is_file():
        return bundled
    engine = source_engine()
    sources = [engine / name for name in ("src/decoder.cpp", "src/main_decoder.cpp", "src/kv_cache.cpp", "src/kernels/transformer.cpp")]
    headers = sorted((engine / "include").rglob("*.h"))
    digest = hashlib.sha256()
    digest.update((platform.system() + ":" + platform.machine()).encode())
    for source in sources + headers:
        digest.update(source.read_bytes())
    destination = cache / "native" / digest.hexdigest()[:16] / ("leaf_decoder.exe" if os.name == "nt" else "leaf_decoder")
    if destination.is_file():
        return destination
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise RuntimeError("Install a C++17 compiler (GCC or Clang), or set LEAF_DECODER_BIN to a supplied native binary")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    command = [compiler, "-std=c++17", "-O3", "-DNDEBUG", "-pthread", "-I", str(engine / "include"),
               *(str(source) for source in sources), "-o", str(temporary)]
    if os.name == "nt":
        command.extend(["-static", "-static-libgcc", "-static-libstdc++", "-lpsapi"])
    print("Building portable native runtime with CPU feature dispatch...", file=sys.stderr)
    subprocess.run(command, check=True)
    temporary.replace(destination)
    return destination


def resolve_model(model: str, cache: Path, offline: bool = False, plan: Path | None = None) -> Path:
    local = Path(model)
    if local.is_dir():
        return local.resolve()
    aliases = cache / "aliases.json"
    if aliases.is_file():
        model = json.loads(aliases.read_text()).get(model, model)
        if Path(model).is_dir():
            return Path(model).resolve()
    from huggingface_hub import snapshot_download
    from tools.decoder_plan import make_plan, validate_snapshot_generation
    # Check configuration/capabilities before fetching large weight shards.
    metadata = snapshot_download(model, cache_dir=cache / "hub", local_files_only=offline,
                                 allow_patterns=["config.json", "generation_config.json"])
    if plan:
        from tools.decoder_plan import validate_plan
        imported_plan = json.loads(plan.read_text())
        validate_plan(imported_plan)
    else:
        imported_plan = make_plan(json.loads((Path(metadata) / "config.json").read_text()))
    validate_snapshot_generation(Path(metadata), imported_plan)
    return Path(snapshot_download(model, cache_dir=cache / "hub", local_files_only=offline,
                                  allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "merges.txt", "vocab.json"]))


def model_directory(snapshot: Path, cache: Path, plan: Path | None = None) -> Path:
    digest = hashlib.sha256(FORMAT.encode())
    digest.update(str(snapshot.resolve()).encode("utf-8"))
    digest.update((snapshot / "config.json").read_bytes())
    generation = snapshot / "generation_config.json"
    digest.update(b"generation_config.json\0")
    digest.update(generation.read_bytes() if generation.is_file() else b"absent")
    for filename in sorted(snapshot.glob("*.safetensors")):
        stat = filename.stat()
        digest.update(f"{filename.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    if plan:
        digest.update(plan.read_bytes())
    return cache / "models" / digest.hexdigest()[:24]


def validate_model_generation(snapshot: Path, plan: Path | None = None) -> None:
    from tools.decoder_plan import make_plan, validate_plan, validate_snapshot_generation
    imported_plan = json.loads(plan.read_text(encoding="utf-8")) if plan else make_plan(
        json.loads((snapshot / "config.json").read_text(encoding="utf-8")))
    validate_plan(imported_plan)
    validate_snapshot_generation(snapshot, imported_plan)


def select_configuration(directory: Path, runtime: Path, requested: str, threads: int = 1,
                         activation_bits: str = "auto", cpu: int | None = None) -> tuple[int, int, Path | None]:
    fallback = (32 if requested == "auto" else int(requested),
                32 if activation_bits == "auto" else int(activation_bits), None)
    if requested != "auto":
        return fallback
    if (not isinstance(threads, int) or isinstance(threads, bool) or not 1 <= threads <= 64 or
            (cpu is not None and (not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0))):
        return fallback
    if any(os.environ.get(name) for name in
           ("LEAF_DISABLE_VNNI", "LEAF_EXPERIMENTAL_FLOAT_TILES", "LEAF_EXPERIMENTAL_FLOAT_GEMV")):
        return fallback  # a changed ISA policy invalidates measured speed claims
    profile = directory / "validation.json"
    if not profile.is_file():
        return fallback
    try:
        result = json.loads(profile.read_text())
        policy = result.get("native_policy")
        if "native_policy" in result and (not isinstance(policy, dict) or
                set(policy) != {"LEAF_DISABLE_VNNI", "LEAF_EXPERIMENTAL_FLOAT_TILES", "LEAF_EXPERIMENTAL_FLOAT_GEMV"} or
                any(value is not False for value in policy.values())):
            return fallback
        reference = result.get("pytorch", {})
        import math
        from tools.validate_decoder import GATES, apply_speed_gates, digest, latency_workload_matches
        if not (latency_workload_matches(result) and latency_workload_matches(reference)):
            return fallback  # legacy full-prefix logits are not a matched generation baseline
        if reference.get("platform") != platform.platform() or reference.get("cpu") != platform.processor():
            return fallback
        if reference.get("pinned_cpu") != cpu:
            return fallback
        reference_cpu = reference.get("pinned_cpu")
        if (not isinstance(reference.get("threads"), int) or reference.get("threads") != threads or
                isinstance(reference.get("threads"), bool) or
                (reference_cpu is not None and (not isinstance(reference_cpu, int) or isinstance(reference_cpu, bool))) or
                reference.get("quality_gate_thresholds") != GATES):
            return fallback
        if result.get("native_executable_sha256") != hashlib.sha256(runtime.read_bytes()).hexdigest():
            return fallback
        native_cases = result.get("native")
        baseline_latencies = reference.get("latency")
        if (not isinstance(native_cases, dict) or not isinstance(native_cases.get("32"), dict) or
                not isinstance(baseline_latencies, dict) or not baseline_latencies or
                not all(latency_workload_matches(item) for item in baseline_latencies.values()) or
                not latency_workload_matches(native_cases["32"].get("latency")) or
                native_cases["32"]["latency"].get("threads") != threads or
                isinstance(native_cases["32"]["latency"].get("threads"), bool) or
                not isinstance(native_cases["32"]["latency"].get("threads"), int) or
                native_cases["32"]["latency"].get("activation_bits") != 32):
            return fallback

        def quality_matches(case):
            bits = case.get("weight_bits")
            measured = case.get("quality")
            if (not isinstance(bits, int) or isinstance(bits, bool) or bits not in (32, 8, 4) or
                    not isinstance(measured, dict) or case.get("quality_gate_passed") is not True):
                return False
            agreement, perplexity = measured.get("next_token_agreement"), measured.get("perplexity_ratio")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                   not math.isfinite(value) for value in (agreement, perplexity)):
                return False
            gate = GATES[str(bits)]
            return (0 <= agreement <= 1 and agreement >= gate["min_next_token_agreement"] and
                    0 < perplexity <= gate["max_perplexity_ratio"])

        if not quality_matches(native_cases["32"]):
            return fallback
        # Retain conservative stored rejections, but never trust stored True
        # decisions instead of checking current thresholds and raw samples.
        originally_eligible = {key for key, case in native_cases.items() if isinstance(case, dict) and
                               case.get("eligible_for_automatic_selection") is True and quality_matches(case) and
                               isinstance(case.get("latency_stability"), dict) and
                               case["latency_stability"].get("passed") is True}
        checked_cases = {key: case for key, case in native_cases.items() if isinstance(case, dict)}
        apply_speed_gates(checked_cases, reference)
        eligible = [(case["latency"]["decode_p50_ms"], key, case) for key, case in checked_cases.items()
                    if key in originally_eligible and case.get("eligible_for_automatic_selection") is True and
                    latency_workload_matches(case.get("latency")) and
                    isinstance(case["latency"].get("threads"), int) and
                    not isinstance(case["latency"].get("threads"), bool) and
                    case["latency"]["threads"] == threads and
                    (activation_bits == "auto" or case["latency"].get("activation_bits", 32) == int(activation_bits))]
        for milliseconds, key, case in sorted(eligible):
            bits = case["weight_bits"] if "weight_bits" in case else int(key.split("-")[0])
            activations = case["latency"].get("activation_bits", 32)
            if (isinstance(milliseconds, bool) or not isinstance(milliseconds, (int, float)) or
                    not isinstance(bits, int) or isinstance(bits, bool) or bits not in (8, 4) or
                    not isinstance(activations, int) or isinstance(activations, bool) or activations not in (8, 32) or
                    not isinstance(case.get("activation_bits"), int) or isinstance(case.get("activation_bits"), bool) or
                    case["activation_bits"] != activations or
                    not math.isfinite(milliseconds) or milliseconds <= 0):
                continue
            artifact = directory / case.get("artifact", f"decoder-{key}.leaf")
            # A profile cannot silently select a replaced or external artifact.
            if (artifact.resolve().parent == directory.resolve() and artifact.is_file() and
                    case.get("artifact_sha256") == digest(artifact)):
                return bits, activations, artifact
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        pass  # malformed or stale profiles never enable lower precision
    return fallback


def prepare(snapshot: Path, directory: Path, bits: int, plan: Path | None = None) -> Path:
    artifact = directory / f"decoder-{bits}.leaf"
    if not artifact.is_file():
        from tools.export_decoder import export_decoder
        print(f"Preparing {bits}-bit native artifact...", file=sys.stderr)
        record = export_decoder(snapshot, artifact, bits, plan_path=plan)
        (directory / f"decoder-{bits}.json").write_text(json.dumps(record, indent=2) + "\n")
    return artifact


def encode_prompt(snapshot: Path, prompt: str, raw: bool = False):
    from tokenizers import Tokenizer
    from jinja2 import TemplateError
    from jinja2.sandbox import SandboxedEnvironment
    tokenizer_path = snapshot / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise RuntimeError("A tokenizer.json is required for the lightweight tokenizer runtime")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    config_path = snapshot / "tokenizer_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if config.get("legacy") is False and tokenizer.pre_tokenizer is None and tokenizer.normalizer is not None:
        # Older SentencePiece-style JSON applies Prepend to every fragment
        # split around an added special token. Nonlegacy tokenization prepends
        # only at the start of the full input, not again immediately after EOS.
        # Recognize the serialized backend structure, not a model/class name.
        normalization = json.loads(tokenizer.normalizer.__getstate__())
        steps = normalization.get("normalizers", [])
        marker = steps[0].get("prepend") if steps and isinstance(steps[0], dict) else None
        if (isinstance(marker, str) and len(marker) == 1 and normalization == {
                "type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": marker},
                    {"type": "Replace", "pattern": {"String": " "}, "content": marker}]}):
            from tokenizers import decoders, pre_tokenizers
            prefix = config.get("add_prefix_space")
            if prefix is not None and not isinstance(prefix, bool):
                raise ValueError("add_prefix_space must be boolean or null")
            prefix = prefix is not False
            tokenizer.normalizer = None
            tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
                replacement=marker, prepend_scheme="first" if prefix else "never", split=False)
            # Preserve unrelated/custom decoders. The canonical whitespace +
            # byte-fallback decoder must follow the same prefix-space policy.
            if tokenizer.decoder is not None:
                decoder = json.loads(tokenizer.decoder.__getstate__())
                parts = [{"type": "Replace", "pattern": {"String": marker}, "content": " "},
                         {"type": "ByteFallback"}, {"type": "Fuse"}]
                strip = {"type": "Strip", "content": " ", "start": 1, "stop": 0}
                if decoder in ({"type": "Sequence", "decoders": parts},
                               {"type": "Sequence", "decoders": parts + [strip]}):
                    replacements = [decoders.Replace(marker, " "), decoders.ByteFallback(), decoders.Fuse()]
                    if prefix:
                        replacements.append(decoders.Strip(content=" ", left=1, right=0))
                    tokenizer.decoder = decoders.Sequence(replacements)
    template = config.get("chat_template")
    if isinstance(template, list):
        template = next((item["template"] for item in template if item["name"] == "default"), None)
    if template and not raw:
        environment = SandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        def raise_exception(message):
            raise ValueError(message)
        environment.globals.update(raise_exception=raise_exception, strftime_now=lambda fmt: datetime.now().strftime(fmt))
        special = {name: config.get(name, "") for name in ("bos_token", "eos_token", "pad_token")}
        special = {name: value.get("content", "") if isinstance(value, dict) else value or ""
                   for name, value in special.items()}
        try:
            text = environment.from_string(template).render(messages=[{"role": "user", "content": prompt}],
                                                            add_generation_prompt=True, **special)
        except TemplateError as error:
            raise ValueError(f"Invalid chat template: {error}") from error
        ids = tokenizer.encode(text, add_special_tokens=False).ids
    else:
        ids = tokenizer.encode(prompt, add_special_tokens=True).ids
    if not ids:
        raise ValueError("Prompt produced no tokens")
    return tokenizer, ids


def run(args):
    from leaf.affinity import pin_cpu
    pin_cpu(args.cpu)
    start = time.perf_counter()
    cache = cache_root()
    snapshot = resolve_model(args.model, cache, args.offline, args.plan)
    validate_model_generation(snapshot, args.plan)
    runtime = executable(cache)
    directory = model_directory(snapshot, cache, args.plan)
    bits, activation_bits, selected = select_configuration(directory, runtime, args.bits, args.threads, args.activation_bits, args.cpu)
    artifact = selected or prepare(snapshot, directory, bits, args.plan)
    prompt = args.prompt if args.prompt is not None else input("Prompt: ")
    tokenizer, ids = encode_prompt(snapshot, prompt, args.raw)
    tokens = []
    rendered = ""
    with tempfile.TemporaryDirectory(prefix="leaf_run_") as temp:
        folder = Path(temp)
        request = folder / "tokens.bin"
        with request.open("wb") as stream:
            stream.write(struct.pack("<II", 1, len(ids)))
            stream.write(struct.pack(f"<{len(ids)}I", *ids))
        command = [str(runtime), str(artifact), str(request), str(folder / "logits.bin"),
                   str(folder / "metrics.json"), "generate", str(args.threads), "1", "0", str(args.max_tokens),
                   "0", "1", str(activation_bits)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        first_token_ms = None
        for line in process.stdout:
            if line.startswith("TOKEN "):
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - start) * 1000
                tokens.append(int(line.split()[1]))
                decoded = tokenizer.decode(tokens, skip_special_tokens=True).rstrip("\ufffd")
                if decoded.startswith(rendered):
                    print(decoded[len(rendered):], end="", flush=True)
                rendered = decoded
        error = process.stderr.read()
        if process.wait():
            raise RuntimeError(error.strip())
        metrics = json.loads((folder / "metrics.json").read_text())
        final_text = tokenizer.decode(tokens, skip_special_tokens=True)
        if final_text.startswith(rendered):
            print(final_text[len(rendered):], end="", flush=True)
        print()
    metrics.update(weight_bits=bits, time_to_first_token_ms=first_token_ms,
                   end_to_end_ms=(time.perf_counter() - start) * 1000)
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Leaf: {bits}-bit, {len(tokens)} tokens, decode p50 {metrics['decode_p50_ms']:.2f} ms", file=sys.stderr)


def optimize(args):
    cache = cache_root()
    snapshot = resolve_model(args.model, cache, args.offline, args.plan)
    validate_model_generation(snapshot, args.plan)
    directory = model_directory(snapshot, cache, args.plan)
    runtime = executable(cache)
    from tools.validate_decoder import main as validate
    original = sys.argv
    try:
        if args.calibration_dataset and args.calibration_dataset.resolve() == args.dataset.resolve():
            raise ValueError("Calibration and held-out datasets must be separate splits")
        calibration = directory / "calibration.npz" if args.calibration_dataset else None
        if calibration:
            command = [sys.executable, "-m", "tools.calibrate_decoder", "--model", str(snapshot),
                       "--dataset", str(args.calibration_dataset), "--text-column", args.text_column,
                       "--output", str(calibration), "--threads", str(args.threads)]
            if args.plan:
                command += ["--plan", str(args.plan)]
            subprocess.run(command, check=True)
        sys.argv = ["validate_decoder", "--model", str(snapshot), "--dataset", str(args.dataset),
                    "--workdir", str(directory), "--executable", str(runtime),
                    "--output", str(directory / "validation.json"), "--threads", str(args.threads),
                    "--text-column", args.text_column]
        for name, default in (("blocks", 8), ("sequence_length", 128), ("runs", 7),
                              ("warmup", 3), ("generate", 16)):
            sys.argv += ["--" + name.replace("_", "-"), str(getattr(args, name, default))]
        if args.dataset_source:
            sys.argv += ["--dataset-source", args.dataset_source]
        if getattr(args, "cpu", None) is not None:
            sys.argv += ["--cpu", str(args.cpu)]
        if args.plan:
            sys.argv += ["--plan", str(args.plan)]
        if calibration:
            sys.argv += ["--calibration", str(calibration)]
        for name in ("protected_int8", "grouped_int8", "grouped_int8_smooth"):
            if getattr(args, name, False):
                sys.argv += ["--" + name.replace("_", "-")]
        validate()
    finally:
        sys.argv = original
    print("Validation profile: " + str(directory / "validation.json"))


def main():
    parser = argparse.ArgumentParser(prog="leaf", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="prepare cached weights and generate with native CPU execution")
    runner.add_argument("model", help="local snapshot, Hugging Face repository ID, or registered alias")
    runner.add_argument("--prompt")
    runner.add_argument("--max-tokens", type=int, default=64)
    runner.add_argument("--threads", type=int, default=1)
    runner.add_argument("--cpu", type=int, help="optional logical CPU pin matching a validation profile")
    runner.add_argument("--bits", choices=("auto", "32", "8", "4"), default="auto")
    runner.add_argument("--activation-bits", choices=("auto", "8", "32"), default="auto",
                        help="dynamic INT8 or floating activations in quantized linear kernels")
    runner.add_argument("--raw", action="store_true", help="skip chat formatting")
    runner.add_argument("--offline", action="store_true")
    runner.add_argument("--plan", type=Path, help="custom decoder plan for a local snapshot")
    runner.add_argument("--metrics", type=Path)
    optimizer = commands.add_parser("optimize", help="validate quality and speed before selecting quantization")
    optimizer.add_argument("model")
    optimizer.add_argument("--dataset", type=Path, required=True, help="held-out TXT, JSON/JSONL, CSV or Parquet text dataset")
    optimizer.add_argument("--text-column", default="text")
    optimizer.add_argument("--dataset-source", help="dataset provenance URL or description")
    optimizer.add_argument("--calibration-dataset", type=Path, help="separate calibration split for channel-smoothed INT8")
    optimizer.add_argument("--plan", type=Path)
    optimizer.add_argument("--threads", type=int, default=1)
    optimizer.add_argument("--cpu", type=int)
    optimizer.add_argument("--offline", action="store_true")
    optimizer.add_argument("--blocks", type=int, default=8, help="held-out blocks used for quality gates")
    optimizer.add_argument("--sequence-length", type=int, default=128, help="tokens per held-out block")
    optimizer.add_argument("--runs", type=int, default=7, help="measured whole-model latency repetitions")
    optimizer.add_argument("--warmup", type=int, default=3)
    optimizer.add_argument("--generate", type=int, default=16, help="greedy tokens used for generation parity")
    optimizer.add_argument("--protected-int8", action="store_true", help="also test FP32 embeddings/output with INT8 linear weights")
    optimizer.add_argument("--grouped-int8", action="store_true", help="also test 64-column grouped INT8 weights/activations")
    optimizer.add_argument("--grouped-int8-smooth", action="store_true", help="also test calibrated grouped INT8; requires calibration data")
    alias = commands.add_parser("alias", help="register a short name for a snapshot or repository")
    alias.add_argument("name"); alias.add_argument("model")
    args = parser.parse_args()
    try:
        if args.command == "run":
            if args.max_tokens < 1 or not 1 <= args.threads <= 64:
                parser.error("max-tokens must be positive and threads must be 1..64")
            run(args)
        elif args.command == "optimize":
            if not 1 <= args.threads <= 64:
                parser.error("threads must be 1..64")
            if min(args.blocks, args.sequence_length - 1, args.runs, args.generate) <= 0 or args.warmup < 0:
                parser.error("quality dimensions and runs must be positive; sequence-length >=2 and warmup >=0")
            if args.grouped_int8_smooth and not args.calibration_dataset:
                parser.error("grouped-int8-smooth requires calibration-dataset")
            optimize(args)
        else:
            path = cache_root() / "aliases.json"
            aliases = json.loads(path.read_text()) if path.is_file() else {}
            aliases[args.name] = str(Path(args.model).resolve()) if Path(args.model).is_dir() else args.model
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(aliases, indent=2) + "\n")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Leaf: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
