"""Verify the native decoder protocol across local and optional WSL builds.

Reduced random-weight models check implementation portability, not trained
model quality or performance. WSL and Windows on one device are two operating
systems on the same physical CPU, not a substitute for independent hardware.
Torch is used only to construct reference fixtures, never by either runtime.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.decoder_validation import quality
from tools.export_decoder import export_decoder, protected_int8_tensors


def digest(filename: Path) -> str:
    result = hashlib.sha256()
    with filename.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def wsl_path(filename: Path) -> str:
    """Map an absolute drive path without invoking a shell or WSL utility."""
    path = PureWindowsPath(str(filename))
    if not path.is_absolute() or not re.fullmatch(r"[A-Za-z]:", path.drive):
        raise ValueError("WSL protocol fixtures must use an absolute Windows drive path")
    return "/mnt/" + path.drive[0].lower() + "/" + "/".join(path.parts[1:])


def write_request(filename: Path, sequences: list[list[int]]) -> None:
    with filename.open("wb") as stream:
        stream.write(struct.pack("<I", len(sequences)))
        for sequence in sequences:
            stream.write(struct.pack("<I", len(sequence)))
            stream.write(np.asarray(sequence, dtype="<u4").tobytes())


def execute(backend: dict, artifact: Path, folder: Path, sequences: list[list[int]],
            mode: str = "verify", activation_bits: int = 32, scalar: bool = False,
            disable_vnni: bool = False):
    request, logits, metrics = (folder / name for name in ("tokens.bin", "logits.bin", "metrics.json"))
    write_request(request, sequences)
    mapping = wsl_path if backend["wsl"] else str
    command = [*backend["prefix"], backend["executable"], mapping(artifact), mapping(request),
               mapping(logits), mapping(metrics), mode, "1", "1", "0", "4",
               str(int(scalar)), "3", str(activation_bits)]
    if disable_vnni and backend["wsl"]:
        raise ValueError("ISA policy overrides are local-only; WSL environment is not implicitly modified")
    environment = None
    if disable_vnni:
        environment = os.environ.copy()
        environment["LEAF_DISABLE_VNNI"] = "1"
    completed = subprocess.run(command, capture_output=True, text=True, env=environment)
    if completed.returncode:
        raise RuntimeError(f"{backend['name']} decoder failed ({completed.returncode}): {completed.stderr.strip()}")
    return np.fromfile(logits, dtype="<f4"), json.loads(metrics.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path,
                        default=Path("build/leaf_decoder.exe" if os.name == "nt" else "build/leaf_decoder"))
    parser.add_argument("--linux-executable", help="explicit Linux executable path, run through WSL on Windows")
    parser.add_argument("--wsl-distribution", help="optional WSL distribution name")
    parser.add_argument("--workdir", type=Path, default=Path("build/decoder_portability"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/decoder_portability.json"))
    args = parser.parse_args()
    if not args.executable.is_file():
        parser.error("--executable must point to the built local native decoder")
    if args.linux_executable and os.name != "nt":
        parser.error("--linux-executable is a Windows-to-WSL check; use --executable directly on Linux")
    if args.wsl_distribution and not args.linux_executable:
        parser.error("--wsl-distribution requires --linux-executable")
    # Heavy validation imports are deliberately lazy.
    import torch
    from transformers import AutoModelForCausalLM, GPT2Config, LlamaConfig
    from tools.calibrate_decoder import collect
    from tools.decoder_plan import make_plan
    torch.set_num_threads(1)
    local = {"name": "windows" if os.name == "nt" else "local", "prefix": [], "wsl": False,
             "executable": str(args.executable.resolve()), "executable_sha256": digest(args.executable),
             "operating_system": platform.platform()}
    backends = [local]
    if args.linux_executable:
        prefix = ["wsl.exe"]
        if args.wsl_distribution:
            prefix += ["--distribution", args.wsl_distribution]
        prefix += ["--exec"]
        fingerprint = subprocess.run([*prefix, "sha256sum", args.linux_executable],
                                     capture_output=True, text=True, check=True).stdout.split()[0]
        operating_system = subprocess.run([*prefix, "uname", "-srmp"], capture_output=True,
                                         text=True, check=True).stdout.strip()
        backends.append({"name": "linux-wsl", "prefix": prefix, "wsl": True,
                         "executable": args.linux_executable, "executable_sha256": fingerprint,
                         "operating_system": operating_system})
    configs = [LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=128, bos_token_id=1, eos_token_id=2),
               GPT2Config(vocab_size=128, n_embd=64, n_inner=128, n_layer=2, n_head=4,
                          n_positions=128, bos_token_id=1, eos_token_id=2)]
    sequences = [[(index * 17 + 4) % 128 for index in range(length)] for length in (13, 37)]
    calibration_sequences = [[(index * 19 + 9) % 128 for index in range(length)] for length in (21, 29)]
    records = []
    args.workdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="protocol-", dir=args.workdir.resolve()) as temporary:
        root = Path(temporary)
        for index, config in enumerate(configs):
            config._attn_implementation = "eager"
            torch.manual_seed(707 + index)
            model = AutoModelForCausalLM.from_config(config).float().eval()
            snapshot = root / config.model_type
            model.save_pretrained(snapshot, safe_serialization=True)
            with torch.inference_mode():
                reference = np.concatenate([model(torch.tensor([ids]), use_cache=False).logits[0].numpy()
                                            for ids in sequences])
                generation = model.generate(torch.tensor([sequences[0]]), max_new_tokens=4,
                                            do_sample=False, pad_token_id=2)[0, len(sequences[0]):].tolist()
            plan = make_plan(config.to_dict())
            calibration_inputs = root / f"{config.model_type}-calibration-tokens.json"
            calibration_inputs.write_text(json.dumps(calibration_sequences), encoding="utf-8")
            metadata = {"format": "leaf-activation-calibration-v1",
                        "tokens": sum(map(len, calibration_sequences)),
                        "dataset_sha256": digest(calibration_inputs),
                        "model_config_sha256": digest(snapshot / "config.json"),
                        "source_weight_sha256": {p.name: digest(p) for p in snapshot.glob("*.safetensors")}}
            calibration = root / f"{config.model_type}-calibration.npz"
            np.savez(calibration, **collect(model, calibration_sequences, plan), __metadata__=json.dumps(metadata))
            policies = [("32", 32, 0, None, [], (32,)),
                        ("8", 8, 0, None, [], (32, 8)),
                        ("4", 4, 0, None, [], (32, 8)),
                        ("8-grouped32", 8, 32, None, [], (32, 8)),
                        ("8-protected", 8, 0, None, protected_int8_tensors(plan), (32,)),
                        ("8-grouped32-smooth", 8, 32, calibration, [], (8,))]
            for key, bits, group_size, calibration_path, kept_fp32, activation_modes in policies:
                artifact = root / f"{config.model_type}-{key}.leaf"
                exported = export_decoder(snapshot, artifact, bits, int8_group_size=group_size,
                                          calibration_path=calibration_path, keep_fp32_tensors=kept_fp32)
                for activations in activation_modes:
                    case = {"architecture": config.model_type, "random_weights": True,
                            "candidate": key, "int8_group_size": exported["int8_group_size"],
                            "keep_fp32_tensors": exported["keep_fp32_tensors"],
                            "smoothing": calibration_path is not None,
                            "weight_bits": bits, "activation_bits": activations,
                            "artifact_sha256": digest(artifact), "backends": {}}
                    if calibration_path:
                        case["calibration"] = exported["calibration"]
                    local_logits, local_generation = None, None
                    for backend in backends:
                        folder = root / f"{config.model_type}-{key}-{activations}-{backend['name']}"
                        folder.mkdir()
                        actual, metrics = execute(backend, artifact, folder, sequences, activation_bits=activations)
                        actual = actual.reshape(reference.shape)
                        measured = quality(actual, reference, sequences)
                        if bits == 32:
                            np.testing.assert_allclose(actual, reference, atol=5e-5, rtol=5e-4)
                        elif measured["relative_logit_rmse"] > (0.05 if calibration_path else 0.04 if bits == 8 else 0.25):
                            raise AssertionError("Quantized random-weight numerical sanity check failed")
                        chunked, _ = execute(backend, artifact, folder, sequences, "chunked", activations)
                        scalar, _ = execute(backend, artifact, folder, sequences, activation_bits=activations, scalar=True)
                        np.testing.assert_allclose(chunked.reshape(reference.shape), actual, atol=5e-5, rtol=5e-4)
                        np.testing.assert_allclose(scalar.reshape(reference.shape), actual, atol=5e-5, rtol=5e-4)
                        fallback_comparison = None
                        if not backend["wsl"] and bits == 8 and activations == 8:
                            fallback, fallback_metrics = execute(backend, artifact, folder, sequences,
                                                                  activation_bits=activations, disable_vnni=True)
                            fallback = fallback.reshape(reference.shape)
                            np.testing.assert_allclose(fallback, actual, atol=5e-5, rtol=5e-4)
                            if fallback_metrics.get("vnni") is True:
                                raise AssertionError("LEAF_DISABLE_VNNI did not select the fallback path")
                            fallback_comparison = {"passed": True, "bit_exact": bool(np.array_equal(actual, fallback)),
                                                   "vnni_path_exercised": metrics.get("vnni") is True,
                                                   "fallback_avx2": fallback_metrics.get("avx2"),
                                                   "fallback_vnni": fallback_metrics.get("vnni"),
                                                   "policy_scope": "Child process only; parent environment unchanged."}
                        _, generated = execute(backend, artifact, folder, [sequences[0]], "generate", activations)
                        generated_ids = generated["generated_tokens"]
                        if bits == 32 and generated_ids != generation:
                            raise AssertionError("FP32 greedy generation differs from PyTorch")
                        if local_logits is None:
                            local_logits, local_generation = actual.copy(), generated_ids
                        else:
                            np.testing.assert_allclose(actual, local_logits, atol=5e-5, rtol=5e-4)
                            if generated_ids != local_generation:
                                raise AssertionError("Cross-OS greedy generation differs")
                        case["backends"][backend["name"]] = {
                            "quality": measured, "chunked_cache_passed": True,
                            "scalar_vector_parity_passed": True, "generated_tokens": generated_ids,
                            "generation_matches_pytorch": generated_ids == generation,
                            "avx2": metrics.get("avx2"), "vnni": metrics.get("vnni"),
                            "vnni_disabled_comparison": fallback_comparison}
                    case["cross_os_checked"] = len(backends) > 1
                    case["cross_os_passed"] = True if len(backends) > 1 else None
                    records.append(case)
            print(f"{config.model_type}: protocol, precision, KV chunks and generation passed", flush=True)
    record = {"benchmark": "native-decoder-portability-sanity",
              "measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "physical_host_cpu": platform.processor(), "machine": platform.machine(),
              "same_physical_host": True,
              "scope": "Reduced random-weight models; no trained quality, latency or independent hardware claim.",
              "backends": [{key: value for key, value in backend.items() if key not in {"prefix", "executable", "wsl"}}
                           for backend in backends],
              "cases": records, "passed": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Portability record: {args.output}", flush=True)


if __name__ == "__main__":
    main()
