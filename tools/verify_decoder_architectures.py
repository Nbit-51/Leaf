"""Full small-model parity across distinct decoder architectures and precisions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from transformers import (AutoModelForCausalLM, GPT2Config, GPTNeoXConfig,
                          LlamaConfig, OPTConfig, Qwen2Config)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.decoder_validation import quality, run_native
from tools.export_decoder import export_decoder
from tools.decoder_plan import make_plan
from tools.calibrate_decoder import collect
from tools.validate_decoder import digest, native_policy


def configurations():
    common = dict(vocab_size=128, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4,
                  max_position_embeddings=128, bos_token_id=1, eos_token_id=2)
    return [LlamaConfig(**common, num_key_value_heads=2),
            Qwen2Config(**common, num_key_value_heads=2),
            GPT2Config(vocab_size=128, n_embd=64, n_inner=128, n_layer=2,
                       n_head=4, n_positions=128, bos_token_id=1, eos_token_id=2),
            GPTNeoXConfig(**common, rotary_pct=0.5),
            OPTConfig(vocab_size=128, hidden_size=64, word_embed_proj_dim=64,
                      ffn_dim=128, num_hidden_layers=2, num_attention_heads=4,
                      max_position_embeddings=128, bos_token_id=1, eos_token_id=2),
            GPT2Config(vocab_size=128, n_embd=30, n_inner=60, n_layer=2, n_head=10,
                       n_positions=128, bos_token_id=1, eos_token_id=2),
            GPTNeoXConfig(**common, rotary_pct=0.5, attention_bias=False, tie_word_embeddings=True),
            OPTConfig(vocab_size=128, hidden_size=64, word_embed_proj_dim=64, ffn_dim=128,
                      num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=128,
                      bos_token_id=1, eos_token_id=2, enable_bias=False),
            LlamaConfig(**common, num_key_value_heads=2, hidden_act="relu")]


def paired_native(executable, before_executable, artifact, sequences, checks, label, **options):
    """Compare identical native protocol requests without recording timings."""
    actual, metrics = run_native(executable, artifact, sequences, **options)
    if before_executable is not None:
        before, before_metrics = run_native(before_executable, artifact, sequences, **options)
        if not np.array_equal(actual, before):
            raise AssertionError(f"{label}: native logits are not exactly equal to the before executable")
        if metrics.get("generated_tokens") != before_metrics.get("generated_tokens"):
            raise AssertionError(f"{label}: native generated tokens differ from the before executable")
        checks[label] = {"logits_array_equal": True, "generated_tokens_equal": True}
    return actual, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build/leaf_decoder.exe"))
    parser.add_argument("--before-executable", type=Path,
                        help="optional frozen binary for exact paired native regression; no timing comparison")
    parser.add_argument("--conservative-executable", type=Path,
                        help="release audit: exact quantized/scalar parity and optimized FP32 dispatch")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect-w8a8-silu", action="store_true",
                        help="audit Windows W8A8-only SiLU default and excluded dispatch paths")
    args = parser.parse_args()
    if args.before_executable and args.conservative_executable:
        parser.error("Choose a full paired regression or a conservative release audit")
    conservative_hash = digest(args.conservative_executable) if args.conservative_executable else None
    if args.conservative_executable and any(native_policy().values()):
        parser.error("Release audit requires the default native policy")
    before_hash = None
    if args.before_executable is not None:
        if not args.executable.is_file() or not args.before_executable.is_file():
            parser.error("Paired regression requires existing before and candidate executables")
        if any(value for value in native_policy().values()):
            parser.error("Paired regression requires the default native kernel policy; unset experimental/ISA flags")
        before_hash = digest(args.before_executable)
    torch.set_num_threads(1)
    sequences = [[(i * 17 + 4) % 128 for i in range(length)] for length in (13, 37)]
    results = []
    with tempfile.TemporaryDirectory(prefix="leaf_architectures_") as temp:
        root = Path(temp)
        for index, config in enumerate(configurations()):
            config._attn_implementation = "eager"
            torch.manual_seed(707 + index)
            model = AutoModelForCausalLM.from_config(config).float().eval()
            snapshot = root / config.model_type
            model.save_pretrained(snapshot, safe_serialization=True)
            with torch.inference_mode():
                expected = np.concatenate([model(torch.tensor([ids]), use_cache=False).logits[0].numpy() for ids in sequences])
                generated = model.generate(torch.tensor([sequences[0]]), max_new_tokens=4,
                                           do_sample=False, pad_token_id=2)[0, len(sequences[0]):].tolist()
            cases = {}
            paired_checks = {}

            def native(artifact, requests, label, **options):
                before = args.before_executable
                quantized = label.startswith(("8/", "4/", "8-smooth/"))
                if args.conservative_executable and (quantized or options.get("scalar")):
                    before = args.conservative_executable
                actual, metrics = paired_native(args.executable, before, artifact, requests,
                                                paired_checks, label, **options)
                if args.expect_w8a8_silu:
                    eligible = bool(sys.platform == "win32" and metrics["avx2"] and
                        not options.get("scalar", False) and options.get("activation_bits", 32) == 8 and
                        label.startswith(("8/", "8-smooth/")) and config.model_type in ("llama", "qwen2") and
                        getattr(config, "hidden_act", None) == "silu")
                    chunk = options.get("chunk") if options.get("mode") == "chunked" else None
                    used = eligible and any(min(len(ids), chunk or len(ids)) >= 8 for ids in requests)
                    if (metrics.get("experimental_silu_gate_build") or
                            metrics.get("vector_silu_gate_enabled") is not eligible or
                            type(metrics.get("vector_silu_gate_calls")) is not int or
                            (metrics["vector_silu_gate_calls"] > 0) != used):
                        raise AssertionError(f"{label}: W8A8-only SiLU release dispatch differs")
                if args.conservative_executable:
                    expected_dispatch = not quantized and not options.get("scalar", False) and metrics["avx2"]
                    if metrics.get("optimized_fp32_active") is not expected_dispatch:
                        raise AssertionError(f"{label}: release FP32 dispatch differs")
                return actual, metrics

            for bits in (32, 8, 4):
                artifact = root / f"{config.model_type}-{bits}.leaf"
                export_decoder(snapshot, artifact, bits)
                actual, _ = native(artifact, sequences, f"{bits}/verify-a32")
                actual = actual.reshape(expected.shape)
                measurements = quality(actual, expected, sequences)
                if bits == 32:
                    np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-4)
                    for scalar, threads in ((True, 1), (False, 2)):
                        alternative, _ = native(artifact, sequences, f"32/verify-scalar{int(scalar)}-threads{threads}",
                                                scalar=scalar, threads=threads)
                        np.testing.assert_allclose(alternative.reshape(expected.shape), expected, atol=5e-5, rtol=5e-4)
                    chunked, _ = native(artifact, sequences, "32/chunk3-a32", mode="chunked", chunk=3)
                    np.testing.assert_allclose(chunked.reshape(expected.shape), expected, atol=5e-5, rtol=5e-4)
                    _, native_generation = native(artifact, [sequences[0]], "32/greedy-a32", mode="generate", generate=4)
                    if native_generation["generated_tokens"] != generated:
                        raise AssertionError(f"{config.model_type}: greedy generation differs")
                    if config.model_type == "llama":
                        corrupt = root / "invalid.leaf"
                        corrupt.write_bytes(artifact.read_bytes()[:32])
                        try:
                            run_native(args.executable, corrupt, sequences)
                        except RuntimeError as error:
                            if "truncated" not in str(error):
                                raise
                        else:
                            raise AssertionError("Truncated artifact was accepted")
                        for invalid in ([], [config.vocab_size], [1] * (config.max_position_embeddings + 1)):
                            try:
                                run_native(args.executable, artifact, [invalid])
                            except RuntimeError:
                                pass
                            else:
                                raise AssertionError("Invalid token request was accepted")
                        # An unknown model ID with an explicit plan exercises
                        # the same runtime without adding a native model-name branch.
                        custom_plan = root / "custom-plan.json"
                        custom_plan.write_text(json.dumps(make_plan(config.to_dict())))
                        saved_config = json.loads((snapshot / "config.json").read_text())
                        saved_config["model_type"] = "custom_decoder"
                        (snapshot / "config.json").write_text(json.dumps(saved_config))
                        custom_artifact = root / "custom.leaf"
                        export_decoder(snapshot, custom_artifact, plan_path=custom_plan)
                        custom, _ = native(custom_artifact, sequences, "custom-plan/verify-a32")
                        np.testing.assert_allclose(custom.reshape(expected.shape), expected, atol=5e-5, rtol=5e-4)
                        saved_config["model_type"] = "llama"
                        (snapshot / "config.json").write_text(json.dumps(saved_config))
                elif measurements["relative_logit_rmse"] > (0.03 if bits == 8 else 0.25):
                    raise AssertionError(f"{config.model_type}: quantized logits diverge")
                if bits != 32:
                    vector, _ = native(artifact, sequences, f"{bits}/verify-a8", activation_bits=8)
                    scalar, _ = native(artifact, sequences, f"{bits}/verify-a8-scalar", activation_bits=8, scalar=True)
                    np.testing.assert_allclose(vector, scalar, atol=5e-5, rtol=5e-4)
                if args.before_executable is not None or args.expect_w8a8_silu:
                    # Multiple requests force reset after a populated KV cache;
                    # repeating the first prefix also exercises table reuse.
                    reset_requests = [sequences[0], sequences[1], sequences[0]]
                    native(artifact, reset_requests, f"{bits}/greedy-reset-a32", mode="generate", generate=4)
                    if bits != 32:
                        if args.expect_w8a8_silu:
                            threaded, _ = native(artifact, sequences, f"{bits}/verify-a8-threads2",
                                                 activation_bits=8, threads=2)
                            np.testing.assert_allclose(threaded, vector, atol=5e-5, rtol=5e-4)
                        native(artifact, sequences, f"{bits}/chunk3-a32", mode="chunked", chunk=3)
                        native(artifact, sequences, f"{bits}/chunk3-a8", mode="chunked", chunk=3, activation_bits=8)
                        native(artifact, reset_requests, f"{bits}/greedy-reset-a8", mode="generate", generate=4,
                               activation_bits=8)
                cases[str(bits)] = measurements
            calibration = root / f"{config.model_type}-calibration.npz"
            arrays = collect(model, sequences, make_plan(config.to_dict()))
            metadata = {"format": "leaf-activation-calibration-v1",
                        "model_config_sha256": digest(snapshot / "config.json"),
                        "source_weight_sha256": {p.name: digest(p) for p in snapshot.glob("*.safetensors")}}
            np.savez(calibration, **arrays, __metadata__=json.dumps(metadata))
            smoothed = root / f"{config.model_type}-smooth.leaf"
            export_decoder(snapshot, smoothed, 8, calibration_path=calibration)
            vector, _ = native(smoothed, sequences, "8-smooth/verify-a8", activation_bits=8)
            scalar, _ = native(smoothed, sequences, "8-smooth/verify-a8-scalar", activation_bits=8, scalar=True)
            np.testing.assert_allclose(vector, scalar, atol=5e-5, rtol=5e-4)
            measured_smooth = quality(vector.reshape(expected.shape), expected, sequences)
            if measured_smooth["relative_logit_rmse"] > 0.05:
                raise AssertionError("Smoothed random-weight logits diverge")
            cases["8-smooth"] = measured_smooth
            if args.before_executable is not None or args.expect_w8a8_silu:
                native(smoothed, sequences, "8-smooth/chunk3-a8", mode="chunked", chunk=3, activation_bits=8)
                native(smoothed, [sequences[0], sequences[1], sequences[0]], "8-smooth/greedy-reset-a8",
                       mode="generate", generate=4, activation_bits=8)
            results.append({"architecture": config.model_type, "random_weights": True,
                            "hidden_act": getattr(config, "hidden_act", None),
                            "hidden_size": getattr(config, "hidden_size", None),
                            "attention_bias": getattr(config, "attention_bias", None),
                            "enable_bias": getattr(config, "enable_bias", None),
                            "tie_word_embeddings": config.tie_word_embeddings,
                            "fp32_scalar_and_threads2_passed": True, "chunked_cache_passed": True,
                            "greedy_generation_passed": True, "cases": cases})
            if args.before_executable is not None:
                results[-1]["paired_bit_exact_regression"] = {"passed": True, "checks": paired_checks}
            if args.conservative_executable:
                results[-1]["release_dispatch_and_unchanged_paths"] = {"passed": True, "checks": paired_checks}
            print(config.model_type + " passed", flush=True)
    record = {"benchmark": "native-decoder-cross-architecture-parity",
              "measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "machine": platform.machine(), "cpu": platform.processor(),
              "torch": torch.__version__, "threads": 1, "native_executable_sha256": digest(args.executable),
              "native_policy": native_policy(),
              "w8a8_silu_dispatch_audited": args.expect_w8a8_silu,
              "scope": "Complete reduced-size random-weight models; trained quality is evaluated separately.",
              "results": results}
    if args.before_executable is not None:
        if digest(args.before_executable) != before_hash:
            raise AssertionError("Before executable changed during paired native regression")
        record.update(before_executable_sha256=before_hash, paired_bit_exact_regression_passed=True,
                      paired_scope="Identical artifacts and requests under default kernel policy; no latency qualification")
    if args.output:
        if args.conservative_executable:
            if digest(args.conservative_executable) != conservative_hash:
                raise AssertionError("Conservative executable changed during validation")
            record.update(conservative_executable_sha256=conservative_hash, release_dispatch_verified=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
