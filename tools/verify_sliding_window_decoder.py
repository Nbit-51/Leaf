"""Reduced Mistral causal-window correctness, not a trained 7B evaluation or benchmark."""
import argparse
import copy
import json
from pathlib import Path
import struct
import sys

import numpy as np
import torch
from transformers import MistralConfig, MistralForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.decoder_plan import make_plan, validate_plan
from tools.decoder_validation import run_native
from tools.export_decoder import export_decoder
from tools.validate_decoder import digest, utc_now, write_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--before-executable", type=Path)
    parser.add_argument("--source-config", type=Path, help="optional original Mistral metadata; no weights")
    parser.add_argument("--output", type=Path, required=True, help="fresh directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(812)
    record = dict(scope=__doc__, complete=False, passed=False, timing_qualified=False,
                  measured_at_utc=utc_now(), executable_sha256=digest(args.executable),
                  tolerance=dict(atol=5e-5, rtol=5e-4), cases=[])
    report = args.output / "validation.json"
    try:
        if args.source_config:
            source = json.loads(args.source_config.read_text())
            plan = make_plan(source)
            validate_plan(plan)
            record.update(source_config_sha256=digest(args.source_config),
                          source_window=plan["config"]["sliding_window"],
                          trained_weights_loaded=False)
        for window in (1, 8, 4096):
            config = MistralConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1 if window == 4096 else 2, num_attention_heads=2,
                num_key_value_heads=1, head_dim=8, max_position_embeddings=max(64, window + 8),
                sliding_window=window, bos_token_id=1, eos_token_id=None, pad_token_id=0,
                attention_dropout=0.0)
            config._attn_implementation = "eager"
            model = MistralForCausalLM(config).eval()
            snapshot = args.output / f"window-{window}"
            model.save_pretrained(snapshot, safe_serialization=True)
            artifact = snapshot / "decoder.leaf"
            export_decoder(snapshot, artifact, 32)
            assert artifact.read_bytes()[:12] == b"LEAFDC03" + struct.pack("<I", 3)
            lengths = (1, 2, 9) if window == 1 else ((7, 8, 9, 17, 33) if window == 8 else (4097,))
            sequences = [[(i * 7 + 3) % 32 for i in range(length)] for length in lengths]
            with torch.inference_mode():
                expected = np.concatenate([model(torch.tensor([ids]), use_cache=False).logits[0].numpy()
                                           for ids in sequences])
            case = dict(window=window, lengths=lengths, checks=[], artifact_sha256=digest(artifact))
            record["cases"].append(case)
            options = [("prefill", {}), ("two-threads", {"threads": 2}), ("scalar", {"scalar": True}),
                       ("chunked", {"mode": "chunked", "chunk": 3 if window < 4096 else 4096})]
            for name, kwargs in options:
                actual, _ = run_native(args.executable, artifact, sequences, **kwargs)
                actual = actual.reshape(expected.shape)
                np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-4)
                case["checks"].append(dict(mode=name, passed=True,
                    max_abs_error=float(np.max(np.abs(actual - expected)))))
                write_record(report, record)
            prompt = sequences[0][:max(1, window - 2)]
            with torch.inference_mode():
                # Full-prefix reference avoids using a rolling-cache algorithm
                # as the oracle for Leaf's different (growing) cache storage.
                generated = model.generate(torch.tensor([prompt]), do_sample=False,
                                           use_cache=False, max_new_tokens=4, pad_token_id=0)[0, len(prompt):].tolist()
                cached = model.generate(torch.tensor([prompt]), do_sample=False,
                                        use_cache=True, max_new_tokens=4, pad_token_id=0)[0, len(prompt):].tolist()
            case.update(reference_generation="full-prefix, use_cache=False", reference_tokens=generated,
                        reference_cached_tokens=cached, reference_cache_agrees=cached == generated)
            # Transformers 5.14.1's window-one cache disagrees with its own
            # full-prefix execution; retain that discrepancy, never hide it.
            if window != 1 and cached != generated:
                raise AssertionError("Reference cached/full-prefix generation disagrees")
            _, native = run_native(args.executable, artifact, [prompt], mode="generate", generate=4)
            if native["generated_tokens"] != generated:
                raise AssertionError("Sliding-window greedy tokens differ")
            case["generation_exact_match"] = True
            if window == 8:
                # Negative control: the fixture must detect replacing the window
                # with full attention. The same weights and inputs are used.
                full_config = copy.deepcopy(config)
                full_config.sliding_window = None
                full = MistralForCausalLM(full_config).eval()
                full.load_state_dict(model.state_dict())
                with torch.inference_mode():
                    full_logits = full(torch.tensor([sequences[-1]]), use_cache=False).logits[0].numpy()
                difference = float(np.max(np.abs(full_logits - expected[-lengths[-1]:])))
                if difference <= 1e-4:
                    raise AssertionError("Fixture does not distinguish full and local attention")
                case["full_attention_negative_control_max_difference"] = difference
                for name, offset, value in (("zero-window", 60, 0), ("version-mismatch", 8, 2)):
                    corrupted = bytearray(artifact.read_bytes())
                    struct.pack_into("<I", corrupted, offset, value)
                    bad = snapshot / (name + ".leaf")
                    bad.write_bytes(corrupted)
                    try:
                        run_native(args.executable, bad, [[1]])
                    except RuntimeError as error:
                        if not any(s in str(error) for s in ("window", "version")):
                            raise
                    else:
                        raise AssertionError("Malformed window artifact accepted")
                case["invalid_artifacts_rejected"] = True
                if args.before_executable:
                    try:
                        run_native(args.before_executable, artifact, [[1]])
                    except RuntimeError as error:
                        if "unsupported artifact magic" not in str(error):
                            raise
                    else:
                        raise AssertionError("Old runtime silently accepted window artifact")
                    case["old_runtime_rejects_v3"] = True
            print("window", window, "passed", flush=True)
        if digest(args.executable) != record["executable_sha256"]:
            raise ValueError("Executable changed during verification")
        record.update(complete=True, passed=True)
    except Exception as error:
        record["error"] = str(error)
        raise
    finally:
        write_record(report, record)


if __name__ == "__main__":
    main()
