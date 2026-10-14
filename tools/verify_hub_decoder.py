"""Correctness-only onboarding of a supported Hub decoder; never qualifies speed.

Uses Leaf's normal model-ID resolution/export and the existing native protocol.
Weights are not quantized. Reference computation promotes the same source
weights to FP32; it does not reconstruct precision absent from the checkpoint.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leaf import cli
from tools.datasets import token_prefix
from tools.decoder_validation import quality, run_native
from tools.validate_decoder import GATES, digest, utc_now, write_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hub ID or local snapshot")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh directory to preserve evidence")
    args.output.mkdir(parents=True)
    record = dict(format="leaf-hub-decoder-correctness-v1", model=args.model,
                  measured_at_utc=utc_now(), complete=False, passed=False,
                  timing_qualified=False, scope=__doc__, checks=[],
                  logit_tolerance=dict(atol=2e-4, rtol=2e-4), quality_gate=GATES["32"])
    report = args.output / "correctness.json"
    try:
        import psutil
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        battery = psutil.sensors_battery()
        record["power"] = None if battery is None else dict(
            plugged_in=battery.power_plugged, battery_percent=battery.percent)
        cache = cli.cache_root()
        snapshot = cli.resolve_model(args.model, cache, args.offline)
        cli.validate_model_generation(snapshot)
        runtime = cli.executable(cache)
        artifact = cli.prepare(snapshot, cli.model_directory(snapshot, cache), 32)
        record.update(snapshot=str(snapshot), artifact=str(artifact),
                      source_sha256={p.name: digest(p) for p in sorted(snapshot.iterdir())
                                     if p.is_file() and p.suffix in (".json", ".safetensors")},
                      artifact_sha256=digest(artifact), executable_sha256=digest(runtime),
                      dataset_sha256=digest(args.dataset), torch=torch.__version__,
                      transformers=transformers.__version__)
        write_record(report, record)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        ids = token_prefix(tokenizer, args.dataset, 256, "text")
        sequences = [ids[:17], ids[17:50], ids[50:114], ids[114:241]]
        prompt = "The purpose of a computer is"
        _, prompt_ids = cli.encode_prompt(snapshot, prompt, raw=True)
        if prompt_ids != tokenizer.encode(prompt):
            raise AssertionError("Leaf and reference tokenization differ")
        record.update(tokenizer_ids_equal=True, sequence_lengths=list(map(len, sequences)),
                      generation_prompt=prompt, generation_input_ids=prompt_ids)
        model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
                    trust_remote_code=False, dtype=torch.float32, attn_implementation="eager").eval()
        with torch.inference_mode():
            references = [model(torch.tensor([row]), use_cache=False).logits[0].numpy().copy()
                          for row in sequences]
            generated = model.generate(torch.tensor([prompt_ids]), do_sample=False, max_new_tokens=16,
                                       pad_token_id=tokenizer.eos_token_id)[0, len(prompt_ids):].tolist()
        del model
        expected = np.concatenate(references)
        np.save(args.output / "reference.npy", expected)
        write_record(args.output / "tokens.json", dict(sequences=sequences, generation_ids=prompt_ids,
                                                       generated_tokens=generated))
        for name, options in (("one-thread", {}), ("two-threads", {"threads": 2}),
                              ("scalar", {"scalar": True}), ("chunked", {"mode": "chunked", "chunk": 7})):
            actual, _ = run_native(runtime, artifact, sequences, **options)
            actual = actual.reshape(expected.shape)
            measured = quality(actual, expected, sequences)
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
            gate = GATES["32"]
            if (measured["next_token_agreement"] < gate["min_next_token_agreement"] or
                    measured["perplexity_ratio"] > gate["max_perplexity_ratio"]):
                raise AssertionError("FP32 quality gate failed")
            record["checks"].append(dict(mode=name, quality=measured, logits_allclose=True))
            write_record(report, record)
            print(name, "passed", measured, flush=True)
        _, native = run_native(runtime, artifact, [prompt_ids], mode="generate", generate=16)
        if native["generated_tokens"] != generated:
            raise AssertionError("Greedy generation differs from the FP32 reference")
        if digest(artifact) != record["artifact_sha256"] or digest(runtime) != record["executable_sha256"]:
            raise ValueError("Runtime or artifact changed during verification")
        for name, checksum in record["source_sha256"].items():
            if digest(snapshot / name) != checksum:
                raise ValueError("Source snapshot changed during verification")
        record.update(complete=True, passed=True, generation_exact_match=True,
                      generated_tokens=generated, generated_text=tokenizer.decode(generated),
                      fixture_sha256={p.name: digest(p) for p in
                                      (args.output / "reference.npy", args.output / "tokens.json")})
    except Exception as error:
        record["error"] = str(error)
        raise
    finally:
        write_record(report, record)
    print("Hub decoder correctness passed; no latency qualification")


if __name__ == "__main__":
    main()
