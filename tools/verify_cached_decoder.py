"""Revalidate FP32 and calibrated W8A8 against frozen trained references.

Uses existing artifacts without re-exporting or overwriting previous results.
This is a quality check; benchmark timing and precision selection are separate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.affinity import pin_cpu
from leaf.power import keep_awake
from tools.benchmark_core_scaling import validate_smoothed_provenance
from tools.decoder_validation import quality, run_native
from tools.validate_decoder import GATES, digest, utc_now, write_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "model", "workdir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Require a fresh output")
    if any(v for k, v in os.environ.items() if k.startswith("LEAF_EXPERIMENTAL_") or
           k in ("LEAF_DISABLE_VNNI", "LEAF_DECODER_PROFILE")):
        parser.error("Clear inherited native policies")
    pin_cpu(args.cpu)
    baseline = json.loads((args.workdir / "pytorch.json").read_text())
    tokens = json.loads((args.workdir / "tokens.json").read_text())
    weights = {p.name: digest(p) for p in args.model.glob("*.safetensors")}
    if not weights or weights != baseline["source_weight_sha256"] or digest(args.model / "config.json") != baseline["model_config_sha256"]:
        raise ValueError("Local model differs from the frozen reference")
    for name, expected in baseline["quality_cache_sha256"].items():
        if digest(args.workdir / name) != expected:
            raise ValueError("Frozen quality cache differs")
    cases = [("fp32", "decoder-32", 32), ("w8a8", "decoder-8-smooth", 8)]
    files = [args.executable, args.workdir / "pytorch.json", args.workdir / "reference.npy",
             args.workdir / "tokens.json", args.model / "config.json", Path(__file__)]
    files += list(args.model.glob("*.safetensors"))
    for _, stem, _ in cases:
        files += [args.workdir / (stem + ".leaf"), args.workdir / (stem + ".provenance.json")]
    hashes = {str(p): digest(p) for p in files}
    reference = np.load(args.workdir / "reference.npy", mmap_mode="r", allow_pickle=False)
    record = dict(format="leaf-cached-trained-quality-v1", measured_at_utc=utc_now(), hashes=hashes,
                  model=args.model.name, threads=1, cpu=args.cpu, cases={}, complete=False,
                  automatic_selection_authorized=False, thresholds=GATES)
    write_record(args.output, record)
    with keep_awake():
        for name, stem, bits in cases:
            artifact = args.workdir / (stem + ".leaf")
            provenance = json.loads((args.workdir / (stem + ".provenance.json")).read_text())
            request = provenance["request"]
            if (provenance["artifact_sha256"] != hashes[str(artifact)] or request["weight_bits"] != bits or
                    any(request[k] != baseline[k] for k in ("source_weight_sha256", "model_config_sha256"))):
                raise ValueError("Artifact precision/source differs")
            if bits == 8:
                validate_smoothed_provenance(provenance, baseline, hashes[str(artifact)])
            print(name, "full held-out logits", flush=True)
            actual, execution = run_native(args.executable, artifact, tokens["sequences"], activation_bits=bits)
            actual = actual.reshape(reference.shape)
            measured = quality(actual, reference, tokens["sequences"])
            close = bool(np.allclose(actual, reference, atol=2e-3, rtol=2e-3))
            prefix = tokens["sequences"][0]
            cached, _ = run_native(args.executable, artifact, [prefix], mode="chunked", chunk=16, activation_bits=bits)
            cache_parity = bool(np.allclose(cached.reshape(len(prefix), -1), actual[:len(prefix)], atol=2e-3, rtol=2e-3))
            del cached, actual
            _, generated = run_native(args.executable, artifact, [tokens["generation_ids"]], mode="generate",
                                      generate=len(tokens["generated_tokens"]), activation_bits=bits)
            exact_generation = generated["generated_tokens"] == tokens["generated_tokens"]
            gate = GATES[str(bits)]
            passed = bool(measured["next_token_agreement"] >= gate["min_next_token_agreement"] and
                          measured["perplexity_ratio"] <= gate["max_perplexity_ratio"] and cache_parity and
                          (bits != 32 or (close and exact_generation)))
            record["cases"][name] = dict(quality=measured, passed=passed, chunked_matches_full=cache_parity,
                fp32_reference_allclose=close, generation_matches_pytorch=exact_generation,
                generated_tokens=generated["generated_tokens"], reference_generated_tokens=tokens["generated_tokens"],
                execution=execution, artifact_bytes=artifact.stat().st_size)
            write_record(args.output, record)
            print(name, "passed", passed, json.dumps(measured), flush=True)
    if any(digest(Path(p)) != value for p, value in hashes.items()):
        raise ValueError("Validation inputs changed")
    record.update(complete=True, passed=all(c["passed"] for c in record["cases"].values()))
    write_record(args.output, record)
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
