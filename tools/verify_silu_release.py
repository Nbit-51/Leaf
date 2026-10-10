"""Bind a release build to accepted SiLU results through exact native parity.

W8A8 must match the accepted experimental executable; FP32 must match the
unchanged baseline. No timings or automatic precision selection are qualified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.affinity import pin_cpu
from leaf.power import keep_awake
from tools.benchmark_default_abba import validate_silu_quality
from tools.decoder_validation import run_native
from tools.validate_decoder import digest, utc_now, write_record


def check_release_dispatch(metrics, bits, *, scalar=False, tokenwise=False):
    if any(v for k, v in metrics.items() if k.startswith("experimental_") and k.endswith("_build")):
        raise ValueError("Release executable contains experimental build flags")
    if metrics.get("diagnostic_timing_build"):
        raise ValueError("Release executable contains diagnostic instrumentation")
    eligible = bits == 8 and not scalar and metrics.get("avx2") is True
    calls = metrics.get("vector_silu_gate_calls")
    if (metrics.get("vector_silu_gate_enabled") is not eligible or type(calls) is not int or
            calls < 0 or (calls > 0) != (eligible and not tokenwise)):
        raise ValueError("Release SiLU dispatch differs from qualified precision/workload")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "before", "candidate", "quality", "acceptance", "workdir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "win32" or args.output.exists():
        parser.error("Require Windows and a fresh output")
    pin_cpu(2)
    quality = json.loads(args.quality.read_text())
    accepted = json.loads(args.acceptance.read_text())
    tokens_path = args.workdir / "tokens.json"
    artifact8 = args.workdir / "decoder-8-smooth.leaf"
    validate_silu_quality(quality, args.candidate, artifact8, tokens_path, 8)
    saved = {Path(p).resolve(): h for p, h in accepted.get("hashes", {}).items()}
    if (accepted.get("complete") is not True or accepted.get("gate", {}).get("passed") is not True or
            accepted.get("activation_bits") != 8):
        raise ValueError("Require completed passing W8A8 acceptance")
    for path in (args.before, args.candidate, artifact8, tokens_path, args.quality):
        if saved.get(path.resolve()) != digest(path):
            raise ValueError("Acceptance does not match the reference binaries/inputs")
    artifact32 = args.workdir / "decoder-32.leaf"
    quality_hashes = {Path(p).resolve(): h for p, h in quality["hashes"].items()}
    if quality_hashes.get(artifact32.resolve()) != digest(artifact32):
        raise ValueError("FP32 artifact differs from quality evidence")
    paths = [args.release, args.before, args.candidate, args.quality, args.acceptance,
             artifact8, artifact32, tokens_path, Path(__file__)]
    hashes = {str(p): digest(p) for p in paths}
    record = dict(format="leaf-silu-release-parity-v1", measured_at_utc=utc_now(), hashes=hashes,
                  automatic_selection_authorized=False, acceptance_timing=False, checks=[], complete=False)
    tokens = json.loads(tokens_path.read_text())
    write_record(args.output, record)
    with keep_awake():
        for bits, artifact, reference in ((32, artifact32, args.before), (8, artifact8, args.candidate)):
            workloads = [("full", tokens["sequences"], {}),
                         ("chunk16", [tokens["sequences"][0]], dict(mode="chunked", chunk=16)),
                         ("tokenwise", [tokens["sequences"][0]], dict(mode="chunked", chunk=1)),
                         ("generation", [tokens["generation_ids"]],
                          dict(mode="generate", generate=len(tokens["generated_tokens"]))),
                         ("scalar", [tokens["sequences"][0]], dict(scalar=True))]
            for name, sequences, options in workloads:
                actual, metrics = run_native(args.release, artifact, sequences, activation_bits=bits, **options)
                expected, reference_metrics = run_native(reference, artifact, sequences, activation_bits=bits, **options)
                check_release_dispatch(metrics, bits, scalar=name == "scalar", tokenwise=name == "tokenwise")
                if (not np.array_equal(actual, expected) or
                        metrics.get("generated_tokens") != reference_metrics.get("generated_tokens")):
                    raise ValueError(f"{bits}/{name}: release differs from qualified reference")
                record["checks"].append(dict(bits=bits, workload=name, logits_exact=True,
                    generation_exact=True, execution=metrics))
                write_record(args.output, record)
                print(bits, name, "exact parity and dispatch passed", flush=True)
                del actual, expected
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("Release audit inputs changed during execution")
    record.update(complete=True, passed=True)
    write_record(args.output, record)


if __name__ == "__main__":
    main()
