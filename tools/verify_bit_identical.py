"""Require bit-identical decoder outputs between two executables on one artifact.

Compares raw logits from `verify` (full-sequence) and `chunked` modes, and the
greedy tokens from `generate`, at the requested thread counts. Exits non-zero
on any byte difference. Timing fields are ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_requests(path: Path, sequences: list[list[int]]) -> None:
    data = struct.pack("<I", len(sequences))
    for ids in sequences:
        data += struct.pack("<I", len(ids)) + struct.pack(f"<{len(ids)}I", *ids)
    path.write_bytes(data)


def run(executable: Path, artifact: Path, requests: Path, workdir: Path, label: str, mode: str,
        threads: int, chunk: int = 1, generate: int = 16) -> tuple[str, dict]:
    logits, metrics = workdir / f"{label}-{mode}-{threads}.bin", workdir / f"{label}-{mode}-{threads}.json"
    command = [str(executable), str(artifact), str(requests), str(logits), str(metrics), mode,
               str(threads), "1", "0", str(generate), "0", str(chunk)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    record = json.loads(metrics.read_text())
    if mode != "generate" and (not logits.stat().st_size or logits.stat().st_size % 4):
        raise ValueError("Expected nonempty FP32 logits")
    if mode == "generate" and len(record.get("generated_tokens", [])) != generate:
        raise ValueError("Missing or incomplete generated tokens")
    return digest(logits), record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.threads or any(t < 1 for t in args.threads):
        parser.error("Require a fresh output and positive thread counts")
    paths = {"before": args.before, "after": args.after, "artifact": args.artifact, "tokens": args.tokens}
    hashes = {name: digest(path) for name, path in paths.items()}
    args.workdir.mkdir(parents=True, exist_ok=True)
    tokens = json.loads(args.tokens.read_text())
    sequences = tokens["sequences"]
    requests = args.workdir / "sequences.bin"
    write_requests(requests, sequences)
    generation = args.workdir / "generation.bin"
    write_requests(generation, [tokens["generation_ids"]])
    results, failures = [], []
    for threads in args.threads:
        for mode, path, chunk in (("verify", requests, 1), ("chunked", requests, 7), ("generate", generation, 1)):
            before = run(args.before, args.artifact, path, args.workdir, "before", mode, threads, chunk)
            after = run(args.after, args.artifact, path, args.workdir, "after", mode, threads, chunk)
            same_logits = before[0] == after[0]
            same_tokens = before[1].get("generated_tokens") == after[1].get("generated_tokens")
            entry = {"mode": mode, "threads": threads, "chunk": chunk, "logits_sha256_before": before[0],
                     "logits_sha256_after": after[0], "logits_identical": same_logits,
                     "generated_tokens_identical": same_tokens,
                     "optimized_fp32_active": [before[1].get("optimized_fp32_active"),
                                               after[1].get("optimized_fp32_active")]}
            if mode == "generate":
                entry["generated_tokens"] = after[1].get("generated_tokens")
                entry["reference_generated_tokens"] = tokens.get("generated_tokens")
                entry["reference_tokens_match"] = (tokens.get("generated_tokens") is None or
                                                    entry["generated_tokens"] == tokens["generated_tokens"])
            results.append(entry)
            print(mode, threads, "logits identical" if same_logits else "LOGITS DIFFER",
                  "tokens identical" if same_tokens else "TOKENS DIFFER")
            if not (same_logits and same_tokens and entry.get("reference_tokens_match", True)):
                failures.append(entry)
    if hashes != {name: digest(path) for name, path in paths.items()}:
        raise ValueError("Validation inputs changed during execution")
    record = {"before": str(args.before), "after": str(args.after), "artifact": str(args.artifact),
              "sequences": len(sequences), "scored_targets": sum(len(s) - 1 for s in sequences),
              "sha256": hashes,
              "results": results, "passed": not failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
