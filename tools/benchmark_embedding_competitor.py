"""Paired CPU-only normalized-embedding benchmark using identical saved token IDs."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.embedding_validation import parity
from tools.verify_embeddinggemma_reference import digest


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Invalid timing samples")
    spread = float(np.percentile(values, 90) / np.percentile(values, 10))
    return dict(
        median_ms=float(np.median(values)),
        p90_over_p10=spread,
        stable=spread <= 1.25,
        samples=len(values),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "leaf-executable",
        "llama-executable",
        "leaf-artifact",
        "gguf",
        "reference",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--runs", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument(
        "--reference-engine", choices=("llama", "leaf_before"), default="llama"
    )
    args = parser.parse_args()
    comparator = args.reference_engine
    if args.output.exists() or min(args.rounds, args.runs) < 1 or args.warmups < 0:
        parser.error("Use positive counts and a fresh output directory")
    args.output.mkdir(parents=True)
    reference = json.loads((args.reference / "reference.json").read_text())
    for name, checksum in reference["fixture_sha256"].items():
        if digest(args.reference / name) != checksum:
            raise ValueError("Reference fixture changed")
    with np.load(args.reference / "reference.npz") as saved:
        sequences = [
            row[mask.astype(bool)].tolist()
            for row, mask in zip(saved["input_ids"], saved["attention_mask"])
        ]
        sequences.append(
            saved["long_input_ids"][0][
                saved["long_attention_mask"][0].astype(bool)
            ].tolist()
        )
        expected = np.concatenate([saved["embeddings"], saved["long_embeddings"]])
    request = args.output / "request.bin"
    with request.open("wb") as stream:
        stream.write(b"LEAFER01" + struct.pack("<I", len(sequences)))
        for ids in sequences:
            stream.write(
                struct.pack("<I", len(ids)) + np.asarray(ids, dtype="<u4").tobytes()
            )
    paths = [
        args.leaf_executable,
        args.llama_executable,
        args.leaf_artifact,
        args.gguf,
        Path(__file__),
        Path("benchmark/embedding_runtime_benchmark.cpp"),
    ]
    hashes = {str(p): digest(p) for p in paths}
    record = dict(
        format="leaf-embedding-paired-comparison-v2",
        reference_engine=comparator,
        measured_at_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),
        cpu=platform.processor(),
        cpu_affinity=[args.cpu],
        threads=1,
        weight_precision="FP32",
        llama_kv_precision="FP32" if comparator == "llama" else None,
        gpu_layers=0,
        scope="Warm token IDs to normalized embedding; includes reset, pooling, projections, normalization and output copy. Excludes load, tokenization, file IO and HTTP.",
        lengths=[len(x) for x in sequences],
        warmups=args.warmups,
        runs=args.runs,
        stability_limit_p90_over_p10=1.25,
        required_speed_ratio=0.98,
        hashes=hashes,
        passes=[],
        complete=False,
        retrieval_quality_qualified=False,
        overall_speed_claim=False,
    )
    report = args.output / "comparison.json"

    def save():
        report.write_text(json.dumps(record, indent=2) + "\n")

    def execute(engine, label, warmups, runs):
        exe, model = (
            (args.leaf_executable, args.leaf_artifact)
            if engine == "leaf"
            else (args.llama_executable, args.gguf)
        )
        raw = args.output / (label + ".bin")
        command = [
            str(exe.resolve()),
            str(model.resolve()),
            str(request.resolve()),
            str(raw.resolve()),
            "1",
            str(warmups),
            str(runs),
            str(args.llama_executable.parent.resolve()),
        ]
        stdout, stderr = args.output / (label + ".json"), args.output / (label + ".log")
        parent = psutil.Process()
        affinity = parent.cpu_affinity()
        peak = 0
        with stdout.open("w") as out, stderr.open("w") as err:
            # Children inherit affinity before loading; restore the orchestrator immediately.
            try:
                parent.cpu_affinity([args.cpu])
                process = subprocess.Popen(command, stdout=out, stderr=err)
            finally:
                parent.cpu_affinity(affinity)
            measured = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    peak = max(peak, measured.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.05)
            if process.returncode:
                raise RuntimeError(f"{engine} failed; see {stderr}")
        values = np.fromfile(raw, dtype="<f4").reshape(len(sequences), 768)
        quality = parity(values, expected)
        timing = json.loads(stdout.read_text())
        if len(timing["samples_ms"]) != len(sequences) or any(
            len(x) != runs for x in timing["samples_ms"]
        ):
            raise ValueError("Incomplete timing output")
        return dict(
            engine=engine, parity=quality, peak_process_rss_bytes=peak, **timing
        )

    save()
    # A mismatch stops the experiment before acceptance timing.
    for engine in ("leaf", comparator):
        checked = execute(engine, "preflight-" + engine, 0, 1)
        record.setdefault("preflight", []).append(checked)
        save()
        print(engine, "parity passed", checked["parity"]["max_abs_error"], flush=True)
    for iteration in range(args.rounds):
        for engine in (
            ("leaf", comparator) if iteration % 2 == 0 else (comparator, "leaf")
        ):
            checked = execute(
                engine, f"{iteration + 1}-{engine}", args.warmups, args.runs
            )
            checked["round"] = iteration + 1
            checked["summaries"] = [distribution(x) for x in checked["samples_ms"]]
            record["passes"].append(checked)
            save()
            print(
                iteration + 1,
                engine,
                [round(x["median_ms"], 3) for x in checked["summaries"]],
                flush=True,
            )
    record["summary"] = []
    for i, sequence in enumerate(sequences):
        by_engine = {
            e: [x for x in record["passes"] if x["engine"] == e]
            for e in ("leaf", comparator)
        }
        stats = {
            e: distribution([t for p in passes for t in p["samples_ms"][i]])
            for e, passes in by_engine.items()
        }
        ratios = [
            a["summaries"][i]["median_ms"] / b["summaries"][i]["median_ms"]
            for a, b in zip(by_engine["leaf"], by_engine[comparator])
        ]
        stable = all(p["summaries"][i]["stable"] for p in record["passes"]) and all(
            s["stable"] for s in stats.values()
        )
        ratio = float(np.median(ratios))
        record["summary"].append(
            dict(
                tokens=len(sequence),
                engines=stats,
                paired_leaf_over_reference=ratio,
                stable=stable,
                leaf_win_qualified=stable
                and ratio <= 0.98
                and args.rounds >= 6
                and args.runs >= 11,
            )
        )
    if any(digest(Path(p)) != value for p, value in hashes.items()):
        raise ValueError("Benchmark inputs changed")
    record["complete"] = True
    save()
    print(json.dumps(record["summary"], indent=2), flush=True)


if __name__ == "__main__":
    from leaf.power import keep_awake
    with keep_awake():
        main()
