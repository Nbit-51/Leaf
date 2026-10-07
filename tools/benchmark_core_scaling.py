"""Matched Windows physical-core scaling, serial forward/reverse process order.

This measures an explicit core budget, not a single-core kernel improvement.
Retain all samples and check within-pass, pooled and between-pass stability.
No automatic engine selection or default change follows from this experiment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_cores(cores, topology):
    if not cores or len(set(cores)) != len(cores):
        raise ValueError("Require distinct logical CPUs")
    selected = []
    for cpu in cores:
        matches = [item for item in topology if item["group"] == 0 and item["logical_cpu"] == cpu]
        if len(matches) != 1:
            raise ValueError("CPU is unavailable in processor group zero")
        selected.append(matches[0])
    if len({item["core"] for item in selected}) != len(selected):
        raise ValueError("Use one logical CPU per physical core, without SMT siblings")
    if len({item["efficiency_class"] for item in selected}) != 1:
        raise ValueError("Do not mix performance and efficiency cores")
    return selected


def child(args):
    import psutil
    from tools.decoder_validation import run_native
    from tools.benchmark_decoder_comparison import validate_timing_request
    from tools.validate_decoder import (LATENCY_WORKLOAD, load_pytorch_model,
                                        measure_pytorch_latency, write_record)
    process = psutil.Process()
    process.cpu_affinity(args.cores)
    if sorted(process.cpu_affinity()) != sorted(args.cores):
        raise RuntimeError("Requested CPU affinity was not applied")
    expected = psutil.ABOVE_NORMAL_PRIORITY_CLASS
    if process.nice() != expected:
        raise RuntimeError("Timed child must have Above Normal priority")
    data = json.loads((args.workdir / "tokens.json").read_text())
    args.threads = len(args.cores)
    if args.child == "leaf":
        os.environ["LEAF_EXPERIMENTAL_FLOAT_TILES"] = "1"
        _, metrics = run_native(args.executable, args.workdir / "decoder-32.leaf",
                                [data["benchmark_ids"]], mode="bench", threads=args.threads,
                                runs=args.runs, warmup=args.warmup, windows_above_normal=True)
        validate_timing_request(metrics, runs=args.runs, threads=args.threads, activation_bits=32)
        # The native bench protocol constructs prefix=ids[:-1], resets its
        # session each iteration and returns last-token logits in both phases.
        metrics["latency_workload"] = dict(LATENCY_WORKLOAD)
    else:
        import torch
        torch.set_num_interop_threads(1)
        model, _ = load_pytorch_model(args)
        metrics = measure_pytorch_latency(model, data["benchmark_ids"], args,
                                           implementations=(args.child,))[args.child]
        metrics["torch_version"] = torch.__version__
        metrics["torch_threads"] = torch.get_num_threads()
        metrics["torch_interop_threads"] = torch.get_num_interop_threads()
    metrics.update(affinity=process.cpu_affinity(), windows_process_priority="above_normal",
                   requested_threads=args.threads)
    write_record(args.output, metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cores", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--child", choices=["leaf", "eager", "sdpa"])
    args = parser.parse_args()
    if sys.platform != "win32" or args.output.exists() or args.runs < 5 or args.warmup < 1:
        parser.error("Require Windows, a fresh output, >=5 samples and >=1 warmup")
    if args.child:
        child(args)
        return
    from leaf.power import keep_awake
    from tools.diagnose_windows_benchmark import cpu_sets
    from tools.validate_decoder import digest, utc_now, write_record
    from tools.benchmark_decoder_comparison import stage_summary
    if any(value < 1 or value > len(args.cores) for value in args.counts) or len(set(args.counts)) != len(args.counts):
        parser.error("Core counts must be distinct and fit the selected CPUs")
    selected = validate_cores(args.cores, cpu_sets())
    conflicts = [key for key, value in os.environ.items() if value and
                 (key.startswith("LEAF_EXPERIMENTAL_") or key in ("LEAF_DISABLE_VNNI", "LEAF_DECODER_PROFILE"))]
    if conflicts:
        parser.error("Clear inherited native policies: " + ", ".join(conflicts))
    files = [args.executable, args.workdir / "decoder-32.leaf", args.workdir / "tokens.json",
             args.workdir / "decoder-32.provenance.json", args.workdir / "pytorch.json",
             args.model / "config.json", Path(__file__), ROOT / "tools/validate_decoder.py"]
    files += sorted(args.model.glob("*.safetensors"))
    hashes = {str(path): digest(path) for path in files}
    baseline = json.loads((args.workdir / "pytorch.json").read_text())
    provenance = json.loads((args.workdir / "decoder-32.provenance.json").read_text())
    weights = {p.name: digest(p) for p in args.model.glob("*.safetensors")}
    for record in (baseline, provenance["request"]):
        if record["source_weight_sha256"] != weights or record["model_config_sha256"] != digest(args.model / "config.json"):
            raise ValueError("Native/PyTorch source weights differ")
    if (provenance["artifact_sha256"] != digest(args.workdir / "decoder-32.leaf") or
            baseline["quality_cache_sha256"]["tokens.json"] != digest(args.workdir / "tokens.json") or
            provenance["request"]["weight_bits"] != 32):
        raise ValueError("Artifact/token provenance differs")
    cases = [(count, engine) for count in args.counts for engine in ("leaf", "eager", "sdpa")]
    record = dict(format="leaf-core-scaling-v1", measured_at_utc=utc_now(), platform=platform.platform(),
                  cpu=platform.processor(), topology=selected, hashes=hashes, runs=args.runs, warmup=args.warmup,
                  passes=[], automatic_selection_authorized=False, quality_validated=False,
                  scope=__doc__, complete=False)
    write_record(args.output, record)
    child_dir = args.output.parent / (args.output.stem + "-passes")
    child_dir.mkdir(exist_ok=False)
    with keep_awake():
        for index, (count, engine) in enumerate(cases + list(reversed(cases))):
            output = child_dir / f"{index:02}-{count}-{engine}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--child", engine,
                       "--executable", str(args.executable.resolve()), "--workdir", str(args.workdir.resolve()),
                       "--model", str(args.model.resolve()), "--output", str(output.resolve()),
                       "--runs", str(args.runs), "--warmup", str(args.warmup),
                       "--cores", *map(str, args.cores[:count])]
            env = dict(os.environ, OMP_NUM_THREADS=str(count), MKL_NUM_THREADS=str(count))
            result = subprocess.run(command, env=env, capture_output=True, text=True,
                                    creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS)
            if result.returncode:
                raise RuntimeError(result.stderr)
            metrics = json.loads(output.read_text())
            record["passes"].append(dict(threads=count, engine=engine, metrics=metrics))
            write_record(args.output, record)
            print(count, engine, metrics["prefill_p50_ms"], metrics["decode_p50_ms"], flush=True)
    if any(digest(Path(path)) != value for path, value in hashes.items()):
        raise ValueError("Benchmark inputs changed during execution")
    record["summary"] = {str(count): {engine: stage_summary([item["metrics"] for item in record["passes"]
        if item["threads"] == count and item["engine"] == engine]) for engine in ("leaf", "eager", "sdpa")}
        for count in args.counts}
    record["complete"] = True
    write_record(args.output, record)


if __name__ == "__main__":
    main()
