"""Windows embedding input/core diagnostics; instrumented timings never qualify speed wins."""
import argparse
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.power import keep_awake
from tools.diagnose_windows_benchmark import cpu_sets
from tools.embedding_validation import parity
from tools.validate_decoder import digest, utc_now


def summarize(metrics, runs):
    if metrics.get("diagnostic_timing_build") is not True:
        raise ValueError("Require instrumented binary")
    names = ("samples_ms", "thread_cpu_ms", "thread_cycles", "process_page_faults")
    count = len(metrics["samples_ms"])
    if not count:
        raise ValueError("Empty diagnostic")
    for name in names:
        values = np.asarray(metrics[name], dtype=float)
        if values.shape != (count, runs) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid diagnostic samples: " + name)
    summaries = []
    for i in range(count):
        wall = np.asarray(metrics["samples_ms"][i])
        cpu = np.asarray(metrics["thread_cpu_ms"][i])
        if (wall <= 0).any():
            raise ValueError("Invalid wall time")
        slow = int(np.argmax(wall))
        summaries.append(dict(
            wall_median_ms=float(np.median(wall)),
            cpu_median_ms=float(np.median(cpu)),
            wall_p90_over_p10=float(np.percentile(wall, 90) / np.percentile(wall, 10)),
            cycles_median=float(np.median(metrics["thread_cycles"][i])),
            faults_total=int(sum(metrics["process_page_faults"][i])),
            slowest={name: metrics[name][i][slow] for name in names},
        ))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("leaf-executable", "llama-executable", "leaf-artifact", "gguf", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cpus", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.runs < 3 or args.warmups < 1:
        parser.error("Use a fresh directory, >=3 samples and >=1 warmup")
    topology = cpu_sets()
    selected = [next((x for x in topology if x["group"] == 0 and x["logical_cpu"] == cpu), None)
                for cpu in args.cpus]
    if any(x is None for x in selected) or len({x["core"] for x in selected}) != len(selected):
        parser.error("Select distinct physical cores in processor group zero")
    args.output.mkdir(parents=True)
    reference = json.loads((args.reference / "reference.json").read_text())
    for name, checksum in reference["fixture_sha256"].items():
        if digest(args.reference / name) != checksum:
            raise ValueError("Reference fixture changed")
    with np.load(args.reference / "reference.npz") as saved:
        sequences = [row[mask.astype(bool)].tolist()
                     for row, mask in zip(saved["input_ids"], saved["attention_mask"])]
        sequences.append(saved["long_input_ids"][0][saved["long_attention_mask"][0].astype(bool)].tolist())
        expected = np.concatenate([saved["embeddings"], saved["long_embeddings"]])
    paths = [args.leaf_executable, args.llama_executable, args.leaf_artifact, args.gguf,
             args.reference / "reference.npz", Path(__file__),
             ROOT / "benchmark/embedding_runtime_benchmark.cpp", ROOT / "tests/native/decoder_diagnostic.h"]
    hashes = {str(p): digest(p) for p in paths}
    record = dict(format="leaf-embedding-input-diagnostic-v1", measured_at_utc=utc_now(),
                  platform=platform.platform(), topology=topology, selected_cpus=args.cpus,
                  threads=1, runs=args.runs, warmups=args.warmups, hashes=hashes,
                  lengths=list(map(len, sequences)), passes=[], complete=False,
                  speed_qualified=False,
                  counter_limits="Thread CPU time is coarse and covers the calling thread only; faults include soft and hard process faults. Cycles are not elapsed time or measured frequency. No context-switch or thermal attribution.")
    report = args.output / "diagnostic.json"

    def save():
        report.write_text(json.dumps(record, indent=2) + "\n")

    save()
    with keep_awake():
        # Reverse the core order too, to reduce confounding core with elapsed run time.
        for reverse in (False, True):
            order = list(range(len(sequences)))
            if reverse:
                order.reverse()
            request = args.output / ("reverse.bin" if reverse else "forward.bin")
            with request.open("wb") as stream:
                stream.write(b"LEAFER01" + struct.pack("<I", len(order)))
                for i in order:
                    stream.write(struct.pack("<I", len(sequences[i])))
                    stream.write(np.asarray(sequences[i], dtype="<u4").tobytes())
            for cpu in (list(reversed(args.cpus)) if reverse else args.cpus):
                for engine in (("llama", "leaf") if reverse else ("leaf", "llama")):
                    label = f"cpu{cpu}-{'reverse' if reverse else 'forward'}-{engine}"
                    executable, model = ((args.leaf_executable, args.leaf_artifact) if engine == "leaf"
                                         else (args.llama_executable, args.gguf))
                    raw = args.output / (label + ".bin")
                    command = [str(executable.resolve()), str(model.resolve()), str(request.resolve()),
                               str(raw.resolve()), "1", str(args.warmups), str(args.runs),
                               str(args.llama_executable.parent.resolve())]
                    parent = psutil.Process()
                    affinity = parent.cpu_affinity()
                    with (args.output / (label + ".log")).open("w") as err:
                        try:
                            parent.cpu_affinity([cpu])
                            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err, text=True)
                        finally:
                            parent.cpu_affinity(affinity)
                        stdout, _ = process.communicate()
                    if process.returncode:
                        raise RuntimeError("Native diagnostic failed: " + label)
                    metrics = json.loads(stdout)
                    summaries = summarize(metrics, args.runs)
                    quality = parity(np.fromfile(raw, dtype="<f4").reshape(len(order), 768), expected[order])
                    record["passes"].append(dict(label=label, engine=engine, cpu=cpu, input_order=order,
                                                 parity=quality, metrics=metrics, summary=summaries))
                    save()
                    print(label, "wall_ms", [round(x["wall_median_ms"], 2) for x in summaries],
                          "cpu_ms", [round(x["cpu_median_ms"], 2) for x in summaries], flush=True)
    if any(digest(Path(p)) != value for p, value in hashes.items()):
        raise ValueError("Diagnostic inputs changed")
    record["complete"] = True
    save()
    print("Complete; diagnostic only:", report, flush=True)


if __name__ == "__main__":
    main()
