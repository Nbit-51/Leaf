"""Windows-oriented serial benchmarks of actual decoder GEMV kernels."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.affinity import pin_cpu
from leaf.power import keep_awake
from tools.benchmark_token_panels import summarize
from tools.validate_decoder import digest, utc_now, write_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=int, choices=(1, 2), default=1)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or not 5 <= args.runs <= 10000 or not 0 <= args.warmup <= 10000:
        parser.error("Require a new output, runs 5..10000 and warmup 0..10000")
    paths = [args.executable, ROOT / "engine/src/decoder.cpp",
             ROOT / "benchmark/decode_gemv_benchmark.cpp",
             ROOT / "engine/include/leaf/kernels/gemv_pair.h", Path(__file__)]
    hashes = {str(path): digest(path) for path in paths}
    pin_cpu(args.cpu)
    with keep_awake():
        result = subprocess.run([str(args.executable.resolve()), str(args.candidate),
                                 str(args.runs), str(args.warmup)], check=True, capture_output=True, text=True)
    record = json.loads(result.stdout)
    if record["candidate"] != args.candidate:
        raise ValueError("Benchmark candidate mismatch")
    for shape in record["shapes"]:
        shape["summary"] = summary = summarize(shape["passes"], "optimized")
        for stage in ("full_k", "optimized"):
            seconds = summary[stage]["median_ms"] / 1000
            summary[stage]["gflops"] = 2 * shape["n"] * shape["k"] / seconds / 1e9
            summary[stage]["logical_weight_gb_per_second"] = 4 * shape["n"] * shape["k"] / seconds / 1e9
        print(shape["n"], shape["k"], summary["ratio_optimized_full_k"], summary["stable"], flush=True)
    if any(digest(Path(path)) != value for path, value in hashes.items()):
        raise ValueError("Benchmark inputs changed")
    record.update(benchmark="decoder-gemv-kernel-abba-v1", measured_at_utc=utc_now(),
                  platform=platform.platform(), cpu=platform.processor(), pinned_cpu=args.cpu,
                  threads=1, runs_per_pass=args.runs, warmup_per_pass=args.warmup,
                  input_sha256=hashes, automatic_selection_authorized=False,
                  scope="Actual FP32 decoder dot kernels, reusable output, no bias/allocation/worker overhead; "
                        "full_k label means current dot_avx. Samples are per-matrix means across resident "
                        "replicas with identical synthetic values. Weight GB/s is logical, not measured DRAM traffic.")
    write_record(args.output, record)


if __name__ == "__main__":
    main()
