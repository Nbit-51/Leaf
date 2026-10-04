"""Serial attention/shared-QKV shape experiments; no whole-model promotion."""
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
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or not 5 <= args.runs <= 10000 or not 0 <= args.warmup <= 10000:
        parser.error("Require a new output, runs 5..10000 and warmup 0..10000")
    paths = [args.executable, ROOT / "benchmark/decoder_operator_benchmark.cpp",
             ROOT / "engine/include/leaf/kernels/attention_vector.h", ROOT / "engine/include/leaf/kernels/token_panel.h",
             ROOT / "engine/src/kernels/transformer.cpp", Path(__file__)]
    hashes = {str(p): digest(p) for p in paths}
    pin_cpu(args.cpu)
    with keep_awake():
        completed = subprocess.run([str(args.executable.resolve()), str(args.runs), str(args.warmup)],
                                   capture_output=True, text=True, check=True)
    record = json.loads(completed.stdout)
    for case in record["cases"]:
        case["summary"] = summarize(case["passes"], "optimized")
        print(case["name"], case["summary"]["ratio_optimized_full_k"], case["summary"]["stable"])
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("Benchmark source/binary changed")
    record.update(benchmark="decoder-operator-abba-v1", measured_at_utc=utc_now(), platform=platform.platform(),
                  cpu=platform.processor(), pinned_cpu=args.cpu, threads=1, runs=args.runs, warmup=args.warmup,
                  input_sha256=hashes, synthetic_inputs=True, automatic_selection_authorized=False,
                  scope="Packing-inclusive three-projection QKV and causal attention. Default stage label full_k means existing implementation. No model performance claim.")
    write_record(args.output, record)


if __name__ == "__main__":
    main()
