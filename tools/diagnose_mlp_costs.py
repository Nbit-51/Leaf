"""Serial diagnostic ABBA: operation breakdown and per-forward thread counters.

Not acceptance timing. GetThreadTimes accounting is coarse; raw cycles must
not be converted to elapsed time. Keep frozen uninstrumented binaries for gates.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.affinity import pin_cpu
from leaf.power import keep_awake
from tools.decoder_validation import run_native
from tools.validate_decoder import digest, utc_now, write_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "artifact", "tokens", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--runs", type=int, default=61)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or args.runs < 5 or args.warmup < 0:
        parser.error("Require a fresh output, at least five runs and nonnegative warmup")
    paths = [args.before, args.after, args.artifact, args.tokens, Path(__file__),
             ROOT / "tests/native/decoder_diagnostic.h", ROOT / "engine/src/decoder.cpp",
             ROOT / "engine/src/main_decoder.cpp"]
    hashes = {str(p): digest(p) for p in paths}
    ids = json.loads(args.tokens.read_text())["benchmark_ids"]
    pin_cpu(args.cpu)
    os.environ["LEAF_EXPERIMENTAL_FLOAT_TILES"] = "1"
    os.environ["LEAF_DECODER_PROFILE"] = "1"
    record = dict(benchmark="mlp-cost-diagnostic-abba", measured_at_utc=utc_now(),
                  hashes=hashes, runs=args.runs, warmup=args.warmup, cpu=args.cpu, threads=1,
                  acceptance_timing=False, automatic_selection_authorized=False, passes=[],
                  scope=__doc__)
    with keep_awake():
        for stage in ("before", "after", "after", "before"):
            binary = args.before if stage == "before" else args.after
            _, metrics = run_native(binary, args.artifact, [ids], mode="bench", threads=1,
                                    runs=args.runs, warmup=args.warmup, capture_profile=True)
            if metrics.get("diagnostic_timing_build") is not True:
                raise ValueError("Require diagnostic timing builds")
            summary = {}
            for phase in ("prefill", "decode"):
                wall = metrics[phase + "_samples_ms"]
                cpu = metrics["thread_cpu_" + phase + "_ms"]
                cycles = metrics["thread_cycles_" + phase]
                if any(len(x) != args.runs for x in (wall, cpu, cycles)):
                    raise ValueError("Diagnostic sample counts differ")
                slowest = sorted(range(len(wall)), key=wall.__getitem__, reverse=True)[:5]
                summary[phase] = dict(wall_median_ms=statistics.median(wall),
                    cpu_median_ms=statistics.median(cpu), cycles_median=statistics.median(cycles),
                    slowest=[dict(index=i, wall_ms=wall[i], cpu_ms=cpu[i], cycles=cycles[i]) for i in slowest],
                    linear_mean_ms={k:v["ms"] / (args.runs + args.warmup)
                                    for k,v in metrics["linear_diagnostic"][phase].items()})
            record["passes"].append(dict(stage=stage, raw=metrics, summary=summary))
            print(stage, json.dumps(summary), flush=True)
    if any(digest(Path(p)) != h for p,h in hashes.items()):
        raise ValueError("Diagnostic inputs changed")
    write_record(args.output, record)


if __name__ == "__main__":
    main()
