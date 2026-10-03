"""Compare default and packed FP32 phase costs; never authorize promotion.

The native profiler aggregates ALL forwards, including warmup and first use.
Reported phase means are normalized by forward calls, not operation calls.
Run unprofiled whole-model quality/ABBA validation separately for acceptance.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.affinity import pin_cpu
from leaf.power import keep_awake
from tools.decoder_validation import run_native
from tools.validate_decoder import digest, utc_now, write_record


@contextmanager
def profile_policy(packed: bool):
    names = {name for name in os.environ if name.startswith("LEAF_EXPERIMENTAL_")}
    names.update(("LEAF_DISABLE_VNNI", "LEAF_DECODER_PROFILE", "LEAF_EXPERIMENTAL_FLOAT_TILES"))
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
        os.environ["LEAF_DECODER_PROFILE"] = "1"
        if packed:
            os.environ["LEAF_EXPERIMENTAL_FLOAT_TILES"] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def summarize_profile(profile: dict, forwards: int, prefill_tokens: int) -> dict:
    result = {}
    for bucket, tokens in (("prefill", prefill_tokens), ("decode", 1)):
        phases = profile["timings"][bucket]
        forward = phases["forward"]
        if forward["calls"] != forwards or forward["tokens"] != forwards * tokens:
            raise ValueError("Profile forward counts do not match the requested workload")
        total = forward["milliseconds"]
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Profile forward time must be finite and positive")
        summary = {}
        for name, phase in phases.items():
            value = phase["milliseconds"]
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid profile phase time")
            summary[name] = {"mean_ms_per_forward": value / forwards,
                             "percent_of_forward": 100 * value / total}
        # Forward is inclusive; do not add it to the component breakdown.
        covered = sum(p["milliseconds"] for name, p in phases.items() if name != "forward")
        summary["unattributed"] = {"mean_ms_per_forward": (total - covered) / forwards,
                                   "percent_of_forward": 100 * (total - covered) / total}
        result[bucket] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True, help="FP32 decoder artifact")
    parser.add_argument("--tokens", type=Path, required=True, help="frozen tokens.json with benchmark_ids")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--runs", type=int, default=11)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.threads < 1:
        parser.error("runs/threads must be positive and warmup nonnegative")
    ids = json.loads(args.tokens.read_text(encoding="utf-8"))["benchmark_ids"]
    if (not isinstance(ids, list) or len(ids) < 3 or
            any(type(value) is not int or not 0 <= value < 2**32 for value in ids)):
        parser.error("benchmark_ids must contain at least three uint32 token IDs")
    paths = {"executable": args.executable, "artifact": args.artifact, "tokens": args.tokens}
    hashes = {name: digest(path) for name, path in paths.items()}
    record = {"benchmark": "decoder-phase-diagnostics-v1", "measured_at_utc": utc_now(),
              "platform": platform.platform(), "cpu": platform.processor(),
              "threads": args.threads, "pinned_cpu": args.cpu, "sha256": hashes,
              "prefill_tokens": len(ids) - 1, "decode_tokens": 1,
              "logits": "last_token_only", "runs": args.runs, "warmup": args.warmup,
              "profile_includes_warmup": True, "profile_includes_first_use": True,
              "automatic_selection_authorized": False, "acceptance_timing": False,
              "order": ["default", "token_panel", "token_panel", "default"], "passes": []}
    pin_cpu(args.cpu)
    with keep_awake():
        for stage in record["order"]:
            with profile_policy(stage == "token_panel"):
                _, measured = run_native(args.executable, args.artifact, [ids], mode="bench",
                                         threads=args.threads, runs=args.runs, warmup=args.warmup,
                                         activation_bits=32, capture_profile=True)
            phases = measured["diagnostic_profile"]["timings"]
            if any(name.startswith("linear_w") for bucket in phases.values() for name in bucket):
                raise ValueError("Phase comparison requires an FP32 artifact")
            summary = summarize_profile(measured["diagnostic_profile"], args.runs + args.warmup, len(ids) - 1)
            record["passes"].append({"stage": stage, "summary": summary, "raw": measured})
            print(stage, json.dumps(summary["prefill"]), flush=True)
    if hashes != {name: digest(path) for name, path in paths.items()}:
        raise ValueError("Diagnostic inputs changed during execution")
    write_record(args.output, record)


if __name__ == "__main__":
    main()
