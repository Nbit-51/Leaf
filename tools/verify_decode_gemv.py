"""Check every cached one-token logit; this measures correctness, not latency."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.affinity import pin_cpu
from leaf.power import keep_awake
from tools.decoder_validation import quality, run_native
from tools.validate_decoder import digest, native_policy, utc_now, write_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Require a new output path")
    artifact = args.workdir / "decoder-32.leaf"
    token_path, reference_path = args.workdir / "tokens.json", args.workdir / "reference.npy"
    paths = [args.before, args.after, artifact, token_path, reference_path]
    hashes = {str(path): digest(path) for path in paths}
    sequences = json.loads(token_path.read_text())["sequences"]
    reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    pin_cpu(args.cpu)
    with keep_awake():
        before, _ = run_native(args.before, artifact, sequences, mode="chunked", chunk=1, threads=1)
        print("BEFORE cached one-token logits collected", flush=True)
        after, metrics = run_native(args.after, artifact, sequences, mode="chunked", chunk=1, threads=1)
    before, after = before.reshape(reference.shape), after.reshape(reference.shape)
    bit_exact = bool(np.array_equal(before.view(np.uint32), after.view(np.uint32)))
    allclose = bool(np.allclose(after, reference, atol=2e-3, rtol=2e-3))
    measured = quality(after, reference, sequences)
    if any(digest(Path(path)) != value for path, value in hashes.items()):
        raise ValueError("Correctness inputs changed")
    passed = bit_exact and allclose and measured["next_token_agreement"] >= .999 and measured["perplexity_ratio"] <= 1.001
    record = dict(benchmark="decode-gemv-cached-one-token-parity", measured_at_utc=utc_now(),
                  hashes=hashes, native_policy=native_policy(), pinned_cpu=args.cpu, threads=1,
                  chunk_tokens=1, output_rows=len(after), quality=measured,
                  bit_exact_before_after=bit_exact, fp32_allclose_pytorch=allclose,
                  max_abs_before_after=float(np.max(np.abs(after - before))),
                  passed=passed, execution=metrics,
                  scope="All held-out tokens decoded one at a time; no latency measurement or performance qualification.")
    write_record(args.output, record)
    print(json.dumps({k: record[k] for k in ("passed", "bit_exact_before_after", "fp32_allclose_pytorch", "quality")}), flush=True)
    if not passed:
        raise AssertionError("Cached one-token GEMV parity failed")


if __name__ == "__main__":
    main()
