"""Validate the native encoder against frozen trained EmbeddingGemma reference vectors."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.embedding_validation import run_native, parity
from tools.verify_embeddinggemma_reference import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "artifact", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output record")
    saved = json.loads((args.reference / "reference.json").read_text())
    provenance = json.loads(args.artifact.with_suffix(".provenance.json").read_text())
    if not saved.get("passed") or not saved.get("complete"):
        raise ValueError("Reference did not pass")
    if provenance["snapshot_sha256"] != saved["snapshot_sha256"]:
        raise ValueError("Artifact and reference snapshots differ")
    if digest(args.artifact) != provenance["artifact_sha256"]:
        raise ValueError("Artifact hash mismatch")
    for name, value in saved["fixture_sha256"].items():
        if digest(args.reference / name) != value:
            raise ValueError("Reference fixture hash mismatch")
    with np.load(args.reference / "reference.npz", allow_pickle=False) as data:
        sequences = [
            row[mask.astype(bool)].tolist()
            for row, mask in zip(data["input_ids"], data["attention_mask"])
        ]
        sequences.append(
            data["long_input_ids"][0][
                data["long_attention_mask"][0].astype(bool)
            ].tolist()
        )
        expected = np.concatenate([data["embeddings"], data["long_embeddings"]])
    # Repeating the first request after the long one catches stale buffers/state.
    sequences.append(sequences[0])
    expected = np.concatenate([expected, expected[:1]])
    inputs = [
        args.executable,
        args.artifact,
        args.reference / "reference.json",
        Path(__file__),
        Path(__file__).with_name("embedding_validation.py"),
    ]
    hashes = {str(p): digest(p) for p in inputs}
    record = dict(
        format="leaf-trained-embedding-parity-v1",
        measured_at_utc=datetime.now(timezone.utc).isoformat(),
        model=saved["model"],
        hashes=hashes,
        lengths=[len(x) for x in sequences],
        checks=[],
        complete=False,
        timing_qualified=False,
        retrieval_quality_qualified=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    save()
    for threads, scalar in [(1, False), (2, False), (1, True)]:
        actual, metrics = run_native(
            args.executable, args.artifact, sequences, threads=threads, scalar=scalar
        )
        checked = parity(actual, expected)
        if scalar and metrics["avx2"]:
            raise ValueError("Forced scalar path used AVX2")
        record["checks"].append(
            dict(threads=threads, scalar=scalar, metrics=metrics, parity=checked)
        )
        save()
        print(threads, scalar, checked, flush=True)
    if any(digest(Path(p)) != value for p, value in hashes.items()):
        raise ValueError("Validation inputs changed")
    record.update(complete=True, passed=True)
    save()


if __name__ == "__main__":
    main()
