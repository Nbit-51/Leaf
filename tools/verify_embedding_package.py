"""Installed, compiler-free embedding CLI checks against the frozen reference."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.embedding_validation import parity
from tools.verify_embeddinggemma_reference import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("installed", "model", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh directory")
    args.output.mkdir(parents=True)
    saved = json.loads((args.reference / "texts.json").read_text())
    with np.load(args.reference / "reference.npz", allow_pickle=False) as data:
        expected = data["embeddings"]
    dataset = args.output / "documents.jsonl"
    dataset.write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in saved["texts"])
    )
    environment = dict(os.environ)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "CXX",
        "LEAF_EMBED_BIN",
        "LEAF_DECODER_BIN",
    ):
        environment.pop(name, None)
    environment.update(
        LEAF_CACHE_DIR=str((args.output / "cache").resolve()),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    environment["PATH"] = (
        str(Path(os.environ["SystemRoot"]) / "System32") if os.name == "nt" else ""
    )
    child = """
import json, pathlib, shutil, sys
sys.path.insert(0, sys.argv[1])
from leaf import cli, embedding
root=pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(embedding.__file__).resolve().is_relative_to(root)
runtime=embedding.executable(cli.cache_root())
assert runtime.resolve().is_relative_to(root)
assert not shutil.which('g++') and not shutil.which('clang++')
sys.argv=['leaf']+json.loads(sys.argv[2])
status=cli.main()
assert not any(name in sys.modules for name in ('torch','transformers','sentence_transformers'))
raise SystemExit(status)
"""
    checks = []
    for phase in ("first", "cached"):
        output, metrics = (
            args.output / (phase + ".npy"),
            args.output / (phase + ".json"),
        )
        command = [
            "embed",
            str(args.model.resolve()),
            "--dataset",
            str(dataset.resolve()),
            "--task",
            "raw",
            "--output",
            str(output.resolve()),
            "--metrics",
            str(metrics.resolve()),
        ]
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                child,
                str(args.installed.resolve()),
                json.dumps(command),
            ],
            env=environment,
            check=True,
        )
        checked = parity(np.load(output), expected)
        checks.append(
            dict(phase=phase, parity=checked, metrics=json.loads(metrics.read_text()))
        )
    runtime = (
        args.installed
        / "leaf/bin"
        / ("leaf_embed.exe" if os.name == "nt" else "leaf_embed")
    )
    record = dict(
        format="leaf-installed-embedding-smoke-v1",
        passed=True,
        checks=checks,
        native_executable_sha256=digest(runtime),
        bundled_runtime_selected=True,
        compiler_path_available=False,
        heavy_frameworks_imported=False,
        offline=True,
        timing_qualified=False,
        reference_record_sha256=digest(args.reference / "reference.json"),
    )
    (args.output / "package.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        "Installed embedding package passed: first preparation and cached JSONL reuse"
    )


if __name__ == "__main__":
    main()
