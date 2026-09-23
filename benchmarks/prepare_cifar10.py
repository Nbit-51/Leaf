"""Prepare a small, offline CIFAR-10 test subset from the HF Parquet mirror.

Download the `uoft-cs/cifar10` test Parquet file into benchmark/data first.
This optional preparation step needs pyarrow and Pillow; the output is ignored
by Git and can be consumed by bench_cifar10.py without pyarrow installed.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("benchmark/data/cifar10-test.parquet"))
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark/data/cifar10-test-subset.npz"))
    parser.add_argument("--samples", type=int, default=132)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if not args.input.is_file():
        parser.error(f"Parquet input is missing: {args.input}")

    data = parquet.read_table(args.input, columns=["img", "label"]).slice(0, args.samples)
    rows = data.to_pylist()
    images = []
    labels = []
    for row in rows:
        with Image.open(BytesIO(row["img"]["bytes"])) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape != (32, 32, 3):
            raise ValueError(f"unexpected CIFAR image shape: {rgb.shape}")
        images.append(rgb)
        labels.append(int(row["label"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, images=np.stack(images),
                        labels=np.asarray(labels, dtype=np.uint8))
    print(json.dumps({"output": str(args.output.resolve()), "samples": len(rows),
                      "shape": list(images[0].shape)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
