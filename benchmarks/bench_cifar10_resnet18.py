"""Benchmark full FP32 ResNet-18 on real CIFAR-10 images in PyTorch and C++.

The weights are seeded and untrained, so the check measures graph execution,
latency, and numerical parity, not classification accuracy.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import tempfile
import time

import numpy as np
import torch
import torchvision.models as models

from tools.verify_cpp_runtime import export_resnet18


_LATENCY = re.compile(
    r"latency ms: min=([0-9.]+), p50=([0-9.]+), p95=([0-9.]+), mean=([0-9.]+)"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-npz", type=Path,
                        default=Path("benchmark/data/cifar10-test-subset.npz"))
    parser.add_argument("--leaf-bench", type=Path, default=Path("build/leaf_graph_bench.exe"))
    parser.add_argument("--leaf-infer", type=Path, default=Path("build/leaf_infer.exe"))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark/results/cifar10_resnet18_cpp.json"))
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 1 or args.runs < 1 or args.threads < 1:
        parser.error("samples, warmup, runs, and threads must be positive")
    if not args.dataset_npz.is_file():
        parser.error(f"prepared CIFAR-10 subset is missing: {args.dataset_npz}")
    if not args.leaf_bench.is_file() or not args.leaf_infer.is_file():
        parser.error("build leaf_bench and leaf_infer before running this benchmark")

    with np.load(args.dataset_npz) as archive:
        images = archive["images"][:args.samples].copy()
        labels = archive["labels"][:args.samples].astype(int).tolist()
    if len(images) != args.samples:
        parser.error(f"requested {args.samples} samples but only {len(images)} are available")

    # The same seeded model is used by export_resnet18.
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    model = models.resnet18(weights=None).eval()
    inputs = [torch.from_numpy(image.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.0
              for image in images]

    torch_p50_ms = []
    native_p50_ms = []
    maximum_error = 0.0
    with tempfile.TemporaryDirectory(prefix="leaf_cifar_resnet18_") as directory:
        folder = Path(directory)
        artifact = folder / "resnet18.leaf"
        export_resnet18(artifact, folder / "dummy_input.bin", folder / "dummy_expected.bin")

        for index, value in enumerate(inputs):
            sample_times_ms = []
            with torch.inference_mode():
                for _ in range(args.warmup):
                    model(value)
                expected = model(value).numpy()
                for _ in range(args.runs):
                    start = time.perf_counter_ns()
                    model(value)
                    sample_times_ms.append((time.perf_counter_ns() - start) / 1e6)
            torch_p50_ms.append(statistics.median(sample_times_ms))

            input_path = folder / f"input_{index}.bin"
            output_path = folder / f"output_{index}.bin"
            value.numpy().astype(np.float32).tofile(input_path)
            subprocess.run([str(args.leaf_infer.resolve()), str(artifact), str(input_path),
                            "1,3,32,32", str(output_path)], check=True, capture_output=True,
                           text=True)
            actual = np.fromfile(output_path, dtype=np.float32).reshape(1, 1000)
            maximum_error = max(maximum_error,
                                float(np.max(np.abs(actual - expected))))
            if not np.allclose(actual, expected, atol=1e-3, rtol=1e-3):
                raise AssertionError(f"C++ output diverged on CIFAR image {index}")

            completed = subprocess.run(
                [str(args.leaf_bench.resolve()), str(artifact), str(input_path),
                 "1,3,32,32", str(args.warmup), str(args.runs)],
                check=True, capture_output=True, text=True,
            )
            match = _LATENCY.search(completed.stdout)
            if match is None:
                raise RuntimeError(f"cannot parse leaf_bench output: {completed.stdout}")
            native_p50_ms.append(float(match.group(2)))

    pytorch_p50 = statistics.median(torch_p50_ms)
    native_p50 = statistics.median(native_p50_ms)
    result = {
        "benchmark": "cifar10-resnet18-fp32-cpp-vs-pytorch",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"platform": platform.platform(), "torch": torch.__version__,
                        "threads": args.threads},
        "dataset": "CIFAR-10 test split, first images in original order",
        "dataset_source": str(args.dataset_npz),
        "model": "ResNet-18, seed 42, untrained weights, eval mode",
        "input_shape": [1, 3, 32, 32],
        "samples": args.samples,
        "warmup_runs_per_image": args.warmup,
        "measured_runs_per_image": args.runs,
        "label_histogram": {str(label): labels.count(label) for label in sorted(set(labels))},
        "pytorch_cpu_p50_ms": pytorch_p50,
        "leaf_cpp_fp32_p50_ms": native_p50,
        "leaf_cpp_over_pytorch_latency_ratio": native_p50 / pytorch_p50,
        "leaf_cpp_vs_pytorch_max_abs": maximum_error,
        "pytorch_per_image_p50_ms": torch_p50_ms,
        "leaf_cpp_per_image_p50_ms": native_p50_ms,
        "scope": "Real images and full graph; random weights mean CIFAR-10 accuracy is undefined.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
