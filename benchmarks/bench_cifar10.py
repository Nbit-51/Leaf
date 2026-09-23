"""Real CIFAR-10 data check for Leaf calibration, parity, and latency.

This uses the current small CNN optimizer workload, not a trained classifier,
so it reports numerical error and inference latency rather than accuracy.
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
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from benchmark.workloads import cnn_workload
from tools.graph_opt.executor import run_graph
from tools.graph_opt.pipeline import optimize_graph
from tools.graph_opt.constant_folding import fold_constants
from tools.graph_opt.fusion import run_fusion_passes
from tools.graph_opt.transformer import run_transformer_rewrites
from tools.graph_opt.export_binary import export_graph


def _latency_ms(function, samples, repeats: int) -> float:
    for sample in samples[:2]:
        function(sample)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for sample in samples:
            function(sample)
        timings.append((time.perf_counter_ns() - start) / 1e6 / len(samples))
    return statistics.median(timings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("benchmark/data"))
    parser.add_argument("--dataset-npz", type=Path,
                        default=Path("benchmark/data/cifar10-test-subset.npz"),
                        help="prepared real CIFAR-10 test images; preferred when present")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--evaluation-samples", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--leaf-infer", type=Path, help="optional native inference executable")
    parser.add_argument("--leaf-bench", type=Path, help="optional native graph benchmark executable")
    parser.add_argument("--native-evaluation-samples", type=int, default=20)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark/results/cifar10_cpu.json"))
    args = parser.parse_args()
    if bool(args.leaf_infer) != bool(args.leaf_bench):
        parser.error("--leaf-infer and --leaf-bench must be provided together")
    if min(args.calibration_samples, args.evaluation_samples, args.repeats, args.threads) < 1:
        parser.error("sample counts, repeats, and threads must all be positive")
    torch.set_num_threads(args.threads)
    transform = Compose([
        Resize((16, 16)), ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    needed = args.calibration_samples + args.evaluation_samples
    if args.dataset_npz.is_file():
        with np.load(args.dataset_npz) as archive:
            images, labels_array = archive["images"], archive["labels"]
            if needed > len(images):
                parser.error(f"requested {needed} samples but NPZ contains {len(images)}")
            records = [(transform(Image.fromarray(images[index])), int(labels_array[index]))
                       for index in range(needed)]
        source = str(args.dataset_npz)
    else:
        try:
            dataset = CIFAR10(args.data_root, train=False, transform=transform,
                              download=args.download)
        except RuntimeError as error:
            raise SystemExit(
                "CIFAR-10 is not cached. Prepare a local subset or use --download.\n"
                f"Original error: {error}"
            ) from error
        if needed > len(dataset):
            parser.error(f"requested {needed} samples but the test split contains {len(dataset)}")
        records = [dataset[index] for index in range(needed)]
        source = str(args.data_root)
    samples = [{"x": image.unsqueeze(0).numpy()} for image, _ in records]
    calibration = samples[:args.calibration_samples]
    evaluation = samples[args.calibration_samples:]
    labels = [int(label) for _, label in records[args.calibration_samples:]]

    graph, torch_reference = cnn_workload()
    optimized = optimize_graph(graph, calibration)

    def pytorch_call(sample):
        with torch.inference_mode():
            return torch_reference(torch.from_numpy(sample["x"])).numpy()

    def leaf_fp32_call(sample):
        return run_graph(graph, sample)["y"]

    def leaf_int8_call(sample):
        return run_graph(optimized.graph, sample)["y"]

    fp32_error = 0.0
    int8_error = 0.0
    for sample in evaluation:
        expected = pytorch_call(sample)
        fp32_error = max(fp32_error, float(np.max(np.abs(leaf_fp32_call(sample) - expected))))
        int8_error = max(int8_error, float(np.max(np.abs(leaf_int8_call(sample) - expected))))

    result = {
        "benchmark": "cifar10-leaf-optimizer-micrograph",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "torch": torch.__version__, "numpy": np.__version__},
        "dataset": "CIFAR-10 test split",
        "dataset_source": source,
        "calibration_samples": len(calibration),
        "evaluation_samples": len(evaluation),
        "threads": args.threads,
        "input_shape": [1, 3, 16, 16],
        "label_histogram": {str(label): labels.count(label) for label in sorted(set(labels))},
        "leaf_fp32_vs_pytorch_max_abs": fp32_error,
        "leaf_int8_vs_pytorch_max_abs": int8_error,
        "latency_ms_per_sample": {
            "pytorch_cpu": _latency_ms(pytorch_call, evaluation, args.repeats),
            "leaf_numpy_reference_fp32": _latency_ms(leaf_fp32_call, evaluation, args.repeats),
            "leaf_numpy_quantized_simulation": _latency_ms(leaf_int8_call, evaluation, args.repeats),
        },
        "scope": "Real dataset; untrained optimizer micrograph, so accuracy is intentionally not reported.",
    }
    if args.leaf_infer:
        native_samples = evaluation[:args.native_evaluation_samples]
        if not native_samples:
            parser.error("--native-evaluation-samples must be positive")
        fp32_graph = run_transformer_rewrites(run_fusion_passes(fold_constants(graph)))
        native_result = {"samples": len(native_samples), "warmup": 5, "runs": 10}
        with tempfile.TemporaryDirectory(prefix="leaf_cifar_native_") as directory_name:
            directory = Path(directory_name)
            input_path = directory / "input.bin"
            output_path = directory / "output.bin"
            for label, candidate in (("fp32", fp32_graph), ("int8", optimized.graph)):
                artifact = directory / f"{label}.leaf"
                export_graph(candidate, str(artifact))
                latencies = []
                max_error = 0.0
                for sample in native_samples:
                    value = sample["x"]
                    value.tofile(input_path)
                    command = [str(args.leaf_infer.resolve()), str(artifact), str(input_path),
                               "1,3,16,16", str(output_path)]
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    actual = np.fromfile(output_path, dtype=np.float32).reshape(1, 8, 16, 16)
                    max_error = max(max_error, float(np.max(np.abs(actual - pytorch_call(sample)))))
                    output = subprocess.run(
                        [str(args.leaf_bench.resolve()), str(artifact), str(input_path),
                         "1,3,16,16", "5", "10"],
                        check=True, capture_output=True, text=True).stdout
                    match = re.search(r"p50=([0-9.]+)", output)
                    if match is None:
                        raise RuntimeError("native benchmark did not report p50 latency")
                    latencies.append(float(match.group(1)))
                native_result[label] = {
                    "median_per_image_p50_ms": statistics.median(latencies),
                    "max_abs_vs_pytorch": max_error,
                }
        result["native_graph"] = native_result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
