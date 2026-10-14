"""Full trained CIFAR-10 accuracy/parity and unoptimized/optimized CPU latency.

The runner is dataset independent; this benchmark supplies a trained ResNet-20
and all 10,000 held-out CIFAR images. Calibration uses the separate train split.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock

from tools.graph_opt.constant_folding import fold_constants
from tools.graph_opt.export_binary import export_graph
from tools.graph_opt.fusion import run_fusion_passes
from tools.graph_opt.ir import Graph
from tools.graph_opt.memory_planner import plan_memory
from tools.graph_opt.quantization import calibrate, quantize_int8_per_channel
from tools.validate_decoder import digest
from leaf.affinity import pin_cpu
from leaf.power import keep_awake

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)[:, None, None]
STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)[:, None, None]
WEIGHT_SOURCE = "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt"


class CifarResNet20(nn.Module):
    """Checkpoint-compatible CIFAR geometry using torchvision residual blocks."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        channels = 16
        for stage, width in enumerate((16, 32, 64), start=1):
            blocks = []
            for block in range(3):
                stride = 2 if stage > 1 and block == 0 else 1
                shortcut = nn.Sequential(nn.Conv2d(channels, width, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(width)) if channels != width else None
                blocks.append(BasicBlock(channels, width, stride, downsample=shortcut))
                channels = width
            setattr(self, f"layer{stage}", nn.Sequential(*blocks))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        for layer in (self.layer1, self.layer2, self.layer3):
            x = layer(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))


def images(path: Path, limit: int | None = None):
    table = pq.read_table(path, columns=["img", "label"])
    rows = table.slice(0, limit) if limit else table
    result = []
    for encoded in rows["img"].to_pylist():
        value = np.asarray(Image.open(BytesIO(encoded["bytes"])).convert("RGB"), dtype=np.float32)
        result.append((value.transpose(2, 0, 1) / 255.0 - MEAN) / STD)
    return np.stack(result).astype(np.float32), np.asarray(rows["label"].to_pylist(), dtype="<u4")


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("benchmark/data/cifar10-test.parquet"))
    parser.add_argument("--calibration-dataset", type=Path, default=Path("benchmark/data/cifar10-train.parquet"))
    parser.add_argument("--checkpoint", type=Path, default=Path("benchmark/data/cifar10_resnet20-4118986f.pt"))
    parser.add_argument("--executable", type=Path, default=Path("build/leaf_dataset.exe"))
    parser.add_argument("--workdir", type=Path, default=Path("build/cifar10_full"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/cifar10_full_trained.json"))
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--cpu", type=int, help="optional logical CPU pin, inherited by native comparisons")
    parser.add_argument("--skip-int8", action="store_true")
    args = parser.parse_args()
    pin_cpu(args.cpu)
    if args.samples <= 0:
        parser.error("samples must be positive")
    if digest(args.dataset) == digest(args.calibration_dataset):
        raise ValueError("Calibration and held-out datasets must be separate splits")
    torch.set_num_threads(1)
    args.workdir.mkdir(parents=True, exist_ok=True)
    checkpoint_digest = digest(args.checkpoint)
    if not checkpoint_digest.startswith("4118986f"):
        raise ValueError("Checkpoint hash does not match the published ResNet-20 weights")
    model = CifarResNet20().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    inputs, labels = images(args.dataset, args.samples)
    if len(inputs) != args.samples:
        raise ValueError("Incomplete CIFAR-10 test split")
    input_path, expected_path, label_path = [args.workdir / name for name in ("inputs.bin", "expected.bin", "labels.u32")]
    inputs.tofile(input_path); labels.tofile(label_path)
    expected = []
    times = []
    with torch.inference_mode():
        for _ in range(20):
            model(torch.from_numpy(inputs[:1]))
        for index, sample in enumerate(inputs):
            tensor = torch.from_numpy(sample[None])
            start = time.perf_counter_ns(); logits = model(tensor)
            times.append((time.perf_counter_ns() - start) / 1e6)
            expected.append(logits.numpy().copy())
            if (index + 1) % 1000 == 0:
                print(f"PyTorch evaluated {index + 1}/{len(inputs)}", flush=True)
    expected = np.concatenate(expected)
    expected.astype("<f4").tofile(expected_path)
    accuracy = float(np.mean(np.argmax(expected, axis=1) == labels))
    onnx_path = args.workdir / "resnet20.onnx"
    torch.onnx.export(model, torch.from_numpy(inputs[:1]), str(onnx_path),
                      input_names=["input"], output_names=["output"], opset_version=13,
                      do_constant_folding=True, dynamo=False)
    graph = Graph.from_onnx(onnx_path)
    artifacts = {}
    for name, candidate in (("unfused_fp32", graph),
                             ("fused_fp32", run_fusion_passes(fold_constants(graph)))):
        artifact = args.workdir / (name + ".leaf")
        export_graph(candidate, artifact)
        artifacts[name] = artifact
    fused = run_fusion_passes(fold_constants(graph))
    planned = args.workdir / "planned_fp32.leaf"
    export_graph(fused, planned, memory_plan=plan_memory(fused))
    artifacts["planned_fp32"] = planned
    if not args.skip_int8:
        calibration, _ = images(args.calibration_dataset, 64)
        ranges = calibrate(fused, [{"input": sample[None]} for sample in calibration])
        quantized = quantize_int8_per_channel(fused, ranges)
        artifact = args.workdir / "int8.leaf"
        export_graph(quantized, artifact)
        artifacts["int8"] = artifact
    measurements = {}
    # Alternating order checks changes against the same native baseline.
    order = list(artifacts) + list(reversed(artifacts))
    for index, name in enumerate(order):
        metrics_path = args.workdir / f"{name}-{index}.json"
        subprocess.run([str(args.executable.resolve()), str(artifacts[name].resolve()), str(input_path.resolve()),
                        str(expected_path.resolve()), str(label_path.resolve()), "1,3,32,32", str(args.samples),
                        str(metrics_path.resolve()), "0" if name == "int8" else "1"], check=True)
        measurements.setdefault(name, []).append(json.loads(metrics_path.read_text()))
    # Bracket the alternating native passes with an independent PyTorch pass.
    # Images have different inherent costs, so stability compares whole-pass
    # medians rather than treating per-image variation as timing noise.
    repeat_times = []
    with torch.inference_mode():
        for _ in range(20):
            model(torch.from_numpy(inputs[:1]))
        for index, sample in enumerate(inputs):
            tensor = torch.from_numpy(sample[None])
            start = time.perf_counter_ns(); model(tensor)
            repeat_times.append((time.perf_counter_ns() - start) / 1e6)
            if (index + 1) % 1000 == 0:
                print(f"PyTorch repeat evaluated {index + 1}/{len(inputs)}", flush=True)
    torch_medians = [statistics.median(times), statistics.median(repeat_times)]
    torch_latency = statistics.median(torch_medians)
    torch_stable = max(torch_medians) / min(torch_medians) <= 1.25
    baseline_latency = statistics.median(case["p50_ms"] for case in measurements["unfused_fp32"])
    baseline_medians = [case["p50_ms"] for case in measurements["unfused_fp32"]]
    baseline_stable = max(baseline_medians) / min(baseline_medians) <= 1.25
    gates = {}
    for name, runs in measurements.items():
        median = statistics.median(case["p50_ms"] for case in runs)
        quality_passed = all(case["accuracy"] >= accuracy - 0.005 for case in runs)
        if name != "int8":
            quality_passed = quality_passed and all(case["parity_failures"] == 0 for case in runs)
        medians = [case["p50_ms"] for case in runs]
        stable = max(medians) / min(medians) <= 1.25 and baseline_stable and torch_stable
        gates[name] = {"quality_passed": quality_passed, "p50_ms": median,
                       "latency_ratio_vs_native_baseline": median / baseline_latency,
                       "latency_ratio_vs_pytorch": median / torch_latency,
                       "timing_stability_passed": stable,
                       "accepted_native_improvement": name != "unfused_fp32" and quality_passed and stable and median < baseline_latency * 0.98,
                       "accepted_improvement": name != "unfused_fp32" and quality_passed and stable and
                           median < baseline_latency * 0.98 and median < torch_latency * 0.98}
    record = {"benchmark": "full-cifar10-trained-resnet20", "measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "cpu": platform.processor(), "pinned_cpu": args.cpu,
              "torch": torch.__version__, "threads": 1,
              "samples": args.samples, "model": "CIFAR ResNet-20, published trained weights",
              "weights_source": WEIGHT_SOURCE, "weights_sha256": checkpoint_digest,
              "dataset_sha256": digest(args.dataset), "calibration_dataset_sha256": digest(args.calibration_dataset),
              "native_executable_sha256": digest(args.executable),
              "normalization": {"mean": MEAN.reshape(-1).tolist(), "std": STD.reshape(-1).tolist()},
              "calibration_split": "first 64 training images; test split is never used for calibration",
              "pytorch": {"accuracy": accuracy, "p50_ms": torch_latency,
                           "pass_medians_ms": torch_medians, "timing_stability_passed": torch_stable,
                           "latency_samples_ms": [times, repeat_times],
                           "mean_ms": statistics.mean(times + repeat_times),
                           "p95_ms": float(np.percentile(times + repeat_times, 95))},
              "native_runs": measurements, "gates": gates,
              "scope": "All requested held-out images; batch 1 forward latency, excluding image decoding and process startup."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"samples": args.samples, "pytorch_accuracy": accuracy,
                      "pytorch_p50_ms": torch_latency, "gates": gates,
                      "record": str(args.output)}, indent=2))


def main():
    with keep_awake():
        _main()


if __name__ == "__main__":
    main()
