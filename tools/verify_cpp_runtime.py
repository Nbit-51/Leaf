"""Cross-check Leaf's C++ runtime against PyTorch on a real ResNet-18.

This is a dev-machine verification tool, not part of the target runtime.
It produces a temporary .leaf artifact and raw input, calls ``leaf_infer``,
and compares final logits against eval-mode PyTorch on the exact same input.

Example (from the repository root):
    python tools/verify_cpp_runtime.py --leaf-infer engine/build/leaf_infer
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "graph_opt"))

from export_binary import export_graph
from ir import Graph


def export_resnet18(artifact_path: Path, input_path: Path, expected_path: Path) -> None:
    """Export one deterministic ResNet-18 inference case and its reference."""
    torch.manual_seed(42)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
    input_tensor = torch.randn(1, 3, 224, 224)

    onnx_path = artifact_path.with_suffix(".onnx")
    torch.onnx.export(
        model,
        input_tensor,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        opset_version=13,
        dynamo=False,
    )
    graph = Graph.from_onnx(str(onnx_path))
    export_graph(graph, str(artifact_path))

    with torch.no_grad():
        expected = model(input_tensor).cpu().numpy().astype(np.float32)
    input_tensor.cpu().numpy().astype(np.float32).tofile(input_path)
    expected.tofile(expected_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf-infer", type=Path, required=True,
                        help="path to the C++ leaf_infer executable")
    parser.add_argument("--keep-artifacts", action="store_true",
                        help="keep generated .onnx/.leaf/.bin files and print their directory")
    args = parser.parse_args()

    executable = args.leaf_infer.resolve()
    if not executable.is_file():
        parser.error(f"leaf_infer executable not found: {executable}")

    temporary_directory = tempfile.TemporaryDirectory(prefix="leaf_cpp_parity_")
    work_dir = Path(temporary_directory.name)
    artifact = work_dir / "resnet18.leaf"
    input_path = work_dir / "input.bin"
    expected_path = work_dir / "pytorch_output.bin"
    actual_path = work_dir / "cpp_output.bin"

    export_resnet18(artifact, input_path, expected_path)
    subprocess.run(
        [str(executable), str(artifact), str(input_path), "1,3,224,224", str(actual_path)],
        check=True,
    )

    expected = np.fromfile(expected_path, dtype=np.float32).reshape(1, 1000)
    actual = np.fromfile(actual_path, dtype=np.float32).reshape(1, 1000)
    max_abs_difference = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, rtol=1e-3, atol=1e-3):
        raise AssertionError(
            "Leaf C++ runtime differs from PyTorch "
            f"(max absolute difference: {max_abs_difference:.8f})"
        )

    print("PASS: Leaf C++ ResNet-18 runtime matches PyTorch")
    print(f"  max absolute difference: {max_abs_difference:.8f}")
    if args.keep_artifacts:
        preserved = ROOT / "work" / "cpp_runtime_parity"
        preserved.mkdir(parents=True, exist_ok=True)
        for source in work_dir.iterdir():
            source.replace(preserved / source.name)
        print(f"  artifacts: {preserved}")
    temporary_directory.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
