"""
End-to-end numerical verification: a real ResNet-18 architecture through
Leaf's IR + reference executor, checked against PyTorch's own forward
pass on the same input.

This is the test that actually proves correctness, not just structure.
A passing fusion test only proves node counts changed as expected; this
proves the executed graph produces the same numbers PyTorch does, across
every op ResNet-18 needs: Conv, Relu, Add (residual connections),
MaxPool, GlobalAveragePool, Flatten, Gemm (final FC head).

Uses deterministic randomly initialized weights so the test is offline and
CI-safe. It requires torchvision and is skipped automatically if unavailable.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "graph_opt"))

from ir import Graph
from executor import run_graph

torchvision = pytest.importorskip("torchvision")


@pytest.fixture(scope="module")
def resnet18_onnx_path(tmp_path_factory):
    import torchvision.models as models

    torch.manual_seed(123)
    model = models.resnet18(weights=None)
    model.eval()

    # A CIFAR-sized input preserves the complete ResNet operator graph while
    # keeping the deliberately simple NumPy convolution oracle fast enough for CI.
    dummy_input = torch.randn(1, 3, 32, 32)
    onnx_path = tmp_path_factory.mktemp("onnx") / "resnet18.onnx"

    torch.onnx.export(
        model, dummy_input, str(onnx_path),
        input_names=["input"], output_names=["output"],
        do_constant_folding=True, opset_version=13, dynamo=False,
    )

    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    return onnx_path, dummy_input.numpy(), torch_out


def test_resnet18_matches_pytorch_numerically(resnet18_onnx_path):
    onnx_path, input_array, torch_out = resnet18_onnx_path

    graph = Graph.from_onnx(str(onnx_path))
    leaf_tensors = run_graph(graph, {"input": input_array})
    leaf_out = leaf_tensors[graph.outputs[0]]

    assert leaf_out.shape == torch_out.shape

    max_abs_diff = np.max(np.abs(torch_out - leaf_out))
    assert np.allclose(torch_out, leaf_out, atol=1e-3, rtol=1e-3), (
        f"Leaf output diverges from PyTorch beyond tolerance "
        f"(max abs diff: {max_abs_diff:.8f})"
    )


def test_resnet18_graph_has_expected_ops(resnet18_onnx_path):
    onnx_path, _, _ = resnet18_onnx_path
    graph = Graph.from_onnx(str(onnx_path))
    counts = graph.op_counts()

    # ResNet-18's known op signature -- guards against a torch/export
    # version bump silently changing the graph shape under us.
    assert counts.get("Conv", 0) == 20
    assert counts.get("Add", 0) == 8  # one per residual connection
    assert counts.get("Gemm", 0) == 1  # final FC head
