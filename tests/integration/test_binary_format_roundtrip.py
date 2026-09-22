"""
Verifies Leaf's binary graph format (export_binary.py / read_binary.py)
round-trips losslessly: export a real graph, read it back with an
independent parser, and check node structure + initializer weights are
bit-exact. This is the correctness proof for the Python<->C++ handoff
format, checked in Python before any C++ parser exists to compare against.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "graph_opt"))

from ir import Graph
from export_binary import export_graph
from read_binary import read_graph

torchvision = pytest.importorskip("torchvision")


@pytest.fixture(scope="module")
def resnet18_leaf_path(tmp_path_factory):
    import torchvision.models as models

    torch.manual_seed(123)
    model = models.resnet18(weights=None)
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)

    onnx_path = tmp_path_factory.mktemp("onnx") / "resnet18.onnx"
    torch.onnx.export(
        model, dummy_input, str(onnx_path),
        input_names=["input"], output_names=["output"],
        do_constant_folding=True, opset_version=13, dynamo=False,
    )

    leaf_path = tmp_path_factory.mktemp("leaf") / "resnet18.leaf"
    original = Graph.from_onnx(str(onnx_path))
    export_graph(original, str(leaf_path))

    return original, leaf_path


def test_binary_format_roundtrips_losslessly(resnet18_leaf_path):
    original, leaf_path = resnet18_leaf_path
    loaded = read_graph(str(leaf_path))

    assert len(loaded["nodes"]) == len(original.nodes)
    assert loaded["inputs"] == original.inputs
    assert loaded["outputs"] == original.outputs

    for orig_node, read_node in zip(original.nodes, loaded["nodes"]):
        assert orig_node.op_type == read_node["op_type"]
        assert orig_node.inputs == read_node["inputs"]
        assert orig_node.outputs == read_node["outputs"]
        assert orig_node.attributes == read_node["attributes"]

    assert set(loaded["initializers"].keys()) == set(original.initializers.keys())

    for name, orig_array in original.initializers.items():
        read_array = loaded["initializers"][name]
        assert orig_array.shape == read_array.shape
        diff = np.max(np.abs(orig_array.astype(np.float32) - read_array))
        assert diff == 0.0, f"initializer '{name}' not bit-exact (max diff: {diff})"
