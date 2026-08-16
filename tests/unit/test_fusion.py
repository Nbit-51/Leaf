"""
Fusion correctness test.

Builds the same Conv->BatchNorm->Relu synthetic graph used in
test_ir_loader.py, runs it through the reference executor BEFORE and
AFTER fusion on identical random input, and asserts the outputs match
to floating-point tolerance. This is the actual correctness proof for
the fusion pass -- structural checks (node count, op types) are useful
sanity checks but do not prove the rewrite preserves behavior.
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "graph_opt"))
from ir import Graph
from fusion import run_fusion_passes, fuse_conv_batchnorm, fuse_conv_activation
from executor import run_graph

sys.path.insert(0, os.path.dirname(__file__))
from test_ir_loader import build_conv_bn_relu_model


def _random_input():
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((1, 3, 8, 8)).astype(np.float32)


def test_fused_graph_matches_reference_numerically():
    model = build_conv_bn_relu_model()
    graph = Graph.from_onnx(model)

    x = _random_input()

    baseline = run_graph(graph, {"x": x})
    y_baseline = baseline["y"]

    fused_graph = run_fusion_passes(graph)
    fused = run_graph(fused_graph, {"x": x})
    y_fused = fused["y"]

    np.testing.assert_allclose(y_baseline, y_fused, rtol=1e-4, atol=1e-5)
    print(f"PASS: numerical equivalence (max abs diff = {np.max(np.abs(y_baseline - y_fused)):.2e})")


def test_fusion_reduces_node_count():
    model = build_conv_bn_relu_model()
    graph = Graph.from_onnx(model)

    assert len(graph.nodes) == 3
    fused_graph = run_fusion_passes(graph)
    assert len(fused_graph.nodes) == 1, f"expected 1 fused node, got {len(fused_graph.nodes)}"

    fused_node = fused_graph.nodes[0]
    assert fused_node.op_type == "Conv"
    assert fused_node.attributes.get("activation") == "Relu"

    print(f"PASS: 3 nodes -> 1 fused node ({fused_node})")


def test_bn_fold_only():
    """BN folding alone (no activation fusion) should leave 2 nodes:
    a fused Conv and the original Relu, unchanged."""
    model = build_conv_bn_relu_model()
    graph = Graph.from_onnx(model)

    bn_fused = fuse_conv_batchnorm(graph)
    assert len(bn_fused.nodes) == 2, f"expected 2 nodes after BN-only fold, got {len(bn_fused.nodes)}"
    op_types = [n.op_type for n in bn_fused.nodes]
    assert op_types == ["Conv", "Relu"], op_types

    print(f"PASS: BN-only fold -> {op_types}")


def test_no_fusion_when_output_has_multiple_consumers():
    """If the Conv output feeds two different nodes, fusing it away
    would silently drop a required value -- the pass must refuse."""
    from ir import Node

    model = build_conv_bn_relu_model()
    graph = Graph.from_onnx(model)

    # Add a second, unrelated consumer of the Conv's output tensor.
    branch_node = Node(
        name="extra_consumer",
        op_type="Relu",  # arbitrary op, just needs to consume conv_out
        inputs=["conv_out"],
        outputs=["extra_out"],
    )
    graph.nodes.append(branch_node)
    graph.outputs.append("extra_out")  # keep it live so it cannot be pruned away

    fused_graph = fuse_conv_batchnorm(graph)
    op_types = [n.op_type for n in fused_graph.nodes]
    assert "BatchNormalization" in op_types, (
        f"BN should NOT have been fused away (conv_out has 2 consumers), got {op_types}"
    )

    print(f"PASS: fusion correctly skipped (multi-consumer tensor) -> {op_types}")


if __name__ == "__main__":
    test_fused_graph_matches_reference_numerically()
    print()
    test_fusion_reduces_node_count()
    print()
    test_bn_fold_only()
    print()
    test_no_fusion_when_output_has_multiple_consumers()
    print("\nAll fusion tests passed.")
