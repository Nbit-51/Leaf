"""
Sanity test for the Graph IR loader.

Builds a small synthetic ONNX model by hand (Conv -> BatchNorm -> ReLU,
the exact pattern the fusion pass in section 4.2 of the README will
target) so this test has zero dependency on PyTorch or any pretrained
model. Run directly with `python test_ir_loader.py` or via pytest.
"""

import sys
import os
import numpy as np
import onnx
from onnx import helper, TensorProto

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "graph_opt"))
from ir import Graph


def build_conv_bn_relu_model() -> onnx.ModelProto:
    """Conv(3->8, 3x3) -> BatchNormalization -> ReLU on a 1x3x8x8 input."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 8, 8])

    # Conv weights/bias
    w = np.random.randn(8, 3, 3, 3).astype(np.float32)
    b = np.zeros(8, dtype=np.float32)
    w_init = helper.make_tensor("conv_w", TensorProto.FLOAT, w.shape, w.flatten())
    b_init = helper.make_tensor("conv_b", TensorProto.FLOAT, b.shape, b.flatten())

    # BatchNorm params
    scale = np.ones(8, dtype=np.float32)
    bn_bias = np.zeros(8, dtype=np.float32)
    mean = np.zeros(8, dtype=np.float32)
    var = np.ones(8, dtype=np.float32)
    scale_init = helper.make_tensor("bn_scale", TensorProto.FLOAT, scale.shape, scale.flatten())
    bn_bias_init = helper.make_tensor("bn_bias", TensorProto.FLOAT, bn_bias.shape, bn_bias.flatten())
    mean_init = helper.make_tensor("bn_mean", TensorProto.FLOAT, mean.shape, mean.flatten())
    var_init = helper.make_tensor("bn_var", TensorProto.FLOAT, var.shape, var.flatten())

    conv_node = helper.make_node(
        "Conv", inputs=["x", "conv_w", "conv_b"], outputs=["conv_out"],
        name="conv1", kernel_shape=[3, 3], pads=[1, 1, 1, 1],
    )
    bn_node = helper.make_node(
        "BatchNormalization",
        inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        outputs=["bn_out"], name="bn1",
    )
    relu_node = helper.make_node("Relu", inputs=["bn_out"], outputs=["y"], name="relu1")

    graph_def = helper.make_graph(
        [conv_node, bn_node, relu_node],
        "conv_bn_relu_test",
        [x],
        [y],
        initializer=[w_init, b_init, scale_init, bn_bias_init, mean_init, var_init],
    )

    model = helper.make_model(graph_def, producer_name="leaf-test")
    model.opset_import[0].version = 13
    onnx.checker.check_model(model)
    return model


def test_load_and_structure():
    model = build_conv_bn_relu_model()
    g = Graph.from_onnx(model)

    assert len(g.nodes) == 3, f"expected 3 nodes, got {len(g.nodes)}"
    assert g.inputs == ["x"], f"unexpected inputs: {g.inputs}"
    assert g.outputs == ["y"], f"unexpected outputs: {g.outputs}"
    assert len(g.initializers) == 6, f"expected 6 initializers, got {len(g.initializers)}"
    assert g.initializers["conv_w"].shape == (8, 3, 3, 3)

    op_counts = g.op_counts()
    assert op_counts == {"Conv": 1, "BatchNormalization": 1, "Relu": 1}, op_counts

    print("PASS: structure test")
    print(g.summary())


def test_topological_order():
    model = build_conv_bn_relu_model()
    g = Graph.from_onnx(model)

    order = g.topological_order()
    op_sequence = [n.op_type for n in order]
    assert op_sequence == ["Conv", "BatchNormalization", "Relu"], op_sequence

    print("PASS: topological order test ->", op_sequence)


def test_cycle_detection():
    # Manually construct a graph with a cycle: A -> B -> A, which should
    # never occur from a valid ONNX load, but a buggy fusion pass could
    # produce one -- the loader/traversal should fail loudly, not hang.
    from ir import Node

    g = Graph()
    g.nodes = [
        Node(name="A", op_type="Foo", inputs=["b_out"], outputs=["a_out"]),
        Node(name="B", op_type="Bar", inputs=["a_out"], outputs=["b_out"]),
    ]

    try:
        g.topological_order()
        raise AssertionError("expected ValueError for cyclic graph, got none")
    except ValueError as e:
        print("PASS: cycle detection ->", e)


if __name__ == "__main__":
    test_load_and_structure()
    print()
    test_topological_order()
    print()
    test_cycle_detection()
    print("\nAll tests passed.")
