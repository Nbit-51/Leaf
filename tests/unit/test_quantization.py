import numpy as np

from tools.graph_opt.executor import run_graph
from tools.graph_opt.quantization import calibrate, quantize_int8_per_channel
from tools.graph_opt.transformer import run_transformer_rewrites
from tools.graph_opt.ir import Graph, Node
from tests.unit.test_transformer_rewrites import build_ffn_graph


def test_calibrated_per_channel_int8_has_bounded_error():
    graph = run_transformer_rewrites(build_ffn_graph())
    rng = np.random.default_rng(11)
    calibration_samples = [
        {"x": rng.normal(size=(2, 4, 8)).astype(np.float32)} for _ in range(8)
    ]
    ranges = calibrate(graph, calibration_samples)
    quantized = quantize_int8_per_channel(graph, ranges)

    node = quantized.nodes[0]
    weight = quantized.initializers[node.inputs[1]]
    scales = node.attributes["quantization"]["weight"]["scale"]
    assert weight.dtype == np.int8
    assert scales.shape == (12,)

    test_input = rng.normal(size=(2, 4, 8)).astype(np.float32)
    expected = run_graph(graph, {"x": test_input})["y"]
    actual = run_graph(quantized, {"x": test_input})["y"]
    np.testing.assert_allclose(actual, expected, rtol=0.08, atol=0.04)


def test_calibration_rejects_empty_dataset():
    graph = run_transformer_rewrites(build_ffn_graph())
    try:
        calibrate(graph, [])
        raise AssertionError("expected empty calibration failure")
    except ValueError as error:
        assert "at least one" in str(error)


def test_shared_weight_with_different_output_axes_is_quantized_from_fp32():
    graph = Graph()
    graph.inputs, graph.outputs = ["x"], ["a", "b"]
    original = np.array([[0.1, 0.8], [-0.4, 0.2]], dtype=np.float32)
    graph.initializers = {"weight": original}
    graph.nodes = [
        Node("a", "Gemm", ["x", "weight"], ["a"], {"transB": 1}),
        Node("b", "MatMul", ["x", "weight"], ["b"]),
    ]
    calibration = calibrate(graph, [{"x": np.array([[0.25, -0.5]], dtype=np.float32)}])
    quantized = quantize_int8_per_channel(graph, calibration)
    first, second = quantized.nodes
    assert first.inputs[1] != second.inputs[1]
    assert first.attributes["quantization"]["weight"]["axis"] == 0
    assert second.attributes["quantization"]["weight"]["axis"] == 1
    assert all(value.dtype == np.int8 for value in quantized.initializers.values())
    assert "weight" not in quantized.initializers
    assert graph.initializers["weight"].dtype == np.float32


def test_shared_weight_keeps_fp32_copy_for_unquantized_consumer():
    graph = Graph()
    graph.inputs, graph.outputs = ["x"], ["a", "w_copy"]
    graph.initializers = {"weight": np.eye(2, dtype=np.float32)}
    graph.nodes = [
        Node("a", "MatMul", ["x", "weight"], ["a"]),
        Node("copy", "Identity", ["weight"], ["w_copy"]),
    ]
    calibration = calibrate(graph, [{"x": np.ones((1, 2), dtype=np.float32)}])
    quantized = quantize_int8_per_channel(graph, calibration)
    assert quantized.nodes[0].inputs[1] != "weight"
    assert quantized.initializers["weight"].dtype == np.float32
    assert quantized.initializers[quantized.nodes[0].inputs[1]].dtype == np.int8
