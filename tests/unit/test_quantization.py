import numpy as np

from tools.graph_opt.executor import run_graph
from tools.graph_opt.quantization import calibrate, quantize_int8_per_channel
from tools.graph_opt.transformer import run_transformer_rewrites
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
