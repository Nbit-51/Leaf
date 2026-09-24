import numpy as np
from onnx import numpy_helper
import torch

from tools.graph_opt.executor import run_graph
from tools.graph_opt.fusion import fuse_attention
from tools.graph_opt.ir import Graph, Node, TensorInfo


def test_attention_reference_mask_polarity_and_fully_masked_row():
    rng = np.random.default_rng(90)
    query = rng.normal(size=(1, 2, 3, 8)).astype(np.float32)
    key = rng.normal(size=(1, 2, 4, 8)).astype(np.float32)
    value = rng.normal(size=(1, 2, 4, 8)).astype(np.float32)
    valid = np.array([[[[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]]]],
                     dtype=np.float32)
    for inverted in (False, True):
        graph = Graph()
        graph.inputs, graph.outputs = ["query"], ["output"]
        graph.initializers = {"key": key, "value": value,
                              "mask": 1.0 - valid if inverted else valid}
        graph.nodes = [Node("attention", "Attention", ["query", "key", "value", "mask"],
                            ["output"], attributes={"scale": 8 ** -0.25,
                                                    "mask_nonzero_is_valid": int(not inverted)})]
        graph.value_info = {"query": TensorInfo("query", query.shape, "float32"),
                            "output": TensorInfo("output", (1, 3, 2, 8), "float32")}
        actual = run_graph(graph, {"query": query})["output"]
        with torch.inference_mode():
            tq, tk, tv = (torch.from_numpy(array) for array in (query, key, value))
            scores = (tq * 8 ** -0.25) @ (tk * 8 ** -0.25).transpose(-1, -2)
            scores = scores.masked_fill(torch.from_numpy(valid) == 0, float("-inf"))
            expected = torch.nan_to_num(torch.softmax(scores, -1), nan=0.0) @ tv
            expected = expected.permute(0, 2, 1, 3).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        assert np.all(actual[:, 2] == 0.0)


def test_attention_fusion_preserves_where_mask_polarity():
    def constant(name, value):
        return Node(name, "Constant", [], [name],
                    {"value": numpy_helper.from_array(np.asarray(value, dtype=np.float32))})

    for inverted in (False, True):
        graph = Graph()
        graph.inputs, graph.outputs = ["q", "k", "v", "mask"], ["output"]
        graph.nodes = [
            constant("scale", 0.5),
            constant("negative_infinity", -np.inf),
            constant("zero", 0.0),
            Node("scaled_query", "Mul", ["q", "scale"], ["qs"]),
            Node("key_transpose", "Transpose", ["k"], ["kt"], {"perm": [0, 1, 3, 2]}),
            Node("scaled_key", "Mul", ["kt", "scale"], ["ks"]),
            Node("scores", "MatMul", ["qs", "ks"], ["scores"]),
            Node("mask_bias", "Where", ["mask", "zero" if inverted else "negative_infinity",
                                        "negative_infinity" if inverted else "zero"], ["bias"]),
            Node("masked_scores", "Add", ["scores", "bias"], ["masked"]),
            Node("softmax", "Softmax", ["masked"], ["probabilities"]),
            Node("is_nan", "IsNaN", ["probabilities"], ["nan_flags"]),
            Node("nan_cleanup", "Where", ["nan_flags", "zero", "probabilities"], ["safe"]),
            Node("weighted", "MatMul", ["safe", "v"], ["weighted"]),
            Node("transpose_output", "Transpose", ["weighted"], ["output"],
                 {"perm": [0, 2, 1, 3]}),
        ]
        rewritten = fuse_attention(graph)
        attention_nodes = [node for node in rewritten.nodes if node.op_type == "Attention"]
        assert len(attention_nodes) == 1
        assert attention_nodes[0].attributes["mask_nonzero_is_valid"] == int(inverted)
