import numpy as np

from tools.graph_opt.executor import run_graph
from tools.graph_opt.ir import Graph, Node, TensorInfo
from tools.graph_opt.transformer import run_transformer_rewrites


def build_ffn_graph(seed=7):
    rng = np.random.default_rng(seed)
    graph = Graph()
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.initializers = {
        "weight": rng.normal(0, 0.2, (8, 12)).astype(np.float32),
        "bias": rng.normal(0, 0.1, (12,)).astype(np.float32),
    }
    graph.nodes = [
        Node("projection", "MatMul", ["x", "weight"], ["projected"]),
        Node("bias", "Add", ["projected", "bias"], ["biased"]),
        Node("gelu", "Gelu", ["biased"], ["y"]),
    ]
    graph.value_info = {
        "x": TensorInfo("x", (2, 4, 8), "float32"),
        "projected": TensorInfo("projected", (2, 4, 12), "float32"),
        "biased": TensorInfo("biased", (2, 4, 12), "float32"),
        "y": TensorInfo("y", (2, 4, 12), "float32"),
    }
    return graph


def test_matmul_bias_gelu_fuses_and_matches():
    graph = build_ffn_graph()
    x = np.random.default_rng(8).normal(size=(2, 4, 8)).astype(np.float32)
    expected = run_graph(graph, {"x": x})["y"]

    rewritten = run_transformer_rewrites(graph)

    assert len(rewritten.nodes) == 1
    assert rewritten.nodes[0].op_type == "Gemm"
    assert rewritten.nodes[0].attributes["activation"] == "Gelu"
    np.testing.assert_allclose(run_graph(rewritten, {"x": x})["y"], expected, rtol=1e-6, atol=1e-6)


def test_rewrite_is_blocked_for_shared_matmul_output():
    graph = build_ffn_graph()
    graph.nodes.append(Node("branch", "Relu", ["projected"], ["branch_out"]))
    graph.outputs.append("branch_out")
    rewritten = run_transformer_rewrites(graph)
    assert rewritten.op_counts()["MatMul"] == 1
