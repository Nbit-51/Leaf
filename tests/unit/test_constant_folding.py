import numpy as np

from tools.graph_opt.constant_folding import fold_constants
from tools.graph_opt.executor import run_graph
from tools.graph_opt.ir import Graph, Node


def test_constant_subgraph_is_evaluated_and_pruned():
    graph = Graph()
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.initializers = {
        "a": np.asarray([1.0, 2.0], dtype=np.float32),
        "b": np.asarray([3.0, 4.0], dtype=np.float32),
    }
    graph.nodes = [
        Node("add_constants", "Add", ["a", "b"], ["sum"]),
        Node("scale_input", "Mul", ["x", "sum"], ["y"]),
    ]
    sample = {"x": np.asarray([2.0, -1.0], dtype=np.float32)}
    expected = run_graph(graph, sample)["y"]

    folded = fold_constants(graph)

    assert [node.name for node in folded.nodes] == ["scale_input"]
    np.testing.assert_array_equal(folded.initializers["sum"], [4.0, 6.0])
    np.testing.assert_allclose(run_graph(folded, sample)["y"], expected)
    assert folded.metadata["passes"]["constant_folding"]["count"] == 1


def test_large_constant_is_not_materialized():
    graph = Graph()
    graph.outputs = ["product"]
    graph.initializers = {
        "a": np.ones((8, 8), dtype=np.float32),
        "b": np.ones((8, 8), dtype=np.float32),
    }
    graph.nodes = [Node("matmul", "MatMul", ["a", "b"], ["product"])]
    folded = fold_constants(graph, max_constant_bytes=16)
    assert len(folded.nodes) == 1
