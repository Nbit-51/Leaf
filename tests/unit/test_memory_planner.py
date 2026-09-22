import json

from tools.graph_opt.ir import Graph, Node, TensorInfo
from tools.graph_opt.memory_planner import plan_memory


def _chain_graph():
    graph = Graph()
    graph.inputs = ["x"]
    graph.outputs = ["c"]
    graph.nodes = [
        Node("first", "Relu", ["x"], ["a"]),
        Node("second", "Relu", ["a"], ["b"]),
        Node("third", "Relu", ["b"], ["c"]),
    ]
    graph.value_info = {
        name: TensorInfo(name, (1, 16), "float32") for name in ("x", "a", "b", "c")
    }
    return graph


def test_liveness_reuses_non_overlapping_buffers():
    plan = plan_memory(_chain_graph())
    allocations = {item.tensor: item for item in plan.allocations}
    assert allocations["a"].offset == allocations["c"].offset
    assert allocations["a"].offset != allocations["b"].offset
    assert plan.arena_size == 128
    assert plan.naive_size == 192


def test_memory_plan_exports_versioned_json(tmp_path):
    destination = tmp_path / "plan.json"
    plan_memory(_chain_graph()).export_json(destination)
    payload = json.loads(destination.read_text())
    assert payload["format"] == "leaf-memory-plan-v1"
    assert payload["bytes_saved"] == 64
    assert len(payload["graph_fingerprint"]) == 16


def test_dynamic_shape_is_reported_not_guessed():
    graph = _chain_graph()
    graph.value_info["b"] = TensorInfo("b", (None, 16), "float32")
    plan = plan_memory(graph)
    assert plan.unplanned_tensors == ["b"]
