"""Execute a real ONNX graph through the public command and native runtime."""

import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from leaf import cli, graph


def test_infer_onnx_runs_native(monkeypatch, tmp_path):
    onnx = pytest.importorskip("onnx")
    executable = os.environ.get("LEAF_TEST_GRAPH_BIN")
    if not executable:
        pytest.skip("Set LEAF_TEST_GRAPH_BIN to the built leaf_infer target")

    runtime = Path(executable).resolve(strict=True)
    (tmp_path / "bin").mkdir()
    shutil.copy2(runtime, tmp_path / "bin" / ("leaf_infer.exe" if os.name == "nt" else "leaf_infer"))
    monkeypatch.setattr(graph, "__file__", str(tmp_path / "graph.py"))
    tensor = onnx.helper.make_tensor_value_info
    model = onnx.helper.make_model(
        onnx.helper.make_graph([onnx.helper.make_node("Relu", ["x"], ["y"])], "relu",
                               [tensor("x", onnx.TensorProto.FLOAT, [1, 3])],
                               [tensor("y", onnx.TensorProto.FLOAT, [1, 3])]),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    model_path, input_path, output_path = (tmp_path / name for name in ("model.onnx", "input.npy", "output.npy"))
    onnx.save(model, model_path)
    arguments = ["leaf", "infer", str(model_path), "--input", str(input_path), "--output", str(output_path)]
    monkeypatch.setattr(sys, "argv", arguments)
    np.save(input_path, np.array([-2, 0, 3], dtype=np.float32))
    assert cli.main() == 1
    assert not output_path.exists()

    np.save(input_path, np.array([[-2, 0, 3]], dtype=np.float32))
    assert cli.main() == 0
    np.testing.assert_array_equal(np.load(output_path), [[0, 0, 3]])
    np.save(input_path, np.ones((1, 3), dtype=np.float32))
    assert cli.main() == 1
    np.testing.assert_array_equal(np.load(output_path), [[0, 0, 3]])
