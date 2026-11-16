"""Run a supported ONNX graph with the bundled native FP32 executor."""

import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def infer(model: Path, input_path: Path, output_path: Path):
    try:
        import onnx
    except ModuleNotFoundError as error:
        if error.name != "onnx":
            raise
        raise RuntimeError("Install ONNX with your Leaf wheel's [graph] extra or run python -m pip install onnx") from error

    from tools.graph_opt.constant_folding import fold_constants
    from tools.graph_opt.export_binary import export_graph
    from tools.graph_opt.fusion import run_fusion_passes
    from tools.graph_opt.ir import Graph
    from google.protobuf.message import DecodeError

    runtime = Path(__file__).resolve().parent / "bin" / ("leaf_infer.exe" if os.name == "nt" else "leaf_infer")
    if not runtime.is_file():
        raise RuntimeError("Native graph executor is missing. Install a Leaf platform wheel.")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    if input_path.suffix.lower() != ".npy" or output_path.suffix.lower() != ".npy":
        raise ValueError("Input and output must be .npy files")
    try:
        source = onnx.load(model)
        if any(node.domain not in ("", "ai.onnx") for node in source.graph.node):
            raise ValueError("ONNX graph contains a non-standard operator domain")
        graph = Graph.from_onnx(source)
    except (onnx.checker.ValidationError, DecodeError) as error:
        raise ValueError(f"Invalid ONNX model: {error}") from error
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise ValueError("This command requires a graph with one input and one output")
    name = graph.inputs[0]
    input_info, output_info = graph.value_info[name], graph.value_info[graph.outputs[0]]
    if input_info.dtype != "float32" or output_info.dtype != "float32":
        raise ValueError("This command requires FP32 graph inputs and outputs")
    try:
        values = np.load(input_path, allow_pickle=False)
    except EOFError as error:
        raise ValueError("Input must be a complete .npy array") from error
    if isinstance(values, np.lib.npyio.NpzFile):
        values.close()
        raise ValueError("Input must be a single .npy array")
    if values.dtype != np.float32 or not values.ndim or not values.size:
        raise ValueError("Input must be a non-empty FP32 array with at least one dimension")
    expected = input_info.shape
    if expected is not None and (len(expected) != values.ndim or any(
        size is not None and size != actual for size, actual in zip(expected, values.shape)
    )):
        raise ValueError(f"Input shape {values.shape} does not match ONNX shape {expected}")

    with tempfile.TemporaryDirectory(prefix="leaf-infer-") as directory:
        work = Path(directory)
        artifact, data, prefix = work / "graph.leaf", work / "input.bin", work / "output"
        export_graph(run_fusion_passes(fold_constants(graph)), str(artifact))
        values.astype("<f4", copy=False).tofile(data)
        shape = ",".join(map(str, values.shape))
        result = subprocess.run([str(runtime), str(artifact), str(prefix), name, shape, str(data)],
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "Native graph inference failed")
        _, separator, dimensions = result.stdout.rstrip("\r\n").rpartition("\t")
        if not separator:
            raise RuntimeError("Native graph executor omitted the output shape")
        output_shape = tuple(map(int, dimensions.split(","))) if dimensions else ()
        output = np.fromfile(prefix.with_suffix(".0.bin"), dtype="<f4").reshape(output_shape)
        with output_path.open("xb") as stream:
            np.save(stream, output, allow_pickle=False)
    print(f"Leaf: output shape {output.shape}, saved to {output_path}")
