"""Verify standalone native RoPE tables and grouped-KV repetition against PyTorch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.graph_opt.executor import run_graph
from tools.graph_opt.export_binary import export_graph
from tools.graph_opt.ir import Graph, Node, TensorInfo
from tools.graph_opt.memory_planner import plan_memory


def _native(executable: Path, artifact: Path, input_path: Path,
            shape: tuple[int, ...], output_path: Path,
            output_shape: tuple[int, ...]) -> np.ndarray:
    result = subprocess.run([str(executable), str(artifact), str(input_path),
                             ",".join(map(str, shape)), str(output_path)],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return np.fromfile(output_path, dtype=np.float32).reshape(output_shape)


def _bench(executable: Path, artifact: Path, input_path: Path,
           shape: tuple[int, ...]) -> float:
    result = subprocess.run([str(executable), str(artifact), str(input_path),
                             ",".join(map(str, shape)), "10", "100"],
                            capture_output=True, text=True, check=True)
    match = re.search(r"p50=([0-9.]+)", result.stdout)
    if match is None:
        raise RuntimeError("native benchmark did not report p50")
    return float(match.group(1))


def _verify(graph: Graph, input_name: str, data: np.ndarray,
            expected: np.ndarray, args: argparse.Namespace,
            directory: Path, label: str) -> dict:
    reference = run_graph(graph, {input_name: data})[graph.outputs[0]]
    np.testing.assert_allclose(reference, expected, rtol=1e-5, atol=1e-5)
    input_path = directory / f"{label}_input.bin"
    data.tofile(input_path)
    measured = {}
    for version, plan in (("v2", None), ("v3", plan_memory(graph))):
        artifact = directory / f"{label}_{version}.leaf"
        export_graph(graph, str(artifact), memory_plan=plan)
        actual = _native(args.leaf_infer.resolve(), artifact, input_path,
                         data.shape, directory / f"{label}_{version}.bin", expected.shape)
        error = float(np.max(np.abs(actual - expected)))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        measured[version] = {"max_abs_vs_pytorch": error,
                             "p50_ms": _bench(args.leaf_bench.resolve(), artifact,
                                              input_path, data.shape)}
    return measured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf-infer", type=Path, required=True)
    parser.add_argument("--leaf-bench", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rng = np.random.default_rng(502)
    positions = np.arange(64, dtype=np.float32).reshape(1, 1, 64)
    frequencies = rng.uniform(0.001, 0.1, (1, 32, 1)).astype(np.float32)
    with torch.inference_mode():
        angles = (torch.from_numpy(frequencies) @
                  torch.from_numpy(positions)).transpose(1, 2)
        duplicated = torch.cat((angles, angles), dim=-1)
        expected_rope = {"cos": (duplicated.cos() * 0.75).numpy(),
                         "sin": (duplicated.sin() * 1.25).numpy()}
    rope = Graph()
    rope.inputs = ["positions"]
    rope.initializers = {"frequencies": frequencies}
    rope.nodes = [Node("rope", "RoPE_Table", ["frequencies", "positions"],
                       ["cos", "sin"], {"cos_scale": 0.75, "sin_scale": 1.25})]
    rope.value_info = {"positions": TensorInfo("positions", positions.shape, "float32"),
                       "cos": TensorInfo("cos", (1, 64, 64), "float32"),
                       "sin": TensorInfo("sin", (1, 64, 64), "float32")}

    kv = rng.normal(0, 0.3, (1, 2, 64, 64)).astype(np.float32)
    with torch.inference_mode():
        expected_repeat = torch.from_numpy(kv).repeat_interleave(7, dim=1).numpy()
    repeat = Graph()
    repeat.inputs, repeat.outputs = ["kv"], ["repeated"]
    repeat.nodes = [Node("repeat", "RepeatKV", ["kv"], ["repeated"], {"n_rep": 7})]
    repeat.value_info = {"kv": TensorInfo("kv", kv.shape, "float32"),
                         "repeated": TensorInfo("repeated", expected_repeat.shape, "float32")}

    with tempfile.TemporaryDirectory(prefix="leaf_rope_repeat_") as temporary:
        directory = Path(temporary)
        cases = {}
        for output_name in ("cos", "sin"):
            rope.outputs = [output_name]
            cases[f"rope_{output_name}"] = _verify(rope, "positions", positions,
                                                    expected_rope[output_name], args,
                                                    directory, f"rope_{output_name}")
        cases["repeat_kv"] = _verify(repeat, "kv", kv, expected_repeat, args,
                                     directory, "repeat_kv")
        named_rope = rope.clone()
        named_rope.inputs = ["frequencies", "positions"]
        named_rope.outputs = ["cos", "sin"]
        named_rope.initializers.clear()
        named_rope.value_info["frequencies"] = TensorInfo(
            "frequencies", frequencies.shape, "float32")
        frequency_path = directory / "frequencies.bin"
        position_path = directory / "positions.bin"
        frequencies.tofile(frequency_path)
        positions.tofile(position_path)
        named_results = {}
        for version, plan in (("v2", None), ("v3", plan_memory(named_rope))):
            artifact = directory / f"rope_named_{version}.leaf"
            prefix = directory / f"rope_named_{version}"
            export_graph(named_rope, str(artifact), memory_plan=plan)
            result = subprocess.run(
                [str(args.leaf_infer.resolve()), str(artifact), str(prefix),
                 "frequencies", "1,32,1", str(frequency_path),
                 "positions", "1,1,64", str(position_path)],
                capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr.strip())
            errors = {}
            for index, name in enumerate(("cos", "sin")):
                actual = np.fromfile(f"{prefix}.{index}.bin", dtype=np.float32).reshape(1, 64, 64)
                np.testing.assert_allclose(actual, expected_rope[name], rtol=1e-5, atol=1e-5)
                errors[name] = float(np.max(np.abs(actual - expected_rope[name])))
            named_results[version] = errors
        cases["rope_named_two_outputs"] = named_results
    result = {"benchmark": "leaf-native-rope-repeatkv",
              "measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "host": platform.platform(), "warmup": 10, "runs": 100,
              "shapes": {"rope_table": [1, 64, 64], "repeat_kv_input": [1, 2, 64, 64],
                         "repeat_kv_output": [1, 14, 64, 64]}, "cases": cases}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
