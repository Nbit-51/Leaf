"""Check two-step native GQA KV-cache execution against PyTorch."""

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

from tools.graph_opt.export_binary import export_graph
from tools.graph_opt.ir import Graph, Node, TensorInfo
from tools.graph_opt.memory_planner import plan_memory


def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               mask: torch.Tensor, scale: float, repeats: int) -> torch.Tensor:
    repeated_key = key.repeat_interleave(repeats, dim=1)
    repeated_value = value.repeat_interleave(repeats, dim=1)
    scores = (query * scale) @ (repeated_key * scale).transpose(-1, -2)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    probabilities = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    return (probabilities @ repeated_value).permute(0, 2, 1, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-exe", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rng = np.random.default_rng(619)
    query_heads, kv_heads, dim, prefix_tokens = 14, 2, 64, 63
    scale = dim ** -0.25
    prefix_query = rng.normal(0, 0.3, (1, query_heads, prefix_tokens, dim)).astype(np.float32)
    prefix_key = rng.normal(0, 0.3, (1, kv_heads, prefix_tokens, dim)).astype(np.float32)
    prefix_value = rng.normal(0, 0.3, (1, kv_heads, prefix_tokens, dim)).astype(np.float32)
    prefix_mask = np.tril(np.ones((prefix_tokens, prefix_tokens), dtype=np.float32))[
        None, None]
    step_query = rng.normal(0, 0.3, (1, query_heads, 1, dim)).astype(np.float32)
    step_key = rng.normal(0, 0.3, (1, kv_heads, 1, dim)).astype(np.float32)
    step_value = rng.normal(0, 0.3, (1, kv_heads, 1, dim)).astype(np.float32)
    step_mask = np.ones((1, 1, 1, 1), dtype=np.float32)
    with torch.inference_mode():
        expected_prefix = _attention(*map(torch.from_numpy,
                                          (prefix_query, prefix_key, prefix_value, prefix_mask)),
                                     scale, query_heads // kv_heads).numpy()
        full_key = torch.cat((torch.from_numpy(prefix_key), torch.from_numpy(step_key)), dim=2)
        full_value = torch.cat((torch.from_numpy(prefix_value), torch.from_numpy(step_value)), dim=2)
        expected_step = _attention(torch.from_numpy(step_query), full_key, full_value,
                                   torch.from_numpy(step_mask), scale,
                                   query_heads // kv_heads).numpy()

    graph = Graph()
    graph.inputs, graph.outputs = ["q", "k", "v", "mask"], ["y"]
    graph.nodes = [Node("cached_attention", "Attention", ["q", "k", "v", "mask"],
                        ["y"], {"scale": scale, "cache_id": "layer_0",
                                "mask_nonzero_is_valid": 1})]
    graph.value_info = {
        "q": TensorInfo("q", prefix_query.shape, "float32"),
        "k": TensorInfo("k", prefix_key.shape, "float32"),
        "v": TensorInfo("v", prefix_value.shape, "float32"),
        "mask": TensorInfo("mask", prefix_mask.shape, "float32"),
        "y": TensorInfo("y", expected_prefix.shape, "float32"),
    }
    arrays = {"prefix.q": prefix_query, "prefix.k": prefix_key,
              "prefix.v": prefix_value, "prefix.mask": prefix_mask,
              "step.q": step_query, "step.k": step_key,
              "step.v": step_value, "step.mask": step_mask}
    cases = {}
    with tempfile.TemporaryDirectory(prefix="leaf_kv_cache_") as temporary:
        directory = Path(temporary)
        for name, array in arrays.items():
            array.tofile(directory / f"{name}.bin")
        for version, plan in (("v2", None), ("v3", plan_memory(graph))):
            artifact = directory / f"cached_{version}.leaf"
            output_prefix = directory / f"cached_{version}"
            export_graph(graph, str(artifact), memory_plan=plan)
            command = [str(args.session_exe.resolve()), str(artifact)]
            command += [str(directory / f"{name}.bin") for name in arrays]
            command += [str(query_heads), str(kv_heads), str(dim),
                        str(prefix_tokens), "50", str(output_prefix)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr.strip())
            match = re.search(r"prefill_p50_ms=([0-9.]+) decode_p50_ms=([0-9.]+)",
                              result.stdout)
            if match is None:
                raise RuntimeError("cached attention harness did not report latency")
            actual_prefix = np.fromfile(f"{output_prefix}.prefill.bin",
                                        dtype=np.float32).reshape(expected_prefix.shape)
            actual_step = np.fromfile(f"{output_prefix}.decode.bin",
                                      dtype=np.float32).reshape(expected_step.shape)
            np.testing.assert_allclose(actual_prefix, expected_prefix, rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(actual_step, expected_step, rtol=1e-4, atol=1e-4)
            cases[version] = {
                "prefill_max_abs_vs_pytorch": float(np.max(np.abs(actual_prefix - expected_prefix))),
                "decode_max_abs_vs_pytorch": float(np.max(np.abs(actual_step - expected_step))),
                "prefill_p50_ms": float(match.group(1)),
                "decode_p50_ms": float(match.group(2)),
            }
    record = {"benchmark": "leaf-native-dynamic-kv-cache",
              "measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "host": platform.platform(), "prefix_tokens": prefix_tokens,
              "decode_tokens": 1, "query_heads": query_heads, "kv_heads": kv_heads,
              "head_dim": dim, "timed_runs": 50, "cases": cases}
    print(json.dumps(record, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
