"""Export a small decoder with the same operator family and audit Leaf coverage.

This is a bounded compatibility probe, not a full-model latency or parity
measurement. It avoids materializing multi-gigabyte FP32 ONNX weights while
identifying the next graph/runtime operators required by a cached model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.graph_opt.constant_folding import fold_constants
from tools.graph_opt.export_binary import export_graph
from tools.graph_opt.fusion import run_fusion_passes
from tools.graph_opt.ir import Graph
from tools.graph_opt.transformer import run_transformer_rewrites

NATIVE_OPS = {
    "Conv", "BatchNormalization", "Relu", "Identity", "Add", "Mul", "Sigmoid",
    "MaxPool", "GlobalAveragePool", "Flatten", "Gemm", "MatMul", "RMSNorm",
    "RoPE_Table", "RepeatKV", "Attention", "SwiGLU_MLP",
}


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids=input_ids, use_cache=False).logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="complete local decoder snapshot; weights are not loaded")
    parser.add_argument("--onnx-output", type=Path, default=Path("build/decoder_coverage_probe.onnx"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--leaf-infer", type=Path)
    args = parser.parse_args()
    config = deepcopy(AutoConfig.from_pretrained(args.config, local_files_only=True))
    model_type = config.model_type
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_hidden_layers = 1
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.vocab_size = 128
    config.max_position_embeddings = 128
    config._attn_implementation = "eager"
    torch.manual_seed(908)
    model = LogitsOnly(AutoModelForCausalLM.from_config(config).float()).eval()
    ids = torch.arange(4, dtype=torch.long).unsqueeze(0)
    args.onnx_output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(model, (ids,), str(args.onnx_output), opset_version=18,
                          input_names=["input_ids"], output_names=["logits"],
                          dynamo=True)
    graph = Graph.from_onnx(args.onnx_output)
    original_counts = Counter(node.op_type for node in graph.nodes)
    optimized = run_transformer_rewrites(run_fusion_passes(fold_constants(graph)))
    optimized_counts = Counter(node.op_type for node in optimized.nodes)
    unsupported = {name: count for name, count in optimized_counts.items()
                   if name not in NATIVE_OPS}
    leaf_attempt: dict[str, object] = {}
    leaf_path = args.onnx_output.with_suffix(".leaf")
    try:
        export_graph(optimized, str(leaf_path))
        leaf_attempt["exported"] = True
    except (TypeError, ValueError, KeyError) as error:
        leaf_attempt = {"exported": False, "error": f"{type(error).__name__}: {error}"}
    if leaf_attempt.get("exported") and args.leaf_infer:
        input_path = args.onnx_output.with_suffix(".input.bin")
        output_path = args.onnx_output.with_suffix(".output.bin")
        ids.numpy().astype(np.float32).tofile(input_path)
        result = subprocess.run([str(args.leaf_infer.resolve()), str(leaf_path),
                                 str(input_path), "1,4", str(output_path)],
                                capture_output=True, text=True)
        leaf_attempt["native_exit_code"] = result.returncode
        if result.returncode:
            leaf_attempt["native_error"] = result.stderr.strip()[:500]
    record = {
        "probe": "small random-weight decoder with cached model architecture",
        "model_type": model_type,
        "probe_layers": 1,
        "probe_hidden_size": 64,
        "source_model_weights_loaded": False,
        "original_ops": dict(sorted(original_counts.items())),
        "optimized_ops": dict(sorted(optimized_counts.items())),
        "unsupported_native_ops": dict(sorted(unsupported.items())),
        "native_graph_ready": not unsupported,
        "leaf_attempt": leaf_attempt,
    }
    print(json.dumps(record, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
