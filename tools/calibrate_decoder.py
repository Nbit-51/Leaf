"""Collect input-channel maxima on an independent calibration text split.

Import adapters map source modules to canonical linear operators. Neither the
statistics format nor native rescaling depends on a model or dataset name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.datasets import token_prefix
from tools.decoder_plan import make_plan, tensor_shapes, validate_plan
from tools.validate_decoder import digest


def collect(model, sequences: list[list[int]], plan: dict) -> dict[str, np.ndarray]:
    import torch
    validate_plan(plan)
    observed = {}
    handles = []

    def observe(name):
        def hook(module, inputs):
            value = inputs[0]
            if not torch.is_floating_point(value):
                return
            maximum = value.detach().float().abs().reshape(-1, value.shape[-1]).amax(dim=0).cpu().numpy()
            observed[name] = np.maximum(observed.get(name, 0), maximum)
        return hook

    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is not None and weight.ndim == 2 and not isinstance(module, torch.nn.Embedding):
            handles.append(module.register_forward_pre_hook(observe(name)))
    try:
        with torch.inference_mode():
            for index, sequence in enumerate(sequences):
                model(torch.tensor([sequence]), use_cache=False)
                print(f"Calibration block {index + 1}/{len(sequences)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    canonical = {}
    for name, (_, cols) in tensor_shapes(plan["config"]).items():
        if not name.endswith(".weight") or name.startswith("model.embed_tokens.") or name.startswith("model.position_embeddings."):
            continue
        specification = plan["tensors"][name]
        source_module = specification.get("input_module", specification["source"].removesuffix(".weight"))
        maximum = observed.get(source_module)
        if maximum is None and name == "lm_head.weight":
            maximum = observed.get("lm_head")  # tied embeddings are read by a different operator
        if maximum is None:
            if "layernorm" in name or name == "model.norm.weight":
                continue
            raise ValueError(f"No calibration inputs for {name}; provide a source module mapping")
        if maximum.shape != (cols,):
            raise ValueError(f"Calibration width mismatch for {name}")
        canonical[name] = maximum.astype(np.float32)
    return canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if min(args.blocks, args.sequence_length, args.threads) <= 0:
        parser.error("calibration dimensions must be positive")
    torch.set_num_threads(args.threads)
    config = json.loads((args.model / "config.json").read_text())
    plan = json.loads(args.plan.read_text()) if args.plan else make_plan(config)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    needed = args.blocks * args.sequence_length
    ids = token_prefix(tokenizer, args.dataset, needed, args.text_column)
    sequences = [ids[i:i + args.sequence_length] for i in range(0, needed, args.sequence_length)]
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                                dtype=torch.float32, attn_implementation="eager").eval()
    arrays = collect(model, sequences, plan)
    metadata = {"format": "leaf-activation-calibration-v1", "tokens": needed,
                "dataset_sha256": digest(args.dataset), "dataset": args.dataset.name,
                "model_config_sha256": digest(args.model / "config.json"),
                "source_weight_sha256": {p.name: digest(p) for p in sorted(args.model.glob("*.safetensors"))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays, __metadata__=json.dumps(metadata))


if __name__ == "__main__":
    main()
