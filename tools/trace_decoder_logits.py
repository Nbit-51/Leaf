"""Trace a retained FP32 decoder quality block against PyTorch eager; no timing claims."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leaf import cli
from leaf.evaluation import write_record
from tools.decoder_validation import run_native
from tools.validate_decoder import digest


ATOL = RTOL = 2e-4
STAGES = ("input_norm", "q", "k", "v", "q_rotary", "k_rotary", "attention",
          "attn_projected", "post_norm", "up", "activation", "down", "hidden")


def compare(actual, expected):
    """Use the release elementwise tolerance, including values close to zero."""
    actual, expected = np.asarray(actual).reshape(-1), np.asarray(expected).reshape(-1)
    if (actual.shape != expected.shape or not actual.size
            or not np.isfinite(actual).all() or not np.isfinite(expected).all()):
        raise ValueError("Invalid trace tensor")
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    tolerance = ATOL + RTOL * np.abs(expected.astype(np.float64))
    ratios = delta / tolerance
    worst = int(np.argmax(ratios))
    violations = int(np.sum(delta > tolerance))
    return dict(size=int(actual.size), max_abs_error=float(delta.max()),
                tolerance_violations=violations, allclose=violations == 0,
                max_tolerance_ratio=float(ratios[worst]), worst_flat_index=worst,
                reference_at_worst=float(expected[worst]), native_at_worst=float(actual[worst]))


def first_row(value, rotary=False):
    # Only rotary tensors use [batch, heads, tokens, head_dim]. All other
    # captures use [batch, tokens, features] (or NeoX Q/K/V's tokens-first form).
    return value[:, :, 0, :] if rotary else value.reshape(-1, value.shape[-1])[0]


def capture_reference(model, row, emit):
    """Adapt operator families, never Hub IDs or model-specific numerical paths."""
    family = model.config.model_type
    if family in ("llama", "qwen2"):
        core = model.model
        embedding, layers, final_norm = core.embed_tokens, core.layers, core.norm
    elif family == "gpt_neox":
        core = model.gpt_neox
        embedding, layers, final_norm = core.embed_in, core.layers, core.final_layer_norm
    else:
        raise ValueError(f"No reference trace adapter for decoder operator family: {family}")
    attention_module = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    hooks = []
    layer_prefix = {}

    def capture(name):
        def save(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            emit(name, value.detach().cpu().numpy().copy())
        return save

    def capture_input(name):
        def save(module, inputs):
            emit(name, inputs[0].detach().cpu().numpy().copy())
        return save

    hooks.append(embedding.register_forward_hook(capture("embedding")))
    for index, layer in enumerate(layers):
        prefix = f"model.layers.{index}."
        attention = layer.self_attn if family in ("llama", "qwen2") else layer.attention
        layer_prefix[id(attention)] = prefix
        for name, module in (("input_norm", layer.input_layernorm),
                             ("post_norm", layer.post_attention_layernorm), ("hidden", layer)):
            hooks.append(module.register_forward_hook(capture(prefix + name)))
        if family in ("llama", "qwen2"):
            for name in ("q", "k", "v"):
                hooks.append(getattr(attention, name + "_proj").register_forward_hook(capture(prefix + name)))
            projected = attention.o_proj
            up, down = layer.mlp.up_proj, layer.mlp.down_proj
            # Gated MLP activation is act(gate_proj(x)) * up_proj(x), which is
            # exactly the tensor passed into down_proj.
            hooks.append(down.register_forward_pre_hook(capture_input(prefix + "activation")))
        else:
            def save_qkv(module, inputs, output, prefix=prefix):
                values = output.detach().cpu().numpy().reshape(1, len(row), model.config.num_attention_heads, 3, -1)
                for offset, name in enumerate(("q", "k", "v")):
                    value = values[:, :, :, offset, :]
                    emit(prefix + name, value.reshape(1, len(row), -1).copy())
            hooks.append(attention.query_key_value.register_forward_hook(save_qkv))
            projected = attention.dense
            up, down = layer.mlp.dense_h_to_4h, layer.mlp.dense_4h_to_h
            hooks.append(layer.mlp.act.register_forward_hook(capture(prefix + "activation")))
        for name, module in (("attn_projected", projected), ("up", up), ("down", down)):
            hooks.append(module.register_forward_hook(capture(prefix + name)))
        hooks.append(projected.register_forward_pre_hook(capture_input(prefix + "attention")))
    hooks.append(final_norm.register_forward_hook(capture("final_norm")))
    original_attention = attention_module.eager_attention_forward

    def trace_attention(module, query, key, value, *args, **kwargs):
        prefix = layer_prefix[id(module)]
        emit(prefix + "q_rotary", query.detach().cpu().numpy().copy())
        emit(prefix + "k_rotary", key.detach().cpu().numpy().copy())
        return original_attention(module, query, key, value, *args, **kwargs)

    attention_module.eager_attention_forward = trace_attention
    try:
        with torch.inference_mode():
            emit("logits", model(torch.tensor([row]), use_cache=False).logits.cpu().numpy().copy())
    finally:
        attention_module.eager_attention_forward = original_attention
        for hook in hooks:
            hook.remove()
    return family, len(layers)


def capture_trace(args, record, row, artifact, snapshot):
    native_dir, expected_dir = args.output / "native", args.output / "expected"
    native_dir.mkdir()
    expected_dir.mkdir()
    prior = os.environ.get("LEAF_TRACE_DIRECTORY")
    os.environ["LEAF_TRACE_DIRECTORY"] = str(native_dir.resolve())
    try:
        native, _ = run_native(args.executable, artifact, [row])
    finally:
        if prior is None:
            os.environ.pop("LEAF_TRACE_DIRECTORY")
        else:
            os.environ["LEAF_TRACE_DIRECTORY"] = prior
    baseline, _ = run_native(args.baseline, artifact, [row])
    record["instrumentation_bit_exact"] = native.shape == baseline.shape and native.tobytes() == baseline.tobytes()
    np.save(args.output / "native_logits.npy", native)
    np.save(args.output / "baseline_logits.npy", baseline)
    if not record["instrumentation_bit_exact"]:
        raise ValueError("Instrumentation changed baseline outputs")
    del native, baseline
    torch.set_num_threads(1)
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                                                dtype=torch.float32, attn_implementation="eager").eval()

    def emit(name, expected):
        path = native_dir / (name + ".f32")
        actual = np.fromfile(path, dtype="<f4")
        if actual.size != expected.size:
            raise ValueError(f"Native/reference shape differs at {name}: {actual.size} vs {expected.shape}")
        actual = actual.reshape(expected.shape)
        np.save(expected_dir / (name + ".npy"), expected)
        rotary = name.endswith(("q_rotary", "k_rotary"))
        record["tensors"][name] = dict(shape=list(expected.shape),
            all_rows=compare(actual, expected), first_row=compare(first_row(actual, rotary), first_row(expected, rotary)))
        current = record["tensors"][name]
        print(f"{name}: all={current['all_rows']['max_tolerance_ratio']:.6g} "
              f"({current['all_rows']['tolerance_violations']}) "
              f"row0={current['first_row']['max_tolerance_ratio']:.6g} "
              f"({current['first_row']['tolerance_violations']})", flush=True)

    family, count = capture_reference(model, row, emit)
    record["operator_family"] = family
    ordered = ["embedding"] + [f"model.layers.{index}.{stage}" for index in range(count) for stage in STAGES] + ["final_norm", "logits"]
    if set(record["tensors"]) != set(ordered):
        raise ValueError("Native/reference trace stage coverage differs")
    if {path.stem for path in native_dir.glob("*.f32")} != set(ordered):
        raise ValueError("Unexpected or missing native trace stages")
    for scope in ("all_rows", "first_row"):
        failures = [name for name in ordered if not record["tensors"][name][scope]["allclose"]]
        record["first_material_divergence_" + scope] = failures[0] if failures else None
    # Check frozen inputs again after inference so a successful trace cannot
    # silently mix identities changed while the model was being loaded.
    for filename, checksum in record["input_sha256"].items():
        if digest(Path(filename)) != checksum:
            raise ValueError(f"Trace input changed during capture: {filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-record", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--block", type=int, required=True, help="zero-based saved block index")
    parser.add_argument("--executable", type=Path, required=True, help="LEAF_DIAGNOSTIC_TRACE build")
    parser.add_argument("--baseline", type=Path, required=True, help="frozen ordinary runtime")
    parser.add_argument("--output", type=Path, required=True, help="fresh output directory")
    args = parser.parse_args()
    source = json.loads(args.quality_record.read_text(encoding="utf-8"))
    if source.get("complete") is not True or source.get("precision") != "FP32":
        raise ValueError("Tracing requires a complete retained FP32 quality record")
    inputs = {args.quality_record.resolve(): digest(args.quality_record), args.tokens.resolve(): digest(args.tokens),
              args.executable.resolve(): digest(args.executable), args.baseline.resolve(): digest(args.baseline)}
    if inputs[args.tokens.resolve()] != source["tokens_sha256"]:
        raise ValueError("Token fixture identity mismatch")
    if inputs[args.baseline.resolve()] != source["executable_sha256"]:
        raise ValueError("Baseline executable identity mismatch")
    snapshot = Path(source["snapshot"])
    for name, checksum in source["source_sha256"].items():
        path = (snapshot / name).resolve()
        if digest(path) != checksum:
            raise ValueError("Model source identity mismatch")
        inputs[path] = checksum
    rows = json.loads(args.tokens.read_text(encoding="utf-8"))["sequences"]
    if not 0 <= args.block < len(rows):
        raise ValueError("Block index outside saved fixture")
    artifact = cli.prepare(snapshot, cli.model_directory(snapshot, cli.cache_root()), 32)
    if digest(artifact) != source["artifact_sha256"]:
        raise ValueError("Native artifact identity mismatch")
    inputs[artifact.resolve()] = source["artifact_sha256"]
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(format="leaf-decoder-trace-v1", complete=False, block=args.block,
                  sequence_length=len(rows[args.block]), atol=ATOL, rtol=RTOL,
                  quality_record_sha256=inputs[args.quality_record.resolve()],
                  trace_tool_sha256=digest(Path(__file__)),
                  executable_sha256=inputs[args.executable.resolve()], baseline_sha256=inputs[args.baseline.resolve()],
                  artifact=str(artifact.resolve()), artifact_sha256=source["artifact_sha256"],
                  input_sha256={str(path): checksum for path, checksum in inputs.items()},
                  torch=torch.__version__, transformers=transformers.__version__, tensors={})
    write_record(args.output / "trace.json", record)
    try:
        capture_trace(args, record, rows[args.block], artifact, snapshot)
        record["complete"] = True
    except BaseException as error:
        record["error"] = str(error) or type(error).__name__
        raise
    finally:
        write_record(args.output / "trace.json", record)


if __name__ == "__main__":
    main()
