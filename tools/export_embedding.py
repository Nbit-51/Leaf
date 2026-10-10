"""Stream an FP32 bidirectional Gemma text encoder into a native embedding artifact."""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.export_decoder import tensor_index, read_tensor
from tools.verify_embeddinggemma_reference import inspect_snapshot, digest

HEADER = struct.Struct("<8sIIQ11I5f")
DESCRIPTOR = struct.Struct("<IIQQ")


def embedding_config(config):
    if (
        config.get("model_type") != "gemma3_text"
        or not config.get("use_bidirectional_attention")
        or config.get("hidden_activation") != "gelu_pytorch_tanh"
        or config.get("attention_bias")
        or config.get("attn_logit_softcapping")
        or config.get("final_logit_softcapping")
        or config.get("rope_scaling")
    ):
        raise ValueError("Unsupported native embedding block semantics")
    h, f, heads, kv, layers, vocab, maximum, dim = (
        config[k]
        for k in (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "num_hidden_layers",
            "vocab_size",
            "max_position_embeddings",
            "head_dim",
        )
    )
    if (
        any(
            type(x) is not int or not 0 < x <= 0xFFFFFFFF
            for x in (h, f, heads, kv, layers, vocab, maximum, dim)
        )
        or heads % kv
        or dim % 2
    ):
        raise ValueError("Invalid embedding geometry")
    types = config.get("layer_types")
    if (
        not isinstance(types, list)
        or len(types) != layers
        or any(x not in ("full_attention", "sliding_attention") for x in types)
    ):
        raise ValueError("Require explicit per-layer attention kinds")
    window = config.get("sliding_window")
    if type(window) is not int or window < 2 or window // 2 >= maximum:
        raise ValueError("Invalid bidirectional attention window")
    rope = config.get("rope_parameters") or {}
    if rope and set(rope) != {"full_attention", "sliding_attention"}:
        raise ValueError("Unsupported embedding RoPE layout")
    for values in rope.values():
        if (
            set(values) - {"rope_type", "rope_theta"}
            or values.get("rope_type", "default") != "default"
        ):
            raise ValueError("Scaled/extended embedding RoPE is unsupported")
    global_theta = rope.get("full_attention", {}).get(
        "rope_theta", config.get("rope_theta", 1000000.0)
    )
    local_theta = rope.get("sliding_attention", {}).get(
        "rope_theta", config.get("rope_local_base_freq", 10000.0)
    )
    eps = config["rms_norm_eps"]
    scale = config["query_pre_attn_scalar"]
    if any(
        not math.isfinite(x) or x <= 0 for x in (global_theta, local_theta, eps, scale)
    ):
        raise ValueError("Invalid embedding numerical constants")
    # Serialized HF config retains the original window. Gemma's bidirectional
    # mask uses abs(query-key) < (window//2 + 1), including both endpoints here.
    return (
        (h, f, heads, kv, layers, vocab, maximum, dim, window // 2),
        (global_theta, local_theta, eps, math.sqrt(h), scale**-0.5),
        types,
    )


def export_embedding(snapshot, output):
    snapshot, output = Path(snapshot), Path(output)
    config, hashes = inspect_snapshot(snapshot)
    dims, constants, types = embedding_config(config)
    h, f, heads, kv, layers, vocab, maximum, dim, _ = dims
    dense = json.loads((snapshot / "2_Dense/config.json").read_text())["out_features"]
    out = json.loads((snapshot / "3_Dense/config.json").read_text())["out_features"]
    shapes = {
        "embed_tokens.weight": (vocab, h),
        "norm.weight": (1, h),
        "pool.up.weight": (dense, h),
        "pool.down.weight": (out, dense),
    }
    for i in range(layers):
        p = f"layers.{i}."
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            shapes[p + name + ".weight"] = (1, h)
        for name in ("q", "k"):
            shapes[p + f"self_attn.{name}_norm.weight"] = (1, dim)
        for name, width in (("q", heads * dim), ("k", kv * dim), ("v", kv * dim)):
            shapes[p + f"self_attn.{name}_proj.weight"] = (width, h)
        shapes[p + "self_attn.o_proj.weight"] = (h, heads * dim)
        shapes[p + "mlp.gate_proj.weight"] = (f, h)
        shapes[p + "mlp.up_proj.weight"] = (f, h)
        shapes[p + "mlp.down_proj.weight"] = (h, f)
    sources = {}
    for file, alias in [
        ("model.safetensors", None),
        ("2_Dense/model.safetensors", "pool.up.weight"),
        ("3_Dense/model.safetensors", "pool.down.weight"),
    ]:
        filename = snapshot / file
        base, entries = tensor_index(filename)
        if alias and set(entries) != {"linear.weight"}:
            raise ValueError("Unexpected projection tensors")
        for name, entry in entries.items():
            target = alias or name
            if target in sources:
                raise ValueError("Duplicate embedding tensor")
            sources[target] = (filename, base, entry)
    if set(sources) != set(shapes):
        raise ValueError("Embedding tensor inventory differs from the supported plan")
    align = lambda x: (x + 63) // 64 * 64
    metadata = (
        HEADER.size
        + 4 * layers
        + sum(4 + len(name.encode()) + DESCRIPTOR.size for name in shapes)
    )
    payload_base = align(metadata)
    offset = 0
    table = []
    for name, shape in shapes.items():
        filename, base, entry = sources[name]
        actual = tuple(entry["shape"])
        expected = shape if shape[0] != 1 else (shape[1],)
        if actual != expected or entry["dtype"] != "F32":
            raise ValueError(f"Wrong FP32 tensor shape/dtype: {name}")
        # Validate source byte ranges before opening the destination.
        read_tensor(filename, base, entry)
        size = math.prod(shape) * 4
        table.append((name, shape, offset, size))
        offset = align(offset + size)
    if output.exists():
        raise FileExistsError("Use a fresh artifact output")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError("An unfinished export already exists")
    with partial.open("xb") as stream:
        stream.write(
            HEADER.pack(
                b"LEAFEM01", 1, len(table), payload_base, *dims, dense, out, *constants
            )
        )
        stream.write(
            struct.pack(
                "<" + "I" * layers, *[int(x == "sliding_attention") for x in types]
            )
        )
        for name, shape, offset, size in table:
            name_bytes = name.encode()
            stream.write(struct.pack("<I", len(name_bytes)) + name_bytes)
            stream.write(DESCRIPTOR.pack(*shape, offset, size))
        for name, shape, offset, size in table:
            stream.seek(payload_base + offset)
            filename, base, entry = sources[name]
            array = read_tensor(filename, base, entry)
            for start in range(0, array.shape[0], 256):
                block = np.asarray(array[start : start + 256], dtype="<f4")
                if not np.isfinite(block).all():
                    raise ValueError(f"Nonfinite tensor: {name}")
                stream.write(block.tobytes())
    if any(digest(snapshot / name) != value for name, value in hashes.items()):
        raise ValueError("Snapshot changed during export")
    partial.replace(output)
    record = dict(
        format="leaf-embedding-v1",
        weight_bits=32,
        snapshot_sha256=hashes,
        artifact_sha256=digest(output),
        artifact_bytes=output.stat().st_size,
        tensors=len(table),
        semantics="bidirectional Gemma 3; zero-centered RMSNorm; masked mean, two identity projections, L2",
        config=config,
        local_radius=dims[-1],
        exporter_sha256=digest(__file__),
    )
    output.with_suffix(".provenance.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_embedding(args.model, args.output), indent=2))
