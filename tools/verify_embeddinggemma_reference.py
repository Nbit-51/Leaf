"""Offline FP32 reference preparation, not a Leaf benchmark or quality qualification.

Requires sentence-transformers in the validation environment. The snapshot must
retain its original module folders. Explicit task prefixes are used because a
browser download may omit the optional Sentence Transformers prompt metadata.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


MODULES = [("", "Transformer"), ("1_Pooling", "Pooling"), ("2_Dense", "Dense"),
           ("3_Dense", "Dense"), ("4_Normalize", "Normalize")]


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def inspect_snapshot(root):
    """Require the exact supported reference pipeline; never guess missing heads."""
    root = Path(root)
    read = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    modules = read("modules.json")
    expected = [(i, path, "sentence_transformers.models." + kind)
                for i, (path, kind) in enumerate(MODULES)]
    if [(m.get("idx"), m.get("path"), m.get("type")) for m in modules] != expected:
        raise ValueError("Unsupported embedding module pipeline")
    config = read("config.json")
    if (config.get("model_type") != "gemma3_text" or
            config.get("use_bidirectional_attention") is not True):
        raise ValueError("Require the bidirectional Gemma 3 text backbone")
    pooling = read("1_Pooling/config.json")
    if (pooling.get("pooling_mode_mean_tokens") is not True or
            pooling.get("include_prompt") is not True or
            any(value for key, value in pooling.items()
                if key.startswith("pooling_mode_") and key != "pooling_mode_mean_tokens")):
        raise ValueError("Require masked mean pooling including prompt tokens")
    width = config["hidden_size"]
    if pooling.get("word_embedding_dimension") != width:
        raise ValueError("Pooling dimension differs from the backbone")
    for folder in ("2_Dense", "3_Dense"):
        dense = read(folder + "/config.json")
        if (dense.get("in_features") != width or dense.get("bias") is not False or
                dense.get("activation_function") != "torch.nn.modules.linear.Identity"):
            raise ValueError("Unsupported dense projection configuration")
        width = dense["out_features"]
    if width != 768:
        raise ValueError("Require the original 768-dimensional embedding output")
    files = ["config.json", "modules.json", "model.safetensors", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "1_Pooling/config.json",
             "2_Dense/config.json", "2_Dense/model.safetensors",
             "3_Dense/config.json", "3_Dense/model.safetensors"]
    files += [name for name in ("added_tokens.json", "config_sentence_transformers.json",
                               "sentence_bert_config.json") if (root / name).is_file()]
    return config, {name: digest(root / name) for name in files}


def mean_pool(hidden, mask):
    hidden, mask = np.asarray(hidden), np.asarray(mask)
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or not np.isin(mask, [0, 1]).all():
        raise ValueError("Invalid hidden-state shape or padding mask")
    counts = mask.sum(axis=1)
    if np.any(counts == 0) or not np.isfinite(hidden).all():
        raise ValueError("Empty or nonfinite embedding input")
    return (hidden * mask.astype(hidden.dtype)[..., None]).sum(axis=1) / counts.astype(hidden.dtype)[:, None]


def project_normalize(pooled, first, second):
    if (pooled.ndim != 2 or first.ndim != 2 or second.ndim != 2 or
            pooled.shape[1] != first.shape[1] or first.shape[0] != second.shape[1]):
        raise ValueError("Incompatible embedding projection shapes")
    embeddings = (pooled.astype(np.float32) @ first.T) @ second.T
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    if not np.isfinite(embeddings).all() or np.any(norms <= 0):
        raise ValueError("Zero or nonfinite embedding output")
    return embeddings / norms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh output directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory to retain previous results")
    config, hashes = inspect_snapshot(args.model)
    import torch
    import transformers
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    from safetensors import safe_open
    from safetensors.numpy import load_file

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = SentenceTransformer(str(args.model.resolve()), device="cpu",
        local_files_only=True, trust_remote_code=False,
        model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"})
    model.max_seq_length = min(config["max_position_embeddings"], 2048)
    model.default_prompt_name = None
    model.eval()
    state = model[0].auto_model.state_dict()
    with safe_open(args.model / "model.safetensors", framework="numpy") as weights:
        if set(weights.keys()) != set(state):
            raise ValueError("Backbone tensor inventory differs from the loaded model")
        for name, tensor in state.items():
            if list(tensor.shape) != weights.get_slice(name).get_shape() or tensor.dtype != torch.float32:
                raise ValueError("Backbone tensor geometry/dtype differs from the FP32 reference")
    first = load_file(args.model / "2_Dense/model.safetensors")["linear.weight"]
    second = load_file(args.model / "3_Dense/model.safetensors")["linear.weight"]
    for folder, weight in (("2_Dense", first), ("3_Dense", second)):
        meta = json.loads((args.model / folder / "config.json").read_text())
        if list(weight.shape) != [meta["out_features"], meta["in_features"]]:
            raise ValueError("Dense weights disagree with snapshot configuration")
    texts = [
        "task: search result | query: Which planet is known as the red planet?",
        "task: search result | query: How do I sort a C++ vector?",
        "title: none | text: Mars is known as the red planet because of iron oxide on its surface.",
        "title: none | text: Use std::sort with the vector's begin and end iterators in C++.",
        "title: none | text: Bread dough rises when yeast produces carbon dioxide.",
    ]
    # Cross the configured local-window boundary, without truncating the request.
    long_text = "title: none | text: " + "A local CPU processes text. " * 110
    args.output.mkdir(parents=True)
    record = dict(format="leaf-embeddinggemma-reference-v1",
        measured_at_utc=datetime.now(timezone.utc).isoformat(), model="google/embeddinggemma-300m",
        revision="unknown; browser downloads bound by file hashes", snapshot_sha256=hashes,
        script_sha256=digest(__file__), torch=torch.__version__, transformers=transformers.__version__,
        sentence_transformers=sentence_transformers.__version__, dtype="float32", threads=1,
        attention="eager", device="cpu", local_files_only=True,
        backbone_tensor_count=len(state), backbone_tensor_inventory_verified=True,
        scope="Reference pipeline smoke checks only; no native Leaf execution, latency gate or retrieval-quality qualification.",
        native_leaf_validated=False, quality_qualified=False, timing_qualified=False, complete=False)
    def save():
        (args.output / "reference.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    save()
    with torch.inference_mode():
        reference = model.encode(texts, batch_size=1, convert_to_numpy=True, show_progress_bar=False)
        batch = model.encode(texts, batch_size=len(texts), convert_to_numpy=True, show_progress_bar=False)
        np.testing.assert_allclose(batch, reference, atol=2e-5, rtol=2e-4)
        features = model.tokenize(texts)
        hidden = model[0](features)["token_embeddings"].cpu().numpy()
        manual = project_normalize(mean_pool(hidden, features["attention_mask"].cpu().numpy()), first, second)
        np.testing.assert_allclose(manual, batch, atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose(np.linalg.norm(reference, axis=1), 1, atol=1e-6)
        long_features = model.tokenize([long_text])
        length = int(long_features["attention_mask"].sum())
        if not config["sliding_window"] < length < model.max_seq_length:
            raise ValueError("Long fixture must cross the configured local window without truncation")
        long_reference = model.encode([long_text], convert_to_numpy=True, show_progress_bar=False)
        if not np.isfinite(long_reference).all():
            raise ValueError("Nonfinite long-context embeddings")
        np.testing.assert_allclose(np.linalg.norm(long_reference, axis=1), 1, atol=1e-6)
    # Preserve inputs/outputs for later native parity; these examples are not a benchmark dataset.
    np.savez(args.output / "reference.npz", embeddings=reference, input_ids=features["input_ids"].numpy(),
        attention_mask=features["attention_mask"].numpy(), long_embeddings=long_reference,
        long_input_ids=long_features["input_ids"].numpy(), long_attention_mask=long_features["attention_mask"].numpy())
    (args.output / "texts.json").write_text(json.dumps(dict(texts=texts, long_text=long_text), indent=2) + "\n")
    if any(digest(args.model / name) != value for name, value in hashes.items()):
        raise ValueError("Snapshot changed during validation")
    record.update(complete=True, passed=True, vectors_shape=list(reference.shape),
        short_token_lengths=features["attention_mask"].sum(dim=1).tolist(), long_token_length=length,
        padded_vs_individual_max_abs=float(np.max(np.abs(batch-reference))),
        independent_pool_projection_max_abs=float(np.max(np.abs(manual-batch))),
        toy_retrieval_scores=(reference[:2] @ reference[2:].T).tolist(),
        toy_retrieval_top_document=np.argmax(reference[:2] @ reference[2:].T, axis=1).tolist(),
        fixture_sha256={name: digest(args.output / name) for name in ("reference.npz", "texts.json")})
    save()
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
