"""Reduced bidirectional Gemma checks, window boundaries and native input rejection."""

import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Gemma3TextConfig, Gemma3TextModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.embedding_validation import run_native, parity
from tools.export_embedding import export_embedding
from tools.verify_embeddinggemma_reference import MODULES, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(1081)
    checks = []
    with tempfile.TemporaryDirectory(prefix="leaf_embed_arch_") as tmp:
        root = Path(tmp)
        snapshot = root / "snapshot"
        snapshot.mkdir()
        # Independent projection width: 4*16 != hidden 32. Exercise both local
        # and full layers, GQA, nonzero norm weights and lengths around radius 8.
        config = Gemma3TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            query_pre_attn_scalar=16,
            sliding_window=16,
            max_position_embeddings=96,
            use_bidirectional_attention=True,
            layer_types=["sliding_attention", "full_attention"],
        )
        config._attn_implementation = "eager"
        model = Gemma3TextModel(config).float().eval()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "norm" in name:
                    param.uniform_(-0.5, 0.5)
        model.save_pretrained(snapshot, safe_serialization=True)
        serialized = config.to_dict()
        serialized["sliding_window"] = 16
        (snapshot / "config.json").write_text(json.dumps(serialized))
        (snapshot / "modules.json").write_text(
            json.dumps(
                [
                    dict(
                        idx=i,
                        name=str(i),
                        path=p,
                        type="sentence_transformers.models." + k,
                    )
                    for i, (p, k) in enumerate(MODULES)
                ]
            )
        )
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ):
            (snapshot / name).write_text("{}")
        for folder in ("1_Pooling", "2_Dense", "3_Dense"):
            (snapshot / folder).mkdir()
        (snapshot / "1_Pooling/config.json").write_text(
            json.dumps(
                dict(
                    word_embedding_dimension=32,
                    pooling_mode_mean_tokens=True,
                    include_prompt=True,
                )
            )
        )
        first = torch.randn(48, 32) * 0.1
        second = torch.randn(768, 48) * 0.1
        for folder, weights in [("2_Dense", first), ("3_Dense", second)]:
            (snapshot / folder / "config.json").write_text(
                json.dumps(
                    dict(
                        in_features=weights.shape[1],
                        out_features=weights.shape[0],
                        bias=False,
                        activation_function="torch.nn.modules.linear.Identity",
                    )
                )
            )
            save_file(
                {"linear.weight": weights}, snapshot / folder / "model.safetensors"
            )
        artifact = root / "encoder.leaf"
        export_embedding(snapshot, artifact)
        sequences = [
            [(7 * i + 3) % 127 + 1 for i in range(n)]
            for n in (1, 7, 8, 9, 16, 17, 33, 64)
        ]
        sequences += [sequences[0], list(reversed(sequences[-1]))]
        expected = []
        with torch.inference_mode():
            for ids in sequences:
                hidden = model(torch.tensor([ids]), use_cache=False).last_hidden_state
                vector = (hidden.mean(dim=1) @ first.T) @ second.T
                expected.append(torch.nn.functional.normalize(vector, dim=1).numpy()[0])
        expected = np.array(expected)
        for threads, scalar in ((1, False), (2, False), (1, True)):
            actual, metrics = run_native(
                args.executable, artifact, sequences, threads=threads, scalar=scalar
            )
            checks.append(
                dict(
                    threads=threads,
                    scalar=scalar,
                    parity=parity(actual, expected),
                    execution=metrics,
                )
            )
        rejected = []
        for name, seqs in [
            ("empty", [[]]),
            ("overlength", [[1] * 97]),
            ("bad_token", [[128]]),
        ]:
            try:
                run_native(args.executable, artifact, seqs)
            except subprocess.CalledProcessError:
                rejected.append(name)
            else:
                raise AssertionError("Malformed request accepted: " + name)
        original = artifact.read_bytes()
        for name, mutation in [
            ("magic", b"WRONGMAG" + original[8:]),
            ("truncated", original[:100]),
            ("bad_version", original[:8] + struct.pack("<I", 2) + original[12:]),
        ]:
            invalid = root / (name + ".leaf")
            invalid.write_bytes(mutation)
            try:
                run_native(args.executable, invalid, [[1]])
            except subprocess.CalledProcessError:
                rejected.append(name)
            else:
                raise AssertionError("Malformed artifact accepted: " + name)
    record = dict(
        format="leaf-embedding-architecture-v1",
        native_executable_sha256=digest(args.executable),
        validation_script_sha256=digest(__file__),
        torch=torch.__version__,
        lengths=[len(x) for x in sequences],
        checks=checks,
        rejected=rejected,
        passed=True,
        trained_quality=False,
        timing_qualified=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
