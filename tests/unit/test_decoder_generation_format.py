"""Versioned termination semantics, including real native generation when enabled.

Set LEAF_TEST_DECODER_BIN to a newly built runtime to run the native cases.
LEAF_TEST_LEGACY_DECODER_BIN optionally checks rejection by an old v2/v3 reader.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from tools.decoder_plan import load_snapshot_plan, tensor_shapes
from tools.decoder_validation import run_native
from tools.export_decoder import HEADER, export_decoder


def fixture_model(root: Path, eos, predicted=3, window=False):
    root.mkdir()
    source = dict(model_type="mistral" if window else "llama", hidden_size=16,
                  intermediate_size=32, num_attention_heads=2, num_hidden_layers=1,
                  vocab_size=8, max_position_embeddings=32, eos_token_id=2,
                  tie_word_embeddings=False)
    if window:
        source["sliding_window"] = 2
    (root / "config.json").write_text(json.dumps(source))
    (root / "generation_config.json").write_text(json.dumps({"eos_token_id": eos}))
    plan = load_snapshot_plan(root)
    entries, chunks, offset = {}, [], 0
    for name, shape in tensor_shapes(plan["config"]).items():
        data = np.zeros(shape, dtype="<f4")
        if name == "model.embed_tokens.weight" or "norm.weight" in name:
            data.fill(1)
        elif name == "lm_head.weight":
            data[predicted].fill(1)
        payload = data.tobytes()
        entries[name] = dict(dtype="F32", shape=list(shape), data_offsets=[offset, offset + len(payload)])
        chunks.append(payload)
        offset += len(payload)
    header = json.dumps(entries).encode("utf-8")
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"".join(chunks))
    artifact = root / "decoder.leaf"
    report = export_decoder(root, artifact)
    return artifact, report


@pytest.mark.parametrize("eos,window,version", [(2, False, 2), (None, False, 2),
    (2, True, 3), ([2, 3], False, 4), ([2, 3], True, 4)])
def test_export_versions_add_metadata_only_when_multiple_eos_are_present(tmp_path, eos, window, version):
    artifact, report = fixture_model(tmp_path / "model", eos, window=window)
    blob = artifact.read_bytes()
    fields = HEADER.unpack_from(blob)
    assert fields[:2] == (f"LEAFDC0{version}".encode(), version)
    assert fields[3] % 64 == 0
    assert report["format"] == f"leaf-decoder-v{version}"
    assert fields[13] == (2 if window else 0)
    if version == 4:
        assert struct.unpack_from("<3I", blob, HEADER.size) == (2, 2, 3)
        assert report["eos_token_ids"] == [2, 3]


@pytest.fixture(scope="module")
def native_decoder():
    configured = os.environ.get("LEAF_TEST_DECODER_BIN")
    if not configured:
        pytest.skip("Set LEAF_TEST_DECODER_BIN to exercise the native artifact reader and generation")
    runtime = Path(configured).resolve()
    result = subprocess.run([str(runtime), "--runtime-info"], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["artifact_versions"] == [2, 3, 4]
    return runtime


@pytest.mark.parametrize("eos,predicted,window,expected", [
    ([2, 3], 3, False, [3]),  # Secondary terminator, the real Qwen metadata case.
    ([2, 3], 2, False, [2]),
    ([0, 3], 0, False, [0]),
    ([2, 3], 3, True, [3]),  # Multi-EOS and sliding-window semantics compose.
    (3, 3, False, [3]),      # v2 remains readable.
    (3, 3, True, [3]),       # v3 remains readable.
    (None, 0, False, [0, 0, 0, 0]),
])
def test_native_generation_honors_all_terminators_and_no_eos(native_decoder, tmp_path, eos, predicted, window, expected):
    artifact, _ = fixture_model(tmp_path / "model", eos, predicted=predicted, window=window)
    _, metrics = run_native(native_decoder, artifact, [[1]], mode="generate", generate=4)
    assert metrics["generated_tokens"] == expected
    if len(expected) == 1:
        assert metrics["decode_samples_ms"] == []  # No forward pass after a terminator.


@pytest.mark.parametrize("mutation", ["zero_count", "singleton_count", "excessive_count", "huge_count",
    "duplicate", "out_of_vocab", "primary_mismatch", "truncated", "old_magic"])
def test_native_reader_rejects_malformed_multi_eos_artifacts(native_decoder, tmp_path, mutation):
    artifact, _ = fixture_model(tmp_path / "model", [2, 3])
    blob = bytearray(artifact.read_bytes())
    count_values = {"zero_count": 0, "singleton_count": 1, "excessive_count": 1025,
                    "huge_count": 0xffffffff}
    if mutation in count_values:
        struct.pack_into("<I", blob, HEADER.size, count_values[mutation])
    elif mutation in ("duplicate", "out_of_vocab"):
        struct.pack_into("<I", blob, HEADER.size + 8, 2 if mutation == "duplicate" else 8)
    elif mutation == "primary_mismatch":
        struct.pack_into("<I", blob, 52, 1)
    elif mutation == "truncated":
        del blob[HEADER.size + 8:]
    else:
        blob[:8] = b"LEAFDC02"
        struct.pack_into("<I", blob, 8, 2)
    artifact.write_bytes(blob)
    with pytest.raises(RuntimeError, match="Native decoder failed"):
        run_native(native_decoder, artifact, [[1]], mode="generate", generate=4)


def test_old_runtime_rejects_new_multi_eos_semantics(tmp_path):
    configured = os.environ.get("LEAF_TEST_LEGACY_DECODER_BIN")
    if not configured:
        pytest.skip("Set LEAF_TEST_LEGACY_DECODER_BIN to check old-reader rejection")
    artifact, _ = fixture_model(tmp_path / "model", [2, 3])
    with pytest.raises(RuntimeError, match="unsupported artifact magic"):
        run_native(Path(configured).resolve(), artifact, [[1]], mode="generate", generate=4)
