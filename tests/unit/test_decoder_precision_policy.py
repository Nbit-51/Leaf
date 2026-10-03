"""Architecture-independent protected weights and mixed-precision metadata."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.decoder_plan import make_plan, tensor_shapes, validate_plan
from tools import export_decoder as exporter
from tools import validate_decoder as validator


@pytest.fixture
def model_fixture(tmp_path):
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    config = dict(model_type="llama", hidden_size=64, intermediate_size=128,
                  num_attention_heads=4, num_hidden_layers=1, vocab_size=32,
                  max_position_embeddings=128, tie_word_embeddings=True)
    (snapshot / "config.json").write_text(json.dumps(config))
    plan = make_plan(config)
    # An explicit plan changes positional semantics without a new runtime
    # model-name branch, and exercises learned-position protection as well.
    plan["config"].update(position=1, rotary_dim=0)
    plan["tensors"]["model.position_embeddings.weight"] = {"source": "local.position.weight"}
    plan["source_architecture"] = "custom_decoder"
    validate_plan(plan)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    rng = np.random.default_rng(1024)
    arrays, entries, payloads, offset = {}, {}, [], 0
    for name, shape in tensor_shapes(plan["config"]).items():
        source = plan["tensors"][name]["source"]
        if source in arrays:
            continue
        array = rng.normal(size=shape).astype("<f4")
        arrays[source] = array
        data = array.tobytes()
        entries[source] = {"dtype": "F32", "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        payloads.append(data)
        offset += len(data)
    header = json.dumps(entries).encode()
    (snapshot / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"".join(payloads))
    return SimpleNamespace(snapshot=snapshot, plan=plan, plan_path=plan_path, arrays=arrays,
                           artifact=tmp_path / "mixed.leaf")


def descriptors(filename):
    blob = filename.read_bytes()
    header = exporter.HEADER.unpack_from(blob)
    position, entries = exporter.HEADER.size, {}
    for _ in range(header[2]):
        length = struct.unpack_from("<I", blob, position)[0]
        position += 4
        name = blob[position:position + length].decode()
        position += length
        entries[name] = exporter.DESCRIPTOR.unpack_from(blob, position)
        position += exporter.DESCRIPTOR.size
    return blob, header[3], entries


def test_protected_io_policy_uses_plan_tensors_not_architecture_name(model_fixture):
    expected = ["model.embed_tokens.weight", "model.position_embeddings.weight", "lm_head.weight"]
    assert exporter.protected_int8_tensors(model_fixture.plan) == expected
    plan = make_plan(dict(model_type="llama", hidden_size=16, intermediate_size=32,
                          num_attention_heads=2, num_hidden_layers=1, vocab_size=32))
    assert exporter.protected_int8_tensors(plan) == ["model.embed_tokens.weight", "lm_head.weight"]
    plan["source_architecture"] = "unrelated_import_adapter"
    assert exporter.protected_int8_tensors(plan) == ["model.embed_tokens.weight", "lm_head.weight"]


@pytest.mark.parametrize("bits", [8, 4])
def test_protected_weights_remain_exact_fp32_and_share_tied_storage(model_fixture, bits):
    kept = exporter.protected_int8_tensors(model_fixture.plan)
    record = exporter.export_decoder(model_fixture.snapshot, model_fixture.artifact, bits,
                                     plan_path=model_fixture.plan_path, keep_fp32_tensors=kept)
    blob, base, entries = descriptors(model_fixture.artifact)
    assert record["keep_fp32_tensors"] == sorted(kept)
    for name in kept:
        descriptor = entries[name]
        assert descriptor[3:5] == (32, 0)
        assert descriptor[8] == 0
        original = model_fixture.arrays[model_fixture.plan["tensors"][name]["source"]].tobytes()
        assert blob[base + descriptor[5]:base + descriptor[5] + descriptor[6]] == original
    assert entries["model.embed_tokens.weight"] == entries["lm_head.weight"]
    assert entries["model.layers.0.self_attn.q_proj.weight"][3] == bits
    assert entries["model.norm.weight"][3] == 32


def test_protecting_only_one_tied_operator_does_not_share_different_precision(model_fixture):
    exporter.export_decoder(model_fixture.snapshot, model_fixture.artifact, 8,
                            plan_path=model_fixture.plan_path, keep_fp32_tensors=["lm_head.weight"])
    _, _, entries = descriptors(model_fixture.artifact)
    assert entries["model.embed_tokens.weight"][3] == 8
    assert entries["lm_head.weight"][3] == 32
    assert entries["model.embed_tokens.weight"][5] != entries["lm_head.weight"][5]


@pytest.mark.parametrize("requested", [["unknown.weight"], ["*"], ["model.embed_tokens"],
    ["model.layers.0.self_attn.q_proj.input_scale"], "lm_head.weight", [None], [1], {}])
def test_invalid_or_noncanonical_precision_policy_is_rejected_before_output(model_fixture, requested):
    with pytest.raises(ValueError, match="FP32 tensor"):
        exporter.export_decoder(model_fixture.snapshot, model_fixture.artifact, 8,
                                plan_path=model_fixture.plan_path, keep_fp32_tensors=requested)
    assert not model_fixture.artifact.exists()
    assert not model_fixture.artifact.with_suffix(".leaf.partial").exists()


def test_redundant_and_duplicate_protected_names_are_normalized(model_fixture):
    names = ["lm_head.weight", "model.norm.weight", "lm_head.weight"]
    record = exporter.export_decoder(model_fixture.snapshot, model_fixture.artifact, 8,
                                     plan_path=model_fixture.plan_path, keep_fp32_tensors=names)
    assert record["keep_fp32_tensors"] == ["lm_head.weight", "model.norm.weight"]


def save_calibration(model_fixture, filename):
    arrays = {name: np.full(shape[1], 2.0, dtype=np.float32)
              for name, shape in tensor_shapes(model_fixture.plan["config"]).items()
              if shape[0] > 1 and not name.startswith(("model.embed_tokens.", "model.position_embeddings."))}
    metadata = {"format": "leaf-activation-calibration-v1", "dataset_sha256": "1" * 64,
                "model_config_sha256": validator.digest(model_fixture.snapshot / "config.json"),
                "source_weight_sha256": {"model.safetensors": validator.digest(model_fixture.snapshot / "model.safetensors")}}
    np.savez(filename, **arrays, __metadata__=json.dumps(metadata))
    return arrays, metadata


def test_protected_linear_bypasses_smoothing_and_input_scale(model_fixture, tmp_path):
    filename = tmp_path / "calibration.npz"
    save_calibration(model_fixture, filename)
    exporter.export_decoder(model_fixture.snapshot, model_fixture.artifact, 8,
                            plan_path=model_fixture.plan_path, calibration_path=filename,
                            keep_fp32_tensors=["lm_head.weight"])
    blob, base, entries = descriptors(model_fixture.artifact)
    descriptor = entries["lm_head.weight"]
    assert "lm_head.input_scale" not in entries
    assert "model.layers.0.self_attn.q_proj.input_scale" in entries
    original = model_fixture.arrays[model_fixture.plan["tensors"]["lm_head.weight"]["source"]].tobytes()
    assert blob[base + descriptor[5]:base + descriptor[5] + descriptor[6]] == original


def test_export_cli_exposes_exact_canonical_protection(model_fixture, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["export_decoder", "--model", str(model_fixture.snapshot),
                         "--output", str(model_fixture.artifact), "--plan", str(model_fixture.plan_path),
                         "--bits", "8", "--keep-fp32-tensors", "lm_head.weight"])
    exporter.main()
    assert json.loads(capsys.readouterr().out)["keep_fp32_tensors"] == ["lm_head.weight"]


def test_validation_adds_precision_candidates_without_replacing_originals(model_fixture, monkeypatch, tmp_path):
    workdir = tmp_path / "validation"
    workdir.mkdir()
    reference = np.zeros((2, 32), dtype=np.float32)
    np.save(workdir / "reference.npy", reference)
    calibration = tmp_path / "calibration.npz"
    save_calibration(model_fixture, calibration)
    runtime = tmp_path / "decoder.exe"
    runtime.write_bytes(b"test executable")
    data = {"sequences": [[0, 1]], "benchmark_ids": [0, 1], "generation_ids": [0, 1], "generated_tokens": [0]}
    latency = {"prefill_p50_ms": 50.0, "decode_p50_ms": 10.0,
               "prefill_samples_ms": [50.0] * 5, "decode_samples_ms": [10.0] * 5,
               "latency_workload": dict(validator.LATENCY_WORKLOAD)}
    pytorch = {"dataset_sha256": "2" * 64, "latency": {"eager": latency},
               "latency_workload": dict(validator.LATENCY_WORKLOAD),
               "model_config_sha256": validator.digest(model_fixture.snapshot / "config.json"),
               "source_weight_sha256": {"model.safetensors": validator.digest(model_fixture.snapshot / "model.safetensors")}}
    args = SimpleNamespace(model=model_fixture.snapshot, workdir=workdir, plan=model_fixture.plan_path,
                           executable=runtime, bits=[32, 8, 4], calibration=calibration, protected_int8=True,
                           grouped_int8=True, grouped_int8_smooth=True,
                           threads=1, generate=1, runs=5, warmup=1, sequence_length=2, output=tmp_path / "result.json")
    monkeypatch.setattr(validator, "load_baseline_cache", lambda requested: (data, pytorch))
    commands = []

    def fake_export(command, **kwargs):
        commands.append(command)
        kept = command[command.index("--keep-fp32-tensors") + 1:] if "--keep-fp32-tensors" in command else None
        calibration_path = calibration if "--calibration" in command else None
        int8_group_size = int(command[command.index("--int8-group-size") + 1]) if "--int8-group-size" in command else 0
        exporter.export_decoder(model_fixture.snapshot, Path(command[command.index("--output") + 1]),
                                int(command[command.index("--bits") + 1]), plan_path=model_fixture.plan_path,
                                calibration_path=calibration_path, keep_fp32_tensors=kept, int8_group_size=int8_group_size)

    def fake_native(executable, artifact, sequences, mode="verify", **kwargs):
        if mode == "generate":
            return np.zeros(32, dtype=np.float32), {"generated_tokens": [0]}
        if mode == "bench":
            return np.zeros(0, dtype=np.float32), dict(latency, threads=1, activation_bits=kwargs["activation_bits"])
        return reference.ravel().copy(), {}

    monkeypatch.setattr(validator.subprocess, "run", fake_export)
    monkeypatch.setattr(validator, "run_native", fake_native)
    validator.native(args)
    result = json.loads(args.output.read_text())
    assert validator.latency_workload_matches(result)
    assert all(validator.latency_workload_matches(case["latency"]) for case in result["native"].values())
    assert set(result["native"]) == {"32", "8", "4", "8-smooth", "8-protected", "8-grouped", "8-grouped-smooth"}
    assert result["native"]["8-protected"]["keep_fp32_tensors"] == sorted(exporter.protected_int8_tensors(model_fixture.plan))
    assert result["native"]["8-protected"]["activation_bits"] == 32
    assert result["native"]["8-protected"]["artifact"] == "decoder-8-protected.leaf"
    assert result["native"]["8"]["keep_fp32_tensors"] == []
    assert result["native"]["8-smooth"]["keep_fp32_tensors"] == []
    assert result["native"]["8-grouped"]["int8_group_size"] == 64
    assert result["native"]["8-grouped"]["activation_bits"] == 8
    assert result["native"]["8-grouped-smooth"]["int8_group_size"] == 64
    assert len([command for command in commands if "--keep-fp32-tensors" in command]) == 1


@pytest.mark.parametrize("group", [0, 4, 16, 64])
def test_grouped_int8_descriptors_and_scale_count(model_fixture, group):
    record = exporter.export_decoder(model_fixture.snapshot, model_fixture.artifact, 8,
                                     plan_path=model_fixture.plan_path, int8_group_size=group)
    _, _, entries = descriptors(model_fixture.artifact)
    assert record["int8_group_size"] == group
    for descriptor in entries.values():
        if descriptor[3] != 8:
            continue
        rows, cols, actual_group = descriptor[1], descriptor[2], descriptor[4]
        assert actual_group == (group or cols)
        assert descriptor[8] == rows * (cols // actual_group)


@pytest.mark.parametrize("group", [-1, -2, 1, 3, True, None, "64", 6, 256])
def test_invalid_or_nondividing_int8_group_is_rejected(model_fixture, group):
    with pytest.raises(ValueError, match="INT8 group"):
        exporter.export_decoder(model_fixture.snapshot, model_fixture.artifact, 8,
                                plan_path=model_fixture.plan_path, int8_group_size=group)
    assert not model_fixture.artifact.exists()


def test_grouped_int8_can_combine_protection_and_smoothing(model_fixture, tmp_path):
    filename = tmp_path / "calibration.npz"
    save_calibration(model_fixture, filename)
    exporter.export_decoder(model_fixture.snapshot, model_fixture.artifact, 8,
                            plan_path=model_fixture.plan_path, calibration_path=filename,
                            keep_fp32_tensors=exporter.protected_int8_tensors(model_fixture.plan), int8_group_size=64)
    _, _, entries = descriptors(model_fixture.artifact)
    assert entries["lm_head.weight"][3] == 32
    assert entries["model.layers.0.self_attn.q_proj.weight"][4] == 64
    assert "lm_head.input_scale" not in entries


def test_export_cli_exposes_grouped_int8(model_fixture, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["export_decoder", "--model", str(model_fixture.snapshot),
                         "--output", str(model_fixture.artifact), "--plan", str(model_fixture.plan_path),
                         "--bits", "8", "--int8-group-size", "64"])
    exporter.main()
    assert json.loads(capsys.readouterr().out)["int8_group_size"] == 64


@pytest.mark.parametrize("bits", [32, 8, 4])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("protected", [False, True])
def test_nonfinite_source_is_rejected_in_every_precision_and_preserves_previous_artifact(model_fixture, bits, value, protected):
    filename = model_fixture.snapshot / "model.safetensors"
    base, index = exporter.tensor_index(filename)
    source = model_fixture.plan["tensors"]["model.embed_tokens.weight"]["source"]
    blob = bytearray(filename.read_bytes())
    start = base + index[source]["data_offsets"][0]
    blob[start:start + 4] = np.asarray(value, dtype="<f4").tobytes()
    filename.write_bytes(blob)
    model_fixture.artifact.write_bytes(b"previous valid artifact")
    kept = ["model.embed_tokens.weight"] if protected else []
    with pytest.raises(ValueError, match="Non-finite source tensor"):
        exporter.export_decoder(model_fixture.snapshot, model_fixture.artifact, bits,
                                plan_path=model_fixture.plan_path, keep_fp32_tensors=kept)
    assert model_fixture.artifact.read_bytes() == b"previous valid artifact"
    assert not model_fixture.artifact.with_suffix(".leaf.partial").exists()


def test_nonfinite_transformed_array_is_rejected(model_fixture, monkeypatch):
    def invalid_transform(array, transforms):
        result = np.array(array, copy=True)
        result.flat[0] = np.nan
        return result
    monkeypatch.setattr(exporter, "transform_array", invalid_transform)
    with pytest.raises(ValueError, match="Non-finite transformed tensor"):
        exporter.export_decoder(model_fixture.snapshot, model_fixture.artifact, 32, plan_path=model_fixture.plan_path)
    assert not model_fixture.artifact.exists()
    assert not model_fixture.artifact.with_suffix(".leaf.partial").exists()


def test_smoothing_overflow_is_rejected_before_serialization(model_fixture, tmp_path, monkeypatch):
    filename = tmp_path / "calibration.npz"
    save_calibration(model_fixture, filename)
    # Keep inputs finite while forcing an otherwise invalid channel scaling
    # to test the final transformed payload guard independently of provenance.
    monkeypatch.setattr(exporter, "channel_factors", lambda array, maximum, alpha: np.full(array.shape[1], np.finfo(np.float32).max))
    with pytest.raises(ValueError, match="Non-finite export tensor"):
        exporter.export_decoder(model_fixture.snapshot, model_fixture.artifact, 8,
                                plan_path=model_fixture.plan_path, calibration_path=filename)
    assert not model_fixture.artifact.exists()
    assert not model_fixture.artifact.with_suffix(".leaf.partial").exists()
