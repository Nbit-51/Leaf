"""Cached exports must identify the exact source, plan and calibration request."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import validate_decoder as validation


@pytest.fixture
def native_fixture(tmp_path, monkeypatch):
    model, workdir = tmp_path / "model", tmp_path / "work"
    model.mkdir()
    workdir.mkdir()
    (model / "config.json").write_text(json.dumps(dict(model_type="llama", hidden_size=16, intermediate_size=32,
                                      num_attention_heads=2, num_hidden_layers=1, vocab_size=8)))
    (model / "model.safetensors").write_bytes(b"source weights identity")
    reference = np.zeros((3, 8), dtype=np.float32)
    np.save(workdir / "reference.npy", reference)
    runtime = tmp_path / "decoder.exe"
    runtime.write_bytes(b"native code identity")
    args = SimpleNamespace(model=model, workdir=workdir, plan=None, calibration=tmp_path / "calibration.npz",
                           bits=[8], executable=runtime, threads=1, generate=1, sequence_length=3,
                           runs=5, warmup=1, output=tmp_path / "validation.json")
    latency = {"prefill_p50_ms": 10.0, "decode_p50_ms": 1.0,
               "prefill_samples_ms": [10.0] * 5, "decode_samples_ms": [1.0] * 5, "threads": 1}
    baseline = {"model_config_sha256": validation.digest(model / "config.json"),
                "source_weight_sha256": {"model.safetensors": validation.digest(model / "model.safetensors")},
                "dataset_sha256": "2" * 64, "latency": {"eager": latency}}
    metadata = {"format": "leaf-activation-calibration-v1", "dataset_sha256": "1" * 64,
                "model_config_sha256": baseline["model_config_sha256"], "source_weight_sha256": baseline["source_weight_sha256"]}
    data = {"sequences": [[1, 2, 3]], "benchmark_ids": [1, 2, 3], "generation_ids": [1], "generated_tokens": [0]}
    monkeypatch.setattr(validation, "load_baseline_cache", lambda requested: (data, baseline))
    exports = []

    def export(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        exports.append(output.name)
        output.write_bytes(f"new export {len(exports)}".encode())

    def native(executable, artifact, sequences, mode="verify", **kwargs):
        if mode == "generate":
            return np.zeros(8), {"generated_tokens": [0]}
        if mode == "bench":
            return np.zeros(0), dict(latency, activation_bits=kwargs["activation_bits"])
        return reference.ravel().copy(), {}

    monkeypatch.setattr(validation.subprocess, "run", export)
    monkeypatch.setattr(validation, "run_native", native)

    def save(values=1.0, changed=None):
        saved = copy.deepcopy(metadata)
        if changed:
            saved.update(changed)
        np.savez(args.calibration, input_max=np.full(16, values), __metadata__=json.dumps(saved))

    save()
    return SimpleNamespace(args=args, baseline=baseline, metadata=metadata, exports=exports, save=save)


def test_missing_sidecar_is_reexported_and_identical_request_is_reused(native_fixture):
    run = native_fixture
    (run.args.workdir / "decoder-8.leaf").write_bytes(b"legacy cached artifact")
    (run.args.workdir / "decoder-8-smooth.leaf").write_bytes(b"legacy smoothed artifact")
    validation.native(run.args)
    assert run.exports == ["decoder-8.leaf", "decoder-8-smooth.leaf"]
    validation.native(run.args)
    assert len(run.exports) == 2
    for key in ("8", "8-smooth"):
        artifact = run.args.workdir / f"decoder-{key}.leaf"
        saved = json.loads(artifact.with_suffix(".provenance.json").read_text())
        assert saved["artifact_sha256"] == validation.digest(artifact)
        assert not artifact.with_suffix(".provenance.json.partial").exists()


def test_changed_calibration_arrays_or_split_only_reexport_smoothed_candidate(native_fixture):
    run = native_fixture
    validation.native(run.args)
    prior = json.loads(run.args.output.read_text())["native"]["8-smooth"]["export_provenance"]
    run.save(values=5.0)
    validation.native(run.args)
    assert run.exports == ["decoder-8.leaf", "decoder-8-smooth.leaf", "decoder-8-smooth.leaf"]
    current = json.loads(run.args.output.read_text())["native"]["8-smooth"]["export_provenance"]
    assert prior["calibration_sha256"] != current["calibration_sha256"]
    run.save(values=5.0, changed={"dataset_sha256": "3" * 64})
    validation.native(run.args)
    result = json.loads(run.args.output.read_text())
    assert run.exports[-1] == "decoder-8-smooth.leaf" and len(run.exports) == 4
    assert result["calibration"]["dataset_sha256"] == "3" * 64


@pytest.mark.parametrize("changed", [{"format": "unknown"}, {"model_config_sha256": "0" * 64},
    {"model_config_sha256": ""}, {"source_weight_sha256": {}}, {"source_weight_sha256": None},
    {"dataset_sha256": "2" * 64}, {"dataset_sha256": ""}, {"dataset_sha256": "not-a-sha256"}])
def test_invalid_calibration_provenance_is_rejected_even_with_matching_artifact_cache(native_fixture, monkeypatch, changed):
    run = native_fixture
    validation.native(run.args)
    count = len(run.exports)
    run.save(changed=changed)
    monkeypatch.setattr(validation, "run_native", lambda *args, **kwargs: pytest.fail("Invalid calibration reached native execution"))
    with pytest.raises(ValueError, match="Calibration"):
        validation.native(run.args)
    assert len(run.exports) == count


def test_plan_change_reexports_all_artifacts_without_changing_source_identity(native_fixture):
    run = native_fixture
    validation.native(run.args)
    plan, _ = validation.decoder_plan_identity(run.args)
    plan["config"]["epsilon"] *= 2
    run.args.plan = run.args.workdir / "custom-plan.json"
    run.args.plan.write_text(json.dumps(plan))
    validation.native(run.args)
    assert run.exports == ["decoder-8.leaf", "decoder-8-smooth.leaf"] * 2


def test_modified_artifact_or_malformed_sidecar_is_not_reused(native_fixture):
    run = native_fixture
    validation.native(run.args)
    artifact = run.args.workdir / "decoder-8-smooth.leaf"
    artifact.write_bytes(b"replaced payload")
    validation.native(run.args)
    assert run.exports[-1] == artifact.name and len(run.exports) == 3
    artifact.with_suffix(".provenance.json").write_text("{truncated")
    validation.native(run.args)
    assert run.exports[-1] == artifact.name and len(run.exports) == 4


def test_sidecar_write_is_atomic_and_preserves_previous_record_on_failure(native_fixture, monkeypatch):
    run = native_fixture
    validation.native(run.args)
    artifact = run.args.workdir / "decoder-8-smooth.leaf"
    sidecar = artifact.with_suffix(".provenance.json")
    original = sidecar.read_bytes()

    def interrupted(path, record):
        path.write_text("interrupted JSON")
        raise OSError("simulated interruption")

    monkeypatch.setattr(validation, "write_record", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        validation.write_export_provenance(artifact, {"new": "request"})
    assert sidecar.read_bytes() == original
    assert not sidecar.with_suffix(".json.partial").exists()
