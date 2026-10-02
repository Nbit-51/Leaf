"""Fast installed-runtime harness contracts without native/model execution."""
from __future__ import annotations

import io
import json
import os

import pytest

from tools import verify_package_runtime as package


def test_package_environment_is_child_only_and_has_no_compiler_override(monkeypatch, tmp_path):
    overrides = ("LEAF_DECODER_BIN", "LEAF_DISABLE_VNNI", "LEAF_DECODER_PROFILE", "LEAF_EXPERIMENTAL_FLOAT_TILES",
                 "LEAF_EXPERIMENTAL_FLOAT_GEMV", "LEAF_EXPERIMENTAL_FUTURE_KERNEL", "PYTHONPATH", "PYTHONHOME", "CXX")
    for name in overrides:
        monkeypatch.setenv(name, "parent value")
    monkeypatch.setenv("LEAF_UNRELATED_SETTING", "retained")
    monkeypatch.setenv("PATH", "parent compiler path")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    environment = package.isolated_environment(tmp_path / "cache")
    for name in overrides:
        assert name not in environment
        assert os.environ[name] == "parent value"
    assert environment["LEAF_UNRELATED_SETTING"] == "retained"
    assert os.environ["PATH"] == "parent compiler path"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["PATH"] == str(package.Path(r"C:\Windows") / "System32") if os.name == "nt" else environment["PATH"] == ""


def test_installed_command_uses_isolated_python_and_explicit_package_path():
    payload = {"installed_dir": "installed directory with spaces", "phase": "probe"}
    command = package.child_command(payload)
    assert command[0] == package.sys.executable
    assert command[1:4] == ["-I", "-u", "-c"]
    assert json.loads(command[-1]) == payload
    assert "sys.path.insert(0, str(installed))" in command[4]
    assert "relative_to(installed)" in command[4]
    assert '"runtime_environment_policy": kernel_policy' in command[4]
    assert 'name.startswith("LEAF_EXPERIMENTAL_")' in command[4]


class Process:
    def __init__(self, stderr, returncode=0):
        self.stdout = io.BytesIO(b"\n\nGenerated text\n")
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode

    def wait(self):
        return self.returncode


def test_external_timer_distinguishes_output_and_visible_text(monkeypatch, tmp_path):
    check = {"heavy_model_libraries_loaded": [], "bundled_runtime_selected": True}
    error = "Preparing 32-bit native artifact...\n" + package.MARKER + json.dumps(check) + "\n"
    monkeypatch.setattr(package.subprocess, "Popen", lambda *args, **kwargs: Process(error.encode()))
    measured = package.run_process(["isolated-python"], {}, tmp_path)
    assert measured["package_check"] == check
    assert measured["preparation_message_seen"] is True
    assert measured["stdout"] == "\n\nGenerated text\n"
    assert measured["external_wall_ms"] >= measured["external_first_visible_text_ms"] >= measured["external_first_stdout_byte_ms"] >= 0


@pytest.mark.parametrize("error", [b"no provenance marker", (package.MARKER + "{}\n" + package.MARKER + "{}\n").encode()])
def test_missing_or_ambiguous_package_provenance_is_rejected(monkeypatch, tmp_path, error):
    monkeypatch.setattr(package.subprocess, "Popen", lambda *args, **kwargs: Process(error))
    with pytest.raises(RuntimeError, match="exactly one"):
        package.run_process(["isolated-python"], {}, tmp_path)


def test_failed_installed_process_reports_error(monkeypatch, tmp_path):
    monkeypatch.setattr(package.subprocess, "Popen", lambda *args, **kwargs: Process(b"native failure", 1))
    with pytest.raises(RuntimeError, match="native failure"):
        package.run_process(["isolated-python"], {}, tmp_path)


@pytest.fixture
def model_reference(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"synthetic source identity")
    baseline = {"model_config_sha256": package.digest(model / "config.json"),
                "source_weight_sha256": {"model.safetensors": package.digest(model / "model.safetensors")},
                "generation_prompt": "Test generation", "generated_tokens": [1, 2], "threads": 1, "pinned_cpu": 2}
    return model, baseline


def test_reference_identity_and_settings_are_preserved(model_reference):
    model, baseline = model_reference
    assert package.validate_reference(model, {"pytorch": baseline}) is baseline


@pytest.mark.parametrize("field,value", [("model_config_sha256", "stale"), ("source_weight_sha256", {}),
    ("generation_prompt", ""), ("generated_tokens", []), ("generated_tokens", [True]),
    ("threads", 0), ("threads", True), ("pinned_cpu", -1), ("pinned_cpu", True)])
def test_invalid_reference_cannot_qualify_package_run(model_reference, field, value):
    model, baseline = model_reference
    baseline[field] = value
    with pytest.raises(ValueError):
        package.validate_reference(model, {"pytorch": baseline})
