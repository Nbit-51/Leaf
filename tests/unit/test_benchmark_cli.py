"""Cached CLI benchmark contracts without trained/native execution."""
from __future__ import annotations

import copy
import json
import os
import platform

import pytest

from leaf import cli
from tools import benchmark_cli as benchmark


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.delenv("LEAF_DISABLE_VNNI", raising=False)
    model, artifacts = tmp_path / "model", tmp_path / "artifacts"
    model.mkdir(); artifacts.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"synthetic weights")
    executable = tmp_path / "native.exe"
    executable.write_bytes(b"frozen native binary")
    baseline = {"model_config_sha256": benchmark.digest(model / "config.json"),
                "source_weight_sha256": {"model.safetensors": benchmark.digest(model / "model.safetensors")},
                "generation_prompt": "A reference prompt", "generated_tokens": [1, 2],
                "threads": 1, "pinned_cpu": 2, "platform": platform.platform(), "cpu": platform.processor()}
    (artifacts / "tokens.json").write_text(json.dumps({"generation_ids": [3, 4], "generated_tokens": [1, 2]}))
    baseline["quality_cache_sha256"] = {"tokens.json": benchmark.digest(artifacts / "tokens.json")}
    native = {}
    for key, bits, activations, tokens, eligible in (("32", 32, 32, [1, 2], False),
                                                  ("8-smooth", 8, 8, [1, 7], True)):
        artifact = artifacts / f"decoder-{key}.leaf"
        artifact.write_bytes(key.encode())
        native[key] = {"artifact": artifact.name, "artifact_sha256": benchmark.digest(artifact),
                       "weight_bits": bits, "activation_bits": activations,
                       "quality_gate_passed": True, "eligible_for_automatic_selection": eligible,
                       "latency_stability": {"passed": True},
                       "latency": {"decode_p50_ms": 1.0, "threads": 1, "activation_bits": activations},
                       "generated_tokens": tokens}
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"pytorch": baseline, "native": native,
                                  "native_executable_sha256": benchmark.digest(executable)}))
    monkeypatch.setattr(cli, "validate_model_generation", lambda model: None)
    monkeypatch.setattr(cli, "encode_prompt", lambda model, prompt, raw=False: (None, [3, 4]))
    return model, artifacts, profile, executable, tmp_path / "cache"


def rewrite_profile(frozen, change):
    profile = frozen[2]
    data = json.loads(profile.read_text())
    change(data)
    profile.write_text(json.dumps(data))


def test_original_profile_and_eligible_mixed_generation_are_preserved(frozen):
    baseline, cases, provenance = benchmark.stage_cache(*frozen, ["32", "auto"])
    directory = cli.model_directory(frozen[0], frozen[4])
    assert (directory / "validation.json").read_bytes() == frozen[2].read_bytes()
    assert cases["auto"]["native_case"] == "8-smooth"
    assert cases["auto"]["expected_tokens"] == [1, 7]
    assert cases["auto"]["expected_tokens"] != baseline["generated_tokens"]
    assert provenance["native_executable_sha256"] == benchmark.digest(frozen[3])
    assert set(provenance["staged_artifacts"]) == {"decoder-32.leaf", "decoder-8-smooth.leaf"}
    assert provenance["prompt_token_ids"] == [3, 4]


def test_cross_volume_copy_fallback_remains_hash_checked(frozen, monkeypatch):
    def unavailable(*args):
        raise OSError("no hard links")
    monkeypatch.setattr(benchmark.os, "link", unavailable)
    _, _, provenance = benchmark.stage_cache(*frozen, ["32"])
    assert provenance["staged_artifacts"]["decoder-32.leaf"]["method"] == "copy"


def test_explicit_fp32_must_have_cached_cli_filename(frozen):
    rewrite_profile(frozen, lambda data: data["native"]["32"].update(artifact="different-fp32.leaf"))
    with pytest.raises(ValueError, match="CLI artifact name"):
        benchmark.stage_cache(*frozen, ["32"])


@pytest.mark.parametrize("mutate,match", [
    (lambda data: data.update(native_executable_sha256="changed"), "executable"),
    (lambda data: data["pytorch"].update(platform="different OS"), "platform"),
    (lambda data: data["pytorch"].update(cpu="different CPU"), "CPU"),
    (lambda data: data["native"]["8-smooth"].update(eligible_for_automatic_selection=False), "eligible"),
    (lambda data: data["native"]["8-smooth"].update(quality_gate_passed=False), "eligible"),
    (lambda data: data["native"]["8-smooth"].update(artifact_sha256="changed"), "artifact"),
    (lambda data: data["native"]["8-smooth"].update(artifact="../outside.leaf"), "filenames"),
    (lambda data: data["native"]["8-smooth"].update(generated_tokens=[]), "generation"),
])
def test_changed_or_ineligible_profiles_fail_closed(frozen, mutate, match):
    rewrite_profile(frozen, mutate)
    with pytest.raises(ValueError, match=match):
        benchmark.stage_cache(*frozen, ["32", "auto"])


def test_changed_prompt_format_is_not_timed_against_other_reference(frozen, monkeypatch):
    monkeypatch.setattr(cli, "encode_prompt", lambda *args: (None, [9, 10]))
    with pytest.raises(ValueError, match="prompt token IDs"):
        benchmark.stage_cache(*frozen, ["32", "auto"], raw=True)


def test_changed_generation_cache_is_rejected(frozen):
    (frozen[1] / "tokens.json").write_text(json.dumps({"generation_ids": [9], "generated_tokens": [1, 2]}))
    with pytest.raises(ValueError, match="SHA256"):
        benchmark.stage_cache(*frozen, ["32", "auto"])


def test_different_isa_policy_is_not_silently_removed(frozen, monkeypatch):
    monkeypatch.setenv("LEAF_DISABLE_VNNI", "1")
    with pytest.raises(ValueError, match="ISA policy"):
        benchmark.stage_cache(*frozen, ["32", "auto"])
    assert os.environ["LEAF_DISABLE_VNNI"] == "1"


def test_child_invokes_source_command_with_frozen_prompt_settings(tmp_path):
    baseline = {"generation_prompt": "same raw text", "generated_tokens": [1, 2], "threads": 1, "pinned_cpu": 2}
    command = benchmark.child_command(tmp_path, baseline, "auto", tmp_path / "metrics.json")
    assert command[1:4] == ["-I", "-u", "-c"]
    payload = json.loads(command[-1])
    arguments = payload["arguments"]
    assert arguments[0:2] == ["run", str(tmp_path)]
    assert arguments[arguments.index("--prompt") + 1] == "same raw text"
    assert arguments[arguments.index("--max-tokens") + 1] == "2"
    assert arguments[arguments.index("--cpu") + 1] == "2"
    assert "--raw" not in arguments
    assert "cli.select_configuration" not in command[4]  # no second timed hash preflight


@pytest.fixture
def observation():
    return ({"external_wall_ms": 30.0, "external_first_stdout_byte_ms": 10.0,
             "external_first_visible_text_ms": 11.0, "preparation_message_seen": False,
             "package_check": {"source_cli": True, "heavy_model_libraries_loaded": []}},
            {"load_ms": 1.0, "prefill_p50_ms": 2.0, "decode_p50_ms": 1.0,
             "time_to_first_token_ms": 8.0, "end_to_end_ms": 25.0,
             "threads": 1, "weight_bits": 8, "activation_bits": 8, "generated_tokens": [1, 7]},
            {"weight_bits": 8, "activation_bits": 8, "expected_tokens": [1, 7]},
            {"threads": 1, "generated_tokens": [1, 2]})


def test_quantized_run_checks_its_exact_case_not_fp32_tokens(observation):
    result = benchmark.check_run(*observation)
    assert result["generated_tokens_match_selected_native"] is True
    assert result["generated_tokens_match_pytorch"] is False
    assert result["frontend_and_startup_remainder_until_first_visible_ms"] == 8.0


@pytest.mark.parametrize("field,value", [("weight_bits", 32), ("activation_bits", 32),
    ("threads", 2), ("generated_tokens", [1, 2]), ("load_ms", float("nan")),
    ("time_to_first_token_ms", None), ("decode_p50_ms", True)])
def test_wrong_settings_tokens_or_invalid_timing_fail_closed(observation, field, value):
    measured, metrics, case, baseline = copy.deepcopy(observation)
    metrics[field] = value
    with pytest.raises(AssertionError):
        benchmark.check_run(measured, metrics, case, baseline)


def test_unexpected_export_cannot_be_called_cached(observation):
    observation[0]["preparation_message_seen"] = True
    with pytest.raises(AssertionError, match="exported"):
        benchmark.check_run(*observation)


def test_summary_retains_all_five_samples_and_stability(observation):
    result = benchmark.check_run(*observation)
    summary = benchmark.summarize([copy.deepcopy(result) for _ in range(5)])
    assert summary["external_wall_ms"]["samples_ms"] == [30.0] * 5
    assert summary["external_wall_ms"]["p50_ms"] == 30.0
    assert summary["external_wall_ms"]["stability"]["passed"] is True
