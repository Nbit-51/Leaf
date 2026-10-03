"""Fast CLI contracts: local caches, safe selection and lightweight tokenization."""
from __future__ import annotations

import hashlib
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from leaf import cli
from tools.validate_decoder import GATES, LATENCY_WORKLOAD


NATIVE_POLICY_NAMES = ("LEAF_DISABLE_VNNI", "LEAF_EXPERIMENTAL_FLOAT_TILES", "LEAF_EXPERIMENTAL_FLOAT_GEMV")


def latency_record(prefill, decode, activations=32):
    return {"prefill_p50_ms": prefill, "decode_p50_ms": decode,
            "prefill_samples_ms": [prefill] * 7, "decode_samples_ms": [decode] * 7,
            "threads": 1, "activation_bits": activations, "latency_workload": dict(LATENCY_WORKLOAD)}


@pytest.fixture
def profile_files(tmp_path, monkeypatch):
    for name in NATIVE_POLICY_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli.platform, "platform", lambda: "test-os")
    monkeypatch.setattr(cli.platform, "processor", lambda: "test-cpu")
    runtime = tmp_path / "decoder.exe"
    runtime.write_bytes(b"portable runtime")
    directory = tmp_path / "model-cache"
    directory.mkdir()
    result = {
        "pytorch": {"platform": "test-os", "cpu": "test-cpu", "latency_workload": dict(LATENCY_WORKLOAD),
                    "threads": 1, "quality_gate_thresholds": copy.deepcopy(GATES),
                    "latency": {"eager": latency_record(100, 20), "sdpa": latency_record(105, 21)}},
        "latency_workload": dict(LATENCY_WORKLOAD),
        "native_executable_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
        "native_policy": {name: False for name in NATIVE_POLICY_NAMES},
        "native": {},
    }

    def candidate(key="8", latency=7.0, *, bits=8, activations=32, eligible=True):
        artifact = directory / f"decoder-{key}.leaf"
        artifact.write_bytes(f"artifact-{key}".encode())
        result["native"][key] = {
            "artifact": artifact.name,
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "weight_bits": bits,
            "activation_bits": activations,
            "quality": {"next_token_agreement": 1.0, "perplexity_ratio": 1.0},
            "quality_gate_passed": True,
            "eligible_for_automatic_selection": eligible,
            "latency_stability": {"passed": True},
            "latency": latency_record(120 if bits == 32 else 90, latency, activations),
        }
        return artifact

    def save():
        (directory / "validation.json").write_text(json.dumps(result))

    candidate("32", 12.0, bits=32, eligible=False)
    return SimpleNamespace(directory=directory, runtime=runtime, result=result, candidate=candidate, save=save)


def test_automatic_selection_without_profile_is_fp32(tmp_path):
    assert cli.select_configuration(tmp_path, tmp_path / "unused", "auto") == (32, 32, None)


@pytest.mark.parametrize("location", ["profile", "pytorch", "candidate"])
@pytest.mark.parametrize("mutation", ["missing", "full_logits", "boolean_integer", "unknown", "null"])
def test_legacy_or_mismatched_generation_workload_cannot_enable_auto(profile_files, location, mutation):
    case = profile_files
    case.candidate()
    target = (case.result if location == "profile" else case.result["pytorch"] if location == "pytorch"
              else case.result["native"]["8"]["latency"])
    if mutation == "missing":
        target.pop("latency_workload")
    elif mutation == "null":
        target["latency_workload"] = None
    elif mutation == "full_logits":
        target["latency_workload"]["logits"] = "all_prefix_tokens"
    elif mutation == "boolean_integer":
        target["latency_workload"]["use_cache"] = 1
    else:
        target["latency_workload"]["unknown"] = True
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (32, 32, None)


def test_changed_isa_policy_does_not_reuse_speed_profile(profile_files, monkeypatch):
    case = profile_files
    case.candidate("8-smooth", 5.0, activations=8)
    case.save()
    monkeypatch.setenv("LEAF_DISABLE_VNNI", "1")
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (32, 32, None)


@pytest.mark.parametrize("flag", ["LEAF_EXPERIMENTAL_FLOAT_TILES", "LEAF_EXPERIMENTAL_FLOAT_GEMV"])
@pytest.mark.parametrize("value", ["1", "0"])
def test_experimental_environment_invalidates_automatic_precision(profile_files, monkeypatch, flag, value):
    case = profile_files
    case.candidate()
    case.save()
    monkeypatch.setenv(flag, value)  # native flags use nonempty-string semantics
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (32, 32, None)


@pytest.mark.parametrize("flag", NATIVE_POLICY_NAMES)
def test_experimental_profile_cannot_be_reused_after_flag_is_removed(profile_files, monkeypatch, flag):
    case = profile_files
    case.candidate()
    case.result["native_policy"][flag] = True
    case.save()
    monkeypatch.setenv(flag, "1")
    monkeypatch.delenv(flag)
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (32, 32, None)


@pytest.mark.parametrize("policy", [None, {}, [], False, 0, "default",
    {"LEAF_DISABLE_VNNI": False},
    {**{name: False for name in NATIVE_POLICY_NAMES}, "unknown_flag": False},
    {**{name: False for name in NATIVE_POLICY_NAMES}, "LEAF_EXPERIMENTAL_FLOAT_TILES": 0},
    {**{name: False for name in NATIVE_POLICY_NAMES}, "LEAF_EXPERIMENTAL_FLOAT_GEMV": "false"}])
def test_malformed_native_policy_cannot_enable_auto_precision(profile_files, policy):
    case = profile_files
    case.candidate()
    case.result["native_policy"] = policy
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (32, 32, None)


def test_missing_legacy_native_policy_means_default_flags(profile_files):
    case = profile_files
    artifact = case.candidate()
    case.result.pop("native_policy")
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (8, 32, artifact)


@pytest.mark.parametrize("flag", ["LEAF_EXPERIMENTAL_FLOAT_TILES", "LEAF_EXPERIMENTAL_FLOAT_GEMV"])
def test_explicit_precision_override_remains_explicit_with_experimental_flag(tmp_path, monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    assert cli.select_configuration(tmp_path, tmp_path / "unused", "8") == (8, 32, None)


def test_cpu_pin_mismatch_does_not_reuse_speed_profile(profile_files):
    case = profile_files
    case.candidate("8", 5.0)
    case.result["pytorch"]["pinned_cpu"] = 2
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (32, 32, None)
    assert cli.select_configuration(case.directory, case.runtime, "auto", cpu=2)[0] == 8


@pytest.mark.parametrize("requested,activations,expected", [
    ("8", "auto", (8, 32, None)),
    ("4", "8", (4, 8, None)),
    ("32", "32", (32, 32, None)),
])
def test_explicit_precision_is_not_silently_overridden(tmp_path, requested, activations, expected):
    assert cli.select_configuration(tmp_path, tmp_path / "unused", requested,
                                    activation_bits=activations) == expected


def test_auto_selects_fastest_validated_configuration(profile_files):
    case = profile_files
    case.candidate("8", 8.0)
    fastest = case.candidate("8-smooth", 5.0, activations=8)
    case.candidate("4", 1.0, bits=4, eligible=False)
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (8, 8, fastest)


def test_auto_respects_explicit_activation_policy(profile_files):
    case = profile_files
    selected = case.candidate("8", 8.0)
    case.candidate("8-smooth", 5.0, activations=8)
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto", activation_bits="32") == (8, 32, selected)


def test_custom_configuration_names_do_not_require_model_or_precision_id_parsing(profile_files):
    case = profile_files
    selected = case.candidate("calibrated-native", bits=8, activations=8)
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (8, 8, selected)


@pytest.mark.parametrize("mismatch", ["os", "cpu", "runtime", "threads", "activation", "rejected"])
def test_rejected_or_nonmatching_profile_falls_back_to_fp32(profile_files, mismatch):
    case = profile_files
    case.candidate()
    entry = case.result["native"]["8"]
    options = {}
    if mismatch == "os":
        case.result["pytorch"]["platform"] = "another-os"
    elif mismatch == "cpu":
        case.result["pytorch"]["cpu"] = "another-cpu"
    elif mismatch == "runtime":
        case.runtime.write_bytes(b"changed native kernel")
    elif mismatch == "threads":
        options["threads"] = 2
    elif mismatch == "activation":
        entry["latency"]["activation_bits"] = 8
        options["activation_bits"] = "32"
    else:
        entry["eligible_for_automatic_selection"] = False
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto", **options) == (32, 32, None)


@pytest.mark.parametrize("mutation", ["replace", "missing", "hash_missing", "external", "traversal"])
def test_auto_never_selects_unverified_or_external_weights(profile_files, mutation):
    case = profile_files
    artifact = case.candidate()
    entry = case.result["native"]["8"]
    if mutation == "replace":
        artifact.write_bytes(b"different weights")
    elif mutation == "missing":
        artifact.unlink()
    elif mutation == "hash_missing":
        entry.pop("artifact_sha256")
    else:
        external = case.directory.parent / "outside.leaf"
        external.write_bytes(artifact.read_bytes())
        entry["artifact"] = str(external) if mutation == "external" else "../outside.leaf"
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (32, 32, None)


def test_auto_tries_next_eligible_artifact_if_fastest_was_replaced(profile_files):
    case = profile_files
    invalid = case.candidate("8-smooth", 1.0, activations=8)
    valid = case.candidate("8", 3.0)
    invalid.write_bytes(b"replaced artifact")
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (8, 32, valid)


@pytest.mark.parametrize("quality", [False, None, 1, "true"])
def test_automatic_selection_requires_an_explicit_boolean_quality_pass(profile_files, quality):
    case = profile_files
    case.candidate()
    case.result["native"]["8"]["quality_gate_passed"] = quality
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (32, 32, None)


@pytest.mark.parametrize("stability", [None, {}, [], {"passed": False}, {"passed": None},
                                      {"passed": 1}, {"passed": "true"}])
def test_automatic_selection_requires_an_explicit_stability_pass(profile_files, stability):
    case = profile_files
    case.candidate()
    case.result["native"]["8"]["latency_stability"] = stability
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (32, 32, None)


def test_historical_profile_without_stability_is_not_automatically_selected(profile_files):
    case = profile_files
    case.candidate()
    case.result["native"]["8"].pop("latency_stability")
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (32, 32, None)


@pytest.mark.parametrize("target", ["eager", "sdpa", "leaf"])
def test_auto_requires_matching_tags_on_every_underlying_baseline(profile_files, target):
    case = profile_files
    case.candidate()
    latency = (case.result["native"]["32"]["latency"] if target == "leaf"
               else case.result["pytorch"]["latency"][target])
    latency.pop("latency_workload")
    case.save()
    original = (case.directory / "validation.json").read_bytes()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (32, 32, None)
    assert (case.directory / "validation.json").read_bytes() == original


@pytest.mark.parametrize("target", ["candidate", "leaf", "eager", "sdpa"])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_auto_recomputes_median_consistency_instead_of_trusting_stored_passes(profile_files, target, phase):
    case = profile_files
    case.candidate()
    timing = (case.result["native"]["8" if target == "candidate" else "32"]["latency"]
              if target in ("candidate", "leaf") else case.result["pytorch"]["latency"][target])
    timing[f"{phase}_p50_ms"] *= 2
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (32, 32, None)


@pytest.mark.parametrize("prefill,decode", [(103, 7), (90, 13)])
def test_auto_recomputes_actual_no_slowdown_even_when_stored_eligibility_is_true(profile_files, prefill, decode):
    case = profile_files
    case.candidate(latency=decode)
    case.result["native"]["8"]["latency"] = latency_record(prefill, decode)
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (32, 32, None)


@pytest.mark.parametrize("field,value", [("next_token_agreement", 0.94), ("perplexity_ratio", 1.021),
                                       ("next_token_agreement", True), ("perplexity_ratio", float("nan"))])
def test_auto_recomputes_quality_thresholds_without_trusting_stored_quality_true(profile_files, field, value):
    case = profile_files
    case.candidate()
    case.result["native"]["8"]["quality"][field] = value
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (32, 32, None)


def test_auto_recomputed_gate_does_not_rewrite_or_relabel_frozen_profile(profile_files):
    case = profile_files
    expected = case.candidate()
    case.save()
    original = (case.directory / "validation.json").read_bytes()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (8, 32, expected)
    assert (case.directory / "validation.json").read_bytes() == original


def test_auto_tries_next_stable_configuration(profile_files):
    case = profile_files
    case.candidate("8-smooth", 1.0, activations=8)
    stable = case.candidate("8", 5.0)
    case.result["native"]["8-smooth"]["latency_stability"]["passed"] = False
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (8, 32, stable)


@pytest.mark.parametrize("field,value", [
    ("weight_bits", 32), ("weight_bits", 16), ("activation_bits", 4),
    ("decode_p50_ms", float("nan")), ("decode_p50_ms", float("inf")),
    ("decode_p50_ms", 0), ("decode_p50_ms", -1), ("decode_p50_ms", "7"),
    ("decode_p50_ms", True),
])
def test_invalid_precision_or_latency_is_not_automatically_selected(profile_files, field, value):
    case = profile_files
    case.candidate()
    entry = case.result["native"]["8"]
    (entry if field == "weight_bits" else entry["latency"])[field] = value
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (32, 32, None)


@pytest.mark.parametrize("malformed", ["{", "[]", "null", '{"pytorch":null}', '{"native":[]}'])
def test_malformed_profile_safely_falls_back_to_fp32(profile_files, malformed):
    case = profile_files
    (case.directory / "validation.json").write_text(malformed)
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (32, 32, None)


@pytest.mark.parametrize("malformed", [None, [], {}, {"decode_p50_ms": 1}])
def test_malformed_candidate_latency_safely_falls_back_to_fp32(profile_files, malformed):
    case = profile_files
    case.candidate()
    case.result["native"]["8"]["latency"] = malformed
    case.save()
    assert cli.select_configuration(case.directory, case.runtime, "auto") == (32, 32, None)


def test_cache_root_can_be_kept_on_a_local_device(tmp_path, monkeypatch):
    path = tmp_path / "local model cache"
    monkeypatch.setenv("LEAF_CACHE_DIR", str(path))
    assert cli.cache_root() == path


def test_local_snapshot_and_alias_do_not_access_network(tmp_path, monkeypatch):
    snapshot = tmp_path / "weights"
    snapshot.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "aliases.json").write_text(json.dumps({"local-model": str(snapshot)}))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=lambda *args, **kwargs: pytest.fail("Local snapshots must not use the network")))
    assert cli.resolve_model(str(snapshot), cache) == snapshot.resolve()
    assert cli.resolve_model("local-model", cache, offline=True) == snapshot.resolve()


def test_repository_configuration_is_checked_before_large_weight_fetch(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "config.json").write_text('{"model_type":"unsupported-topology"}')
    requests = []

    def download(model, **options):
        requests.append((model, options))
        return str(metadata)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    with pytest.raises(ValueError, match="Supply --plan"):
        cli.resolve_model("organization/custom-model", tmp_path / "cache", offline=True)
    assert len(requests) == 1
    assert requests[0][1]["allow_patterns"] == ["config.json", "generation_config.json"]
    assert requests[0][1]["local_files_only"] is True


def test_custom_plan_allows_repository_without_model_id_branch(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "config.json").write_text('{"model_type":"independent-decoder"}')
    plan = tmp_path / "custom-plan.json"
    from tools.decoder_plan import make_plan
    custom_plan = make_plan({"model_type": "llama", "hidden_size": 16, "intermediate_size": 32,
                            "num_attention_heads": 2, "num_hidden_layers": 1, "vocab_size": 64})
    custom_plan["source_architecture"] = "independent-decoder"
    plan.write_text(json.dumps(custom_plan))
    requests = []

    def download(model, **options):
        requests.append(options)
        return str(metadata)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    assert cli.resolve_model("organization/custom-model", tmp_path / "cache", offline=True, plan=plan) == metadata
    assert len(requests) == 2
    assert all(request["local_files_only"] for request in requests)
    assert "*.safetensors" in requests[1]["allow_patterns"]


def test_model_cache_invalidates_on_config_weights_or_plan_changes(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config = snapshot / "config.json"
    config.write_text('{"model_type":"custom"}')
    weight = snapshot / "model.safetensors"
    weight.write_bytes(b"weights")
    plan = tmp_path / "plan.json"
    plan.write_text('{"config":1}')
    cache = tmp_path / "cache"
    original = cli.model_directory(snapshot, cache, plan)
    assert cli.model_directory(snapshot, cache, plan) == original
    config.write_text('{"model_type":"another"}')
    config_changed = cli.model_directory(snapshot, cache, plan)
    assert config_changed != original
    current = weight.stat()
    os.utime(weight, ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000))
    weights_changed = cli.model_directory(snapshot, cache, plan)
    assert weights_changed != config_changed
    plan.write_text('{"config":2}')
    assert cli.model_directory(snapshot, cache, plan) != weights_changed


def test_distinct_local_snapshots_cannot_reuse_each_others_weights(tmp_path):
    first, second = tmp_path / "model-a", tmp_path / "model-b"
    for snapshot, weights in [(first, b"weights-a"), (second, b"weights-b")]:
        snapshot.mkdir()
        (snapshot / "config.json").write_text('{"model_type":"custom"}')
        weight = snapshot / "model.safetensors"
        weight.write_bytes(weights)
        os.utime(weight, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
    cache = tmp_path / "cache"
    assert cli.model_directory(first, cache) != cli.model_directory(second, cache)


def test_configured_native_executable_is_used_without_compilation(tmp_path, monkeypatch):
    runtime = tmp_path / "native runtime.exe"
    runtime.write_bytes(b"native")
    monkeypatch.setenv("LEAF_DECODER_BIN", str(runtime))
    monkeypatch.setattr(cli, "source_engine", lambda: pytest.fail("Configured runtime must not build"))
    assert cli.executable(tmp_path / "cache") == runtime.resolve()


def test_missing_configured_executable_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAF_DECODER_BIN", str(tmp_path / "missing.exe"))
    with pytest.raises(FileNotFoundError, match="LEAF_DECODER_BIN"):
        cli.executable(tmp_path / "cache")


def test_native_build_is_reused_and_source_changes_invalidate_it(tmp_path, monkeypatch):
    monkeypatch.delenv("LEAF_DECODER_BIN", raising=False)
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "package" / "cli.py"))
    engine = tmp_path / "engine"
    for relative in ("src/decoder.cpp", "src/main_decoder.cpp", "src/kv_cache.cpp",
                     "src/kernels/transformer.cpp", "include/leaf/runtime/decoder.h"):
        path = engine / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    monkeypatch.setattr(cli, "source_engine", lambda: engine)
    monkeypatch.setattr(cli.platform, "system", lambda: "test-os")
    monkeypatch.setattr(cli.platform, "machine", lambda: "test-machine")
    monkeypatch.setattr(cli.shutil, "which", lambda name: "test-cxx")
    builds = []

    def compile_native(command, *, check):
        assert check
        builds.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"compiled runtime")

    monkeypatch.setattr(cli.subprocess, "run", compile_native)
    cache = tmp_path / "cache"
    first = cli.executable(cache)
    assert cli.executable(cache) == first
    assert len(builds) == 1
    assert "-mavx2" not in builds[0]  # Portable dispatch, not an AVX2-only package.
    (engine / "src/decoder.cpp").write_text("new native implementation")
    assert cli.executable(cache) != first
    assert len(builds) == 2


def test_prepared_model_artifact_is_reused(tmp_path, monkeypatch):
    from tools import export_decoder
    snapshot = tmp_path / "snapshot"
    directory = tmp_path / "model-cache"
    plan = tmp_path / "custom-plan.json"
    exports = []

    def export(model, output, bits, *, plan_path):
        exports.append((model, bits, plan_path))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"native artifact")
        return {"weight_bits": bits}

    monkeypatch.setattr(export_decoder, "export_decoder", export)
    artifact = cli.prepare(snapshot, directory, 8, plan)
    assert cli.prepare(snapshot, directory, 8, plan) == artifact
    assert exports == [(snapshot, 8, plan)]
    assert json.loads((directory / "decoder-8.json").read_text()) == {"weight_bits": 8}


@pytest.fixture
def tokenizer_snapshot(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.processors import TemplateProcessing
    vocabulary = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "user": 3, "answer": 4, "hello": 5, "world": 6}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.post_processor = TemplateProcessing(single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", 1), ("[EOS]", 2)])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    return tmp_path


def test_chat_template_is_rendered_without_duplicate_special_tokens(tokenizer_snapshot):
    config = {"bos_token": "[BOS]", "eos_token": "[EOS]",
              "chat_template": "{{ bos_token }} user {{ messages[0]['content'] }}{% if add_generation_prompt %} answer{% endif %}"}
    (tokenizer_snapshot / "tokenizer_config.json").write_text(json.dumps(config))
    previous = set(sys.modules)
    tokenizer, ids = cli.encode_prompt(tokenizer_snapshot, "hello world")
    assert ids == [1, 3, 5, 6, 4]
    assert tokenizer.decode([5, 6]) == "hello world"
    assert not any(name == "torch" or name.startswith("torch.") for name in set(sys.modules) - previous)


def test_raw_prompt_skips_chat_template_but_preserves_tokenizer_special_tokens(tokenizer_snapshot):
    (tokenizer_snapshot / "tokenizer_config.json").write_text(json.dumps({"chat_template": "answer"}))
    _, ids = cli.encode_prompt(tokenizer_snapshot, "hello world", raw=True)
    assert ids == [1, 5, 6, 2]


def test_default_named_chat_template_is_selected(tokenizer_snapshot):
    templates = [{"name": "tool_use", "template": "world"},
                 {"name": "default", "template": "user {{ messages[0]['content'] }} answer"}]
    (tokenizer_snapshot / "tokenizer_config.json").write_text(json.dumps({"chat_template": templates}))
    _, ids = cli.encode_prompt(tokenizer_snapshot, "hello")
    assert ids == [3, 5, 4]


def test_template_raise_exception_surfaces_a_clear_prompt_error(tokenizer_snapshot):
    (tokenizer_snapshot / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "{{ raise_exception('Invalid user prompt') }}"}))
    with pytest.raises(ValueError, match="Invalid user prompt"):
        cli.encode_prompt(tokenizer_snapshot, "hello")


def test_empty_formatted_prompt_is_rejected(tokenizer_snapshot):
    (tokenizer_snapshot / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ '' }}"}))
    with pytest.raises(ValueError, match="no tokens"):
        cli.encode_prompt(tokenizer_snapshot, "hello")


def test_missing_lightweight_tokenizer_fails_without_loading_pytorch(tmp_path):
    with pytest.raises(RuntimeError, match="tokenizer.json"):
        cli.encode_prompt(tmp_path, "hello")


def test_run_parser_preserves_general_model_plan_and_precision_options(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(cli, "run", captured.append)
    plan = tmp_path / "any-plan.json"
    monkeypatch.setattr(sys, "argv", ["leaf", "run", "custom/model", "--prompt", "hello",
        "--plan", str(plan), "--bits", "8", "--activation-bits", "8", "--threads", "2", "--offline"])
    assert cli.main() == 0
    args = captured[0]
    assert (args.model, args.plan, args.bits, args.activation_bits, args.threads, args.offline) == (
        "custom/model", plan, "8", "8", 2, True)


def test_optimize_parser_preserves_custom_dataset_and_calibration_flags(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(cli, "optimize", captured.append)
    dataset, calibration, plan = [tmp_path / name for name in ("test.jsonl", "train.csv", "plan.json")]
    monkeypatch.setattr(sys, "argv", ["leaf", "optimize", "custom/model", "--dataset", str(dataset),
        "--text-column", "document", "--dataset-source", "Locally held-out documents",
        "--calibration-dataset", str(calibration), "--plan", str(plan)])
    assert cli.main() == 0
    args = captured[0]
    assert args.dataset == dataset and args.calibration_dataset == calibration and args.plan == plan
    assert args.text_column == "document" and args.dataset_source == "Locally held-out documents"


@pytest.mark.parametrize("threads", ["-1", "0", "65"])
def test_invalid_optimize_threads_fail_cleanly_before_model_preparation(tmp_path, monkeypatch, capsys, threads):
    monkeypatch.setattr(cli, "optimize", lambda args: pytest.fail("Invalid thread count must not prepare a model"))
    monkeypatch.setattr(sys, "argv", ["leaf", "optimize", "custom/model", "--dataset",
                                    str(tmp_path / "held-out.txt"), "--threads", threads])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "threads must be 1..64" in capsys.readouterr().err


def test_optimize_forwards_general_flags_and_restores_process_arguments(tmp_path, monkeypatch):
    from tools import validate_decoder
    snapshot, directory, runtime = tmp_path / "snapshot", tmp_path / "cache", tmp_path / "runtime"
    monkeypatch.setattr(cli, "resolve_model", lambda *args: snapshot)
    monkeypatch.setattr(cli, "validate_model_generation", lambda *args: None)
    monkeypatch.setattr(cli, "model_directory", lambda *args: directory)
    monkeypatch.setattr(cli, "executable", lambda *args: runtime)
    commands, validations = [], []
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(validate_decoder, "main", lambda: validations.append(list(sys.argv)))
    args = SimpleNamespace(model="custom/model", offline=True, plan=tmp_path / "plan.json", threads=2,
        dataset=tmp_path / "test.jsonl", calibration_dataset=tmp_path / "train.csv", text_column="document",
        dataset_source="held-out local test", protected_int8=True, grouped_int8=True, grouped_int8_smooth=True)
    previous = sys.argv
    cli.optimize(args)
    assert sys.argv is previous
    calibration_command, validation_command = commands[0], validations[0]
    for command in (calibration_command, validation_command):
        assert command[command.index("--plan") + 1] == str(args.plan)
        assert command[command.index("--text-column") + 1] == "document"
        assert command[command.index("--threads") + 1] == "2"
    assert calibration_command[calibration_command.index("--dataset") + 1] == str(args.calibration_dataset)
    assert validation_command[validation_command.index("--dataset") + 1] == str(args.dataset)
    assert validation_command[validation_command.index("--dataset-source") + 1] == args.dataset_source
    assert validation_command[validation_command.index("--calibration") + 1] == str(directory / "calibration.npz")
    assert validation_command[validation_command.index("--runs") + 1] == "7"
    assert validation_command[validation_command.index("--warmup") + 1] == "3"
    assert "--protected-int8" in validation_command
    assert "--grouped-int8" in validation_command
    assert "--grouped-int8-smooth" in validation_command


@pytest.mark.parametrize("flag,value", [("--blocks", "0"), ("--sequence-length", "1"),
    ("--runs", "0"), ("--warmup", "-1"), ("--generate", "0")])
def test_optimize_rejects_invalid_dimensions_before_preparing_model(flag, value, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["leaf", "optimize", "model", "--dataset", "test.txt", flag, value])
    monkeypatch.setattr(cli, "optimize", lambda *args: pytest.fail("Invalid settings must not prepare a model"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_optimize_rejects_calibrating_on_the_held_out_dataset(tmp_path, monkeypatch):
    from tools import validate_decoder
    monkeypatch.setattr(cli, "resolve_model", lambda *args: tmp_path / "snapshot")
    monkeypatch.setattr(cli, "validate_model_generation", lambda *args: None)
    monkeypatch.setattr(cli, "model_directory", lambda *args: tmp_path / "cache")
    monkeypatch.setattr(cli, "executable", lambda *args: tmp_path / "runtime")
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: pytest.fail("Leaked calibration must not run"))
    monkeypatch.setattr(validate_decoder, "main", lambda: pytest.fail("Leaked validation must not run"))
    dataset = tmp_path / "test.jsonl"
    args = SimpleNamespace(model="custom/model", offline=True, plan=None, threads=1,
        dataset=dataset, calibration_dataset=dataset, text_column="text", dataset_source=None)
    previous = sys.argv
    with pytest.raises(ValueError, match="separate splits"):
        cli.optimize(args)
    assert sys.argv is previous


def test_alias_command_records_local_snapshot_without_importing_model(tmp_path, monkeypatch):
    cache, snapshot = tmp_path / "cache", tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setenv("LEAF_CACHE_DIR", str(cache))
    monkeypatch.setattr(sys, "argv", ["leaf", "alias", "my-local-model", str(snapshot)])
    assert cli.main() == 0
    assert json.loads((cache / "aliases.json").read_text()) == {"my-local-model": str(snapshot.resolve())}
