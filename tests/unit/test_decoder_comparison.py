"""Frozen provenance and conservative native-experiment gates; no native runs."""
from __future__ import annotations

import copy
from contextlib import nullcontext
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import benchmark_decoder_comparison as comparison


def latency(prefill=100.0, decode=10.0, count=7):
    return {"prefill_p50_ms": prefill, "decode_p50_ms": decode,
            "prefill_samples_ms": [prefill] * count, "decode_samples_ms": [decode] * count}


def test_diagnostic_build_cannot_qualify_timing():
    metrics = dict(latency(), threads=1, activation_bits=32, diagnostic_timing_build=True)
    with pytest.raises(ValueError, match="Diagnostic timing"):
        comparison.validate_timing_request(metrics, runs=7, threads=1, activation_bits=32)


def stage(prefill=100.0, decode=10.0):
    return comparison.stage_summary([latency(prefill, decode), latency(prefill, decode)])


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    for name in comparison.NATIVE_POLICY_NAMES:
        monkeypatch.delenv(name, raising=False)
    workdir = tmp_path / "work"
    workdir.mkdir()
    before, after = tmp_path / "before.exe", tmp_path / "after.exe"
    before.write_bytes(b"validated before decoder")
    after.write_bytes(b"candidate after decoder")
    reference = np.array([[3., 1., 0., 0.], [0., 3., 1., 0.], [0., 0., 3., 1.],
                          [3., 0., 0., 0.], [0., 3., 0., 0.], [0., 0., 3., 0.]], dtype=np.float32)
    np.save(workdir / "reference.npy", reference)
    data = {"sequences": [[0, 1, 2], [0, 1, 2]], "benchmark_ids": [0, 1, 2],
            "generation_ids": [0, 1], "generated_tokens": [2, 3]}
    (workdir / "tokens.json").write_text(json.dumps(data))
    cases = {}
    for key in ("32", "8-custom"):
        artifact = workdir / (key + ".leaf")
        artifact.write_bytes(("validated " + key).encode())
        cases[key] = {"weight_bits": 32 if key == "32" else 8, "activation_bits": 32,
                      "artifact": artifact.name, "artifact_sha256": comparison.digest(artifact),
                      "quality": {"evaluated_tokens": 4}, "quality_gate_passed": True,
                      "generated_tokens": [2, 3]}
    record = {"benchmark": "trained-full-decoder-quality-and-latency",
              "native_policy": dict(comparison.DEFAULT_NATIVE_POLICY),
              "native_executable_sha256": comparison.digest(before),
              "quality_measured_at_utc": "2026-10-01T00:00:00+00:00", "native": cases,
              "pytorch": {"threads": 1, "pinned_cpu": 2, "blocks": 2, "sequence_length": 3,
                          "platform": comparison.platform.platform(), "cpu": comparison.platform.processor(),
                          "generated_tokens": [2, 3], "quality_gate_thresholds": copy.deepcopy(comparison.GATES),
                          "quality_cache_sha256": {name: comparison.digest(workdir / name)
                                                   for name in ("reference.npy", "tokens.json")},
                          "latency": {"eager": latency(300, 30)}}}
    args = SimpleNamespace(before=before, after=after, workdir=workdir, record=tmp_path / "frozen.json",
                           keys=None, runs=7, warmup=3, output=tmp_path / "comparison.json")
    comparison.write_record(args.record, record)
    return SimpleNamespace(args=args, record=record, reference=reference, data=data)


def save_record(frozen):
    comparison.write_record(frozen.args.record, frozen.record)


def enable_scoped_tiles(frozen):
    frozen.args.after_float_tiles = True
    frozen.record["native_policy"] = dict(comparison.DEFAULT_NATIVE_POLICY)
    save_record(frozen)


def test_frozen_provenance_resolves_precisions_without_model_name(frozen):
    context = comparison.validate_inputs(frozen.args)
    assert context["keys"] == ["32", "8-custom"]
    assert context["threads"] == 1 and context["cpu"] == 2
    np.testing.assert_array_equal(context["reference"], frozen.reference)


@pytest.mark.parametrize("cpu", [0, 3])
def test_cpu_override_accepts_nonnegative_integers_without_changing_frozen_reference(frozen, cpu):
    frozen.args.cpu = cpu
    original_record = frozen.args.record.read_bytes()
    context = comparison.validate_inputs(frozen.args)
    assert context["cpu"] == cpu
    assert context["record"]["pytorch"]["pinned_cpu"] == 2
    assert frozen.args.record.read_bytes() == original_record


def test_absent_cpu_override_retains_frozen_pin(frozen):
    frozen.args.cpu = None
    assert comparison.validate_inputs(frozen.args)["cpu"] == 2


@pytest.mark.parametrize("cpu", [True, False, -1, 1.0, "1", [], {}])
def test_cpu_override_rejects_bool_negative_and_noninteger_values(frozen, cpu):
    frozen.args.cpu = cpu
    with pytest.raises(ValueError, match="CPU override must be a nonnegative integer"):
        comparison.validate_inputs(frozen.args)


def test_selected_keys_use_only_matching_frozen_artifacts(frozen):
    frozen.args.keys = ["8-custom"]
    context = comparison.validate_inputs(frozen.args)
    assert context["keys"] == ["8-custom"]
    assert set(context["artifacts"]) == {"8-custom"}


@pytest.mark.parametrize("keys", [[], ["32", "32"], ["missing"]])
def test_invalid_candidate_keys_are_rejected(frozen, keys):
    frozen.args.keys = keys
    with pytest.raises(ValueError, match="candidate|unique"):
        comparison.validate_inputs(frozen.args)


@pytest.mark.parametrize("changed", ["before", "reference.npy", "tokens.json", "32.leaf", "8-custom.leaf"])
def test_changed_frozen_inputs_are_rejected_before_execution(frozen, changed):
    path = frozen.args.before if changed == "before" else frozen.args.workdir / changed
    path.write_bytes(b"changed content")
    with pytest.raises(ValueError, match="executable|cache hashes|Artifact"):
        comparison.validate_inputs(frozen.args)


@pytest.mark.parametrize("hashes", [None, {}, {"reference.npy": "missing tokens"},
                                    {"reference.npy": "x", "tokens.json": "y", "unexpected": "z"}])
def test_frozen_reference_and_token_hashes_are_mandatory(frozen, hashes):
    frozen.record["pytorch"]["quality_cache_sha256"] = hashes
    save_record(frozen)
    with pytest.raises(ValueError, match="quality-cache hashes"):
        comparison.validate_inputs(frozen.args)


@pytest.mark.parametrize("field,value", [("threads", 0), ("threads", True), ("threads", "1"),
                                        ("pinned_cpu", -1), ("pinned_cpu", True), ("pinned_cpu", "2")])
def test_frozen_cpu_settings_are_validated(frozen, field, value):
    frozen.record["pytorch"][field] = value
    save_record(frozen)
    with pytest.raises(ValueError, match="threads|CPU pin"):
        comparison.validate_inputs(frozen.args)


@pytest.mark.parametrize("field,value", [("blocks", True), ("blocks", 0), ("sequence_length", 1),
                                        ("sequence_length", "3")])
def test_frozen_evaluation_dimensions_are_validated(frozen, field, value):
    frozen.record["pytorch"][field] = value
    save_record(frozen)
    with pytest.raises(ValueError, match="invalid " + field):
        comparison.validate_inputs(frozen.args)


def test_old_quality_thresholds_cannot_authorize_experiment(frozen):
    frozen.record["pytorch"]["quality_gate_thresholds"] = {}
    save_record(frozen)
    with pytest.raises(ValueError, match="thresholds differ"):
        comparison.validate_inputs(frozen.args)


@pytest.mark.parametrize("changed", ["platform", "cpu"])
def test_frozen_os_and_cpu_description_must_match_current_host(frozen, changed):
    frozen.record["pytorch"][changed] = "different host"
    save_record(frozen)
    with pytest.raises(ValueError, match="Current OS/CPU description differs"):
        comparison.validate_inputs(frozen.args)


@pytest.mark.parametrize("disabled", ["1", "0", "true"])
def test_changed_isa_policy_is_rejected_before_execution(frozen, monkeypatch, disabled):
    monkeypatch.setenv("LEAF_DISABLE_VNNI", disabled)
    with pytest.raises(ValueError, match="Unset LEAF_DISABLE_VNNI"):
        comparison.validate_inputs(frozen.args)


@pytest.mark.parametrize("flag", [*comparison.NATIVE_POLICY_NAMES, "LEAF_EXPERIMENTAL_FUTURE_KERNEL"])
@pytest.mark.parametrize("value", ["1", "0", "false"])
def test_scoped_tiles_rejects_conflicting_parent_native_flags_before_execution(frozen, monkeypatch, flag, value):
    enable_scoped_tiles(frozen)
    monkeypatch.setenv(flag, value)
    monkeypatch.setattr(comparison, "run_native", lambda *args, **kwargs: pytest.fail("policy conflict reached child"))
    monkeypatch.setattr(comparison, "pin_cpu", lambda *args: pytest.fail("policy conflict changed affinity"))
    original = dict(comparison.os.environ)
    with pytest.raises(ValueError, match="conflicts with parent native policy.*" + flag):
        comparison.compare(frozen.args)
    assert dict(comparison.os.environ) == original
    assert not frozen.args.output.exists()


@pytest.mark.parametrize("policy", [None, {}, [], False, 0,
    {"LEAF_DISABLE_VNNI": False},
    {**comparison.DEFAULT_NATIVE_POLICY, "unknown": False},
    {**comparison.DEFAULT_NATIVE_POLICY, "LEAF_EXPERIMENTAL_FLOAT_TILES": True},
    {**comparison.DEFAULT_NATIVE_POLICY, "LEAF_EXPERIMENTAL_FLOAT_GEMV": True},
    {**comparison.DEFAULT_NATIVE_POLICY, "LEAF_EXPERIMENTAL_FLOAT_TILES": 0}])
def test_scoped_tiles_requires_explicit_default_before_quality_policy(frozen, policy):
    enable_scoped_tiles(frozen)
    frozen.record["native_policy"] = policy
    save_record(frozen)
    with pytest.raises(ValueError, match="explicitly default frozen BEFORE native policy"):
        comparison.validate_inputs(frozen.args)


def test_scoped_tiles_cannot_relabel_legacy_missing_before_policy(frozen):
    frozen.args.after_float_tiles = True
    del frozen.record["native_policy"]
    save_record(frozen)
    assert "native_policy" not in frozen.record
    with pytest.raises(ValueError, match="explicitly default frozen BEFORE native policy"):
        comparison.validate_inputs(frozen.args)


def test_external_artifact_path_is_rejected_even_when_hash_matches(frozen):
    external = frozen.args.workdir.parent / "external.leaf"
    external.write_bytes(b"external weights")
    frozen.record["native"]["32"].update(artifact="../external.leaf", artifact_sha256=comparison.digest(external))
    save_record(frozen)
    with pytest.raises(ValueError, match="Artifact 32 is external"):
        comparison.validate_inputs(frozen.args)


def test_evaluated_token_count_cannot_silently_change(frozen):
    frozen.record["native"]["32"]["quality"]["evaluated_tokens"] = 6
    save_record(frozen)
    with pytest.raises(ValueError, match="trained-quality provenance"):
        comparison.validate_inputs(frozen.args)


def test_stage_summary_preserves_every_raw_sample_and_both_passes():
    first, second = latency(100, 10), latency(105, 10.5)
    summary = comparison.stage_summary([first, second])
    assert summary["stability_passed"] is True
    assert summary["prefill_samples_ms"] == first["prefill_samples_ms"] + second["prefill_samples_ms"]
    assert summary["prefill_p50_ms"] == 102.5
    assert summary["pass_medians_ms"]["prefill"] == [100, 105]
    assert len(summary["pass_stability"]) == 2


def test_stage_summary_rejects_unstable_individual_pass_without_filtering():
    first, second = latency(), latency()
    second["decode_samples_ms"][-1] = 100
    summary = comparison.stage_summary([first, second])
    assert summary["stability_passed"] is False
    assert summary["decode_samples_ms"][-1] == 100
    assert summary["pass_stability"][1]["decode"]["passed"] is False


def test_stage_summary_rejects_between_pass_drift_even_if_each_pass_is_stable():
    summary = comparison.stage_summary([latency(100, 10), latency(200, 20)])
    assert all(item["passed"] for item in summary["pass_stability"])
    assert summary["stability_passed"] is False
    assert summary["between_pass_stability"]["prefill"]["max_min_pass_median_ratio"] == 2


def test_pooled_samples_do_not_hide_short_individual_passes():
    summary = comparison.stage_summary([latency(count=4), latency(count=4)])
    assert summary["stability"]["passed"] is True
    assert summary["stability_passed"] is False


@pytest.mark.parametrize("prefill,decode,accepted", [(98, 10.2, True), (98.00001, 10, False),
                                                    (97, 10.20001, False), (99, 9, False), (97, 9, True)])
def test_native_gate_requires_two_percent_prefill_gain_and_no_phase_slowdown(prefill, decode, accepted):
    gate = comparison.native_experiment_gate(stage(), stage(prefill, decode), True)
    assert gate["accepted_native_experiment"] is accepted
    assert gate["automatic_selection_authorized"] is False
    assert gate["fresh_pytorch_comparison"] is False


@pytest.mark.parametrize("quality", [False, None, 1, "true"])
def test_native_gate_requires_explicit_boolean_quality_pass(quality):
    assert comparison.native_experiment_gate(stage(), stage(90, 9), quality)["accepted_native_experiment"] is False


@pytest.mark.parametrize("unstable", ["before", "after"])
def test_native_gate_requires_both_stages_stable(unstable):
    before, after = stage(), stage(90, 9)
    (before if unstable == "before" else after)["stability_passed"] = False
    assert comparison.native_experiment_gate(before, after, True)["accepted_native_experiment"] is False


@pytest.mark.parametrize("invalid", [None, 0, float("nan"), float("inf"), True])
def test_native_gate_rejects_invalid_medians(invalid):
    after = stage(90, 9)
    after["decode_p50_ms"] = invalid
    assert comparison.native_experiment_gate(stage(), after, True)["accepted_native_experiment"] is False


def test_native_gate_extreme_ratio_fails_with_json_safe_fields():
    before, after = stage(), stage()
    before["decode_p50_ms"], after["decode_p50_ms"] = 5e-324, 1e308
    gate = comparison.native_experiment_gate(before, after, True)
    assert gate["accepted_native_experiment"] is False
    assert gate["decode_ratio_after_before"] is None
    json.dumps(gate, allow_nan=False)


def mock_native(frozen, *, changed_generation=False):
    calls = []

    def execute(executable, artifact, sequences, **options):
        calls.append((executable, artifact.name, options))
        mode = options.get("mode", "verify")
        if mode == "generate":
            return np.zeros(0, dtype=np.float32), {"generated_tokens": [1, 2] if changed_generation else [2, 3]}
        if mode == "bench":
            measured = latency(100, 10) if executable == frozen.args.before else latency(90, 9)
            measured.update(threads=options["threads"], activation_bits=options["activation_bits"])
            return np.zeros(0, dtype=np.float32), measured
        expected = frozen.reference if mode == "verify" else frozen.reference[:3]
        return expected.reshape(-1).copy(), {"mode": mode}

    return execute, calls


def test_after_fp32_requires_exact_generation_but_quantized_case_reports_difference(frozen, monkeypatch):
    context = comparison.validate_inputs(frozen.args)
    execute, _ = mock_native(frozen, changed_generation=True)
    monkeypatch.setattr(comparison, "run_native", execute)
    fp32 = comparison.verify_after(frozen.args, context, "32")
    quantized = comparison.verify_after(frozen.args, context, "8-custom")
    assert fp32["trained_quality_gate_passed"] is True
    assert fp32["quality_gate_passed"] is False
    assert quantized["quality_gate_passed"] is True
    assert quantized["generation_exact_match_pytorch"] is False
    assert quantized["generation_exact_match_before"] is False


def test_single_chunk_quality_gate_is_diagnostic_not_full_quality_qualification(frozen, monkeypatch):
    context = comparison.validate_inputs(frozen.args)
    execute, _ = mock_native(frozen)
    monkeypatch.setattr(comparison, "run_native", execute)
    measured = iter([{"next_token_agreement": 0.97, "perplexity_ratio": 1.01},
                     {"next_token_agreement": 0.8, "perplexity_ratio": 1.04}])
    monkeypatch.setattr(comparison, "quality", lambda *args: next(measured))
    result = comparison.verify_after(frozen.args, context, "8-custom")
    assert result["trained_quality_gate_passed"] is True
    assert result["chunked"]["quality_gate_passed"] is False
    assert result["chunked"]["parity_vs_full"] is True
    assert result["quality_gate_passed"] is True


def test_chunked_vs_full_parity_is_required_for_quantized_experiment(frozen, monkeypatch):
    context = comparison.validate_inputs(frozen.args)
    execute, _ = mock_native(frozen)

    def incorrect_chunk(*args, **options):
        actual, metrics = execute(*args, **options)
        if options.get("mode") == "chunked":
            actual = actual + 0.1
        return actual, metrics

    monkeypatch.setattr(comparison, "run_native", incorrect_chunk)
    result = comparison.verify_after(frozen.args, context, "8-custom")
    assert result["trained_quality_gate_passed"] is True
    assert result["chunked"]["parity_vs_full"] is False
    assert result["quality_gate_passed"] is False


def test_comparison_is_serial_abba_disables_profiling_and_never_promotes_auto(frozen, monkeypatch):
    execute, calls = mock_native(frozen)
    monkeypatch.setattr(comparison, "run_native", execute)
    pins = []
    monkeypatch.setattr(comparison, "pin_cpu", pins.append)
    monkeypatch.setattr(comparison, "keep_awake", nullcontext)
    monkeypatch.setenv("LEAF_DECODER_PROFILE", "1")
    original = execute

    def unprofiled(*args, **kwargs):
        assert "LEAF_DECODER_PROFILE" not in comparison.os.environ
        return original(*args, **kwargs)

    monkeypatch.setattr(comparison, "run_native", unprofiled)
    result = comparison.compare(frozen.args)
    assert pins == [2]
    assert comparison.os.environ["LEAF_DECODER_PROFILE"] == "1"
    assert result["automatic_selection_authorized"] is False
    assert result["profiling_enabled"] is False
    assert result["frozen_pytorch_context"] == frozen.record["pytorch"]
    for key in result["native"]:
        bench_calls = [item for item in calls if item[1] == key + ".leaf" and item[2].get("mode") == "bench"]
        assert [item[0] for item in bench_calls] == [frozen.args.before, frozen.args.after, frozen.args.after, frozen.args.before]
        assert all(item[2]["runs"] == 7 and item[2]["warmup"] == 3 for item in bench_calls)
        assert result["native"][key]["gate"]["accepted_native_experiment"] is True
        assert result["native"][key]["timing"]["order"] == ["before", "after", "after", "before"]


def test_scoped_tiles_changes_only_after_quality_and_abba_children_and_restores_parent(frozen, monkeypatch):
    enable_scoped_tiles(frozen)
    execute, calls = mock_native(frozen)
    observed = []
    for name in comparison.NATIVE_POLICY_NAMES:
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("LEAF_DECODER_PROFILE", "parent profiling policy")
    monkeypatch.setenv("LEAF_UNRELATED_SETTING", "retained")
    original = dict(comparison.os.environ)

    def scoped_execute(executable, *args, **kwargs):
        expected = dict(comparison.DEFAULT_NATIVE_POLICY)
        expected["LEAF_EXPERIMENTAL_FLOAT_TILES"] = executable == frozen.args.after
        assert comparison.native_policy() == expected
        assert comparison.os.environ.get("LEAF_EXPERIMENTAL_FLOAT_TILES") == (
            "1" if executable == frozen.args.after else None)
        assert "LEAF_EXPERIMENTAL_FLOAT_GEMV" not in comparison.os.environ
        assert "LEAF_DISABLE_VNNI" not in comparison.os.environ
        assert "LEAF_DECODER_PROFILE" not in comparison.os.environ
        observed.append((executable, kwargs.get("mode", "verify"), comparison.native_policy()))
        return execute(executable, *args, **kwargs)

    monkeypatch.setattr(comparison, "run_native", scoped_execute)
    monkeypatch.setattr(comparison, "pin_cpu", lambda cpu: None)
    monkeypatch.setattr(comparison, "keep_awake", nullcontext)
    record_before = frozen.args.record.read_bytes()
    result = comparison.compare(frozen.args)
    assert dict(comparison.os.environ) == original
    assert frozen.args.record.read_bytes() == record_before
    after_policy = {**comparison.DEFAULT_NATIVE_POLICY, "LEAF_EXPERIMENTAL_FLOAT_TILES": True}
    assert result["after_float_tiles"] is True
    assert result["native_policy_by_stage"] == {"before": comparison.DEFAULT_NATIVE_POLICY, "after": after_policy}
    assert result["automatic_selection_authorized"] is False
    assert result["experimental_policy"] == {"LEAF_EXPERIMENTAL_FLOAT_TILES": "",
                                              "LEAF_EXPERIMENTAL_FLOAT_GEMV": ""}
    for key, case in result["native"].items():
        assert case["native_policy"] == after_policy
        assert case["gate"]["accepted_native_experiment"] is True
        for timing_pass in case["timing"]["passes"]:
            assert timing_pass["native_policy"] == result["native_policy_by_stage"][timing_pass["stage"]]
        native_calls = [item for item in calls if item[1] == key + ".leaf"]
        assert [call[0] for call in native_calls[:3]] == [frozen.args.after] * 3
    assert len(observed) == 14


@pytest.mark.parametrize("failure", ["verify", "chunked", "generate", "before_bench", "after_bench"])
def test_scoped_tiles_restores_parent_environment_when_any_child_fails(frozen, monkeypatch, failure):
    enable_scoped_tiles(frozen)
    execute, _ = mock_native(frozen)
    monkeypatch.setenv("LEAF_EXPERIMENTAL_FLOAT_GEMV", "")
    monkeypatch.setenv("LEAF_DECODER_PROFILE", "parent profiling policy")
    original = dict(comparison.os.environ)

    def fail(executable, *args, **kwargs):
        mode = kwargs.get("mode", "verify")
        actual_stage = "before" if executable == frozen.args.before else "after"
        if (mode if mode != "bench" else actual_stage + "_bench") == failure:
            assert comparison.native_policy()["LEAF_EXPERIMENTAL_FLOAT_TILES"] == (actual_stage == "after")
            raise RuntimeError("owned child failed")
        return execute(executable, *args, **kwargs)

    monkeypatch.setattr(comparison, "run_native", fail)
    monkeypatch.setattr(comparison, "pin_cpu", lambda cpu: None)
    monkeypatch.setattr(comparison, "keep_awake", nullcontext)
    with pytest.raises(RuntimeError, match="owned child failed"):
        comparison.compare(frozen.args)
    assert dict(comparison.os.environ) == original
    assert not frozen.args.output.exists()


def test_existing_global_experimental_policy_is_unchanged_without_scoped_option(frozen, monkeypatch):
    execute, _ = mock_native(frozen)
    monkeypatch.setenv("LEAF_EXPERIMENTAL_FLOAT_TILES", "global tiles policy")
    monkeypatch.setenv("LEAF_EXPERIMENTAL_FLOAT_GEMV", "global gemv policy")
    expected = {**comparison.DEFAULT_NATIVE_POLICY, "LEAF_EXPERIMENTAL_FLOAT_TILES": True,
                "LEAF_EXPERIMENTAL_FLOAT_GEMV": True}
    frozen.record["native_policy"] = expected
    save_record(frozen)
    original = dict(comparison.os.environ)

    def global_execute(*args, **kwargs):
        assert comparison.native_policy() == expected
        return execute(*args, **kwargs)

    monkeypatch.setattr(comparison, "run_native", global_execute)
    monkeypatch.setattr(comparison, "pin_cpu", lambda cpu: None)
    monkeypatch.setattr(comparison, "keep_awake", nullcontext)
    result = comparison.compare(frozen.args)
    assert dict(comparison.os.environ) == original
    assert result["after_float_tiles"] is False
    assert result["native_policy_by_stage"] == {"before": expected, "after": expected}
    assert result["experimental_policy"] == {"LEAF_EXPERIMENTAL_FLOAT_TILES": "global tiles policy",
                                              "LEAF_EXPERIMENTAL_FLOAT_GEMV": "global gemv policy"}
    for case in result["native"].values():
        assert case["native_policy"] == expected
        assert all(item["native_policy"] == expected for item in case["timing"]["passes"])


def test_cpu_override_is_applied_to_both_stages_and_recorded_separately_from_frozen_pin(frozen, monkeypatch):
    frozen.args.cpu = 3
    frozen.args.keys = ["32"]
    execute, calls = mock_native(frozen)
    pins = []
    monkeypatch.setattr(comparison, "run_native", execute)
    monkeypatch.setattr(comparison, "pin_cpu", pins.append)
    monkeypatch.setattr(comparison, "keep_awake", nullcontext)
    result = comparison.compare(frozen.args)
    assert pins == [3]
    assert result["pinned_cpu"] == 3
    assert result["frozen_reference_pinned_cpu"] == 2
    assert result["frozen_pytorch_context"]["pinned_cpu"] == 2
    assert result["automatic_selection_authorized"] is False
    assert [item[0] for item in calls if item[2].get("mode") == "bench"] == [
        frozen.args.before, frozen.args.after, frozen.args.after, frozen.args.before]


def test_cli_cpu_override_is_parsed_as_integer(frozen, monkeypatch):
    received = []
    monkeypatch.setattr(comparison.sys, "argv", ["benchmark_decoder_comparison.py", "--before", str(frozen.args.before),
                                                "--after", str(frozen.args.after), "--workdir", str(frozen.args.workdir),
                                                "--record", str(frozen.args.record), "--cpu", "0"])
    monkeypatch.setattr(comparison, "compare", lambda args: received.append(args.cpu) or {"native": {}})
    comparison.main()
    assert received == [0]


def test_cli_scoped_after_float_tiles_option_is_parsed(frozen, monkeypatch):
    received = []
    monkeypatch.setattr(comparison.sys, "argv", ["benchmark_decoder_comparison.py", "--before", str(frozen.args.before),
                                                "--after", str(frozen.args.after), "--workdir", str(frozen.args.workdir),
                                                "--record", str(frozen.args.record), "--after-float-tiles"])
    monkeypatch.setattr(comparison, "compare", lambda args: received.append(args.after_float_tiles) or {"native": {}})
    comparison.main()
    assert received == [True]


def test_profiling_environment_is_restored_on_failure(monkeypatch):
    monkeypatch.setenv("LEAF_DECODER_PROFILE", "existing profiling policy")
    with pytest.raises(RuntimeError):
        with comparison.profiling_disabled():
            assert "LEAF_DECODER_PROFILE" not in comparison.os.environ
            raise RuntimeError("child failed")
    assert comparison.os.environ["LEAF_DECODER_PROFILE"] == "existing profiling policy"


def test_comparison_cannot_overwrite_frozen_record(frozen):
    frozen.args.output = frozen.args.record
    with pytest.raises(ValueError, match="must not overwrite"):
        comparison.compare(frozen.args)


@pytest.mark.parametrize("runs,warmup", [(4, 3), (7, -1)])
def test_comparison_requires_repeated_samples_before_any_execution(frozen, runs, warmup):
    frozen.args.runs, frozen.args.warmup = runs, warmup
    with pytest.raises(ValueError, match="at least 5 runs"):
        comparison.compare(frozen.args)


@pytest.mark.parametrize("policy", [None, {}, {**comparison.DEFAULT_NATIVE_POLICY,
    "LEAF_EXPERIMENTAL_FLOAT_TILES": True}, {**comparison.DEFAULT_NATIVE_POLICY, "LEAF_DISABLE_VNNI": 0}])
def test_unscoped_comparison_cannot_reuse_quality_under_different_policy(frozen, policy):
    frozen.record["native_policy"] = policy
    save_record(frozen)
    with pytest.raises(ValueError, match="explicit frozen BEFORE quality policy"):
        comparison.validate_inputs(frozen.args)


@pytest.mark.parametrize("stage_name", ["before", "after"])
@pytest.mark.parametrize("field,value", [("threads", 2), ("threads", True), ("activation_bits", 8),
    ("prefill_samples_ms", [100] * 5), ("decode_samples_ms", [10] * 8)])
def test_comparison_rejects_mismatched_child_workload(frozen, monkeypatch, stage_name, field, value):
    execute, _ = mock_native(frozen)
    def wrong(executable, *args, **options):
        output, measured = execute(executable, *args, **options)
        stage = "before" if executable == frozen.args.before else "after"
        if options.get("mode") == "bench" and stage == stage_name:
            measured[field] = value
        return output, measured
    monkeypatch.setattr(comparison, "run_native", wrong)
    monkeypatch.setattr(comparison, "pin_cpu", lambda cpu: None)
    monkeypatch.setattr(comparison, "keep_awake", nullcontext)
    with pytest.raises(ValueError, match="differs from requested"):
        comparison.compare(frozen.args)
    assert not frozen.args.output.exists()


@pytest.mark.parametrize("prefill,decode,accepted", [(100, 98, True), (102, 98, True),
    (102.0001, 97, False), (98, 98.0001, False), (100, 100, False)])
def test_decode_objective_preserves_thresholds(prefill, decode, accepted):
    gate = comparison.native_experiment_gate(stage(100, 100), stage(prefill, decode), True, "decode")
    assert gate["accepted_native_experiment"] is accepted
    assert gate["required_decode_ratio"] == .98
    assert "required_prefill_ratio" not in gate


def test_invalid_objective_rejected():
    with pytest.raises(ValueError, match="objective"):
        comparison.native_experiment_gate(stage(), stage(), True, "fastest")
