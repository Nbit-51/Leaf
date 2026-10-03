import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.decoder_validation import (benchmark_stability, comparison_stability,
                                      latency_median, latency_stability, quality)
from tools.validate_decoder import LATENCY_WORKLOAD, apply_speed_gates, digest, native


def latency(prefill=100.0, decode=10.0):
    return {"prefill_p50_ms": prefill, "decode_p50_ms": decode,
            "prefill_samples_ms": [prefill] * 5, "decode_samples_ms": [decode] * 5,
            "latency_workload": dict(LATENCY_WORKLOAD)}


def speed_comparison(prefill=100.0, decode=9.0):
    return ({"32": {"latency": latency(120.0, 12.0), "quality_gate_passed": True},
             "8": {"latency": latency(prefill, decode), "quality_gate_passed": True}},
            {"latency": {"eager": latency(), "sdpa": latency(105.0, 11.0)},
             "latency_workload": dict(LATENCY_WORKLOAD)})


@pytest.mark.parametrize("samples", [None, {}, "1,2,3,4,5", 10,
    [[1], [2], [3, 4]], [[1, 2], [3, 4]], [1, 2, None, 4, 5],
    [1, 2, "3", 4, 5], [1, 2, True, 4, 5], [1, 2, np.bool_(False), 4, 5],
    [1, 2, complex(3), 4, 5], [1, 2, float("nan"), 4, 5],
    [1, 2, float("inf"), 4, 5], [1, 2, 0, 4, 5], [1, 2, -1, 4, 5],
    [1, 2, 10 ** 400, 4, 5], np.array(1.0), np.ones((2, 3)),
    np.array([1, [2, 3]], dtype=object)])
def test_invalid_samples_fail_closed_without_raising(samples):
    result = latency_stability(samples)
    assert result["passed"] is False
    assert result["reason"] in {"missing_or_invalid_samples", "samples_must_be_finite_positive_numbers"}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("samples,count", [([], 0), ([10] * 4, 4)])
def test_insufficient_timing_samples_never_qualify(samples, count):
    result = latency_stability(samples)
    assert result["passed"] is False
    assert result["sample_count"] == count
    if count:
        assert result["reason"] == "insufficient_samples"


@pytest.mark.parametrize("samples", [[10] * 5, (10,) * 7, np.full(5, 10, dtype=np.int32)])
def test_stable_samples_use_all_values(samples):
    result = latency_stability(samples)
    assert result["passed"] is True
    assert result["sample_count"] == len(samples)
    assert result["p10_ms"] == result["p90_ms"] == 10
    assert result["p90_p10_ratio"] == 1


def test_spread_threshold_is_inclusive_and_percentiles_are_linear():
    assert latency_stability([10, 10, 10, 12.5, 12.5])["passed"] is True
    result = latency_stability([10, 10, 10, 12.50001, 12.50001])
    assert result["passed"] is False
    assert result["reason"] == "timing_spread_exceeds_threshold"
    result = latency_stability([10, 20, 30, 40, 50])
    assert result["p10_ms"] == 14
    assert result["p90_ms"] == 46
    assert result["p90_p10_ratio"] == pytest.approx(46 / 14)


def test_timing_outlier_is_not_filtered_or_mutated():
    samples = [10, 10, 10, 10, 100]
    before = samples.copy()
    result = latency_stability(samples)
    assert result["passed"] is False
    assert result["p90_ms"] == pytest.approx(64)
    assert samples == before


def test_extreme_spread_has_json_safe_failure_record():
    result = latency_stability([5e-324, 5e-324, 1, 1e308, 1e308])
    assert result["passed"] is False
    assert result["p90_p10_ratio"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("malformed", [None, [], {}, {"prefill_samples_ms": [1] * 5}])
def test_both_timing_phases_are_required(malformed):
    assert benchmark_stability(malformed)["passed"] is False


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_stable_raw_samples_cannot_bless_an_inconsistent_declared_median(phase):
    measured = latency()
    measured[f"{phase}_p50_ms"] *= 0.1
    before = copy.deepcopy(measured)
    result = benchmark_stability(measured)
    assert result["passed"] is False
    assert result[phase]["median_consistent"] is False
    assert result[phase]["reason"] == "reported_median_does_not_match_samples"
    assert measured == before


def test_median_consistency_tolerates_native_six_digit_json_rounding():
    values = [100.1004999] * 3 + [100.1014999] * 3
    measured = latency()
    measured["prefill_samples_ms"] = [float(format(value, ".6g")) for value in values]
    measured["prefill_p50_ms"] = float(format(float(np.median(values)), ".6g"))
    assert benchmark_stability(measured)["passed"] is True


@pytest.mark.parametrize("target", ["candidate", "leaf", "eager", "sdpa"])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_speed_gates_reject_inconsistent_medians_in_any_baseline_or_candidate(target, phase):
    cases, baseline = speed_comparison()
    measured = (cases["8" if target == "candidate" else "32"]["latency"]
                if target in ("candidate", "leaf") else baseline["latency"][target])
    measured[f"{phase}_p50_ms"] *= 2
    apply_speed_gates(cases, baseline)
    assert cases["8"]["eligible_for_automatic_selection"] is False
    assert cases["8"]["latency_stability"]["passed"] is False


@pytest.mark.parametrize("value", [None, "10", True, np.bool_(True), 0, -1,
                                   float("nan"), float("inf"), 10 ** 400])
def test_invalid_medians_fail_closed(value):
    assert latency_median({"decode_p50_ms": value}, "decode") is None


def test_numpy_numeric_median_is_supported():
    assert latency_median({"decode_p50_ms": np.float32(2)}, "decode") == 2


def test_fastest_pytorch_path_is_checked_independently_for_each_phase():
    baselines = {"eager": latency(100, 12), "sdpa": latency(110, 10)}
    result = comparison_stability(latency(90, 8), latency(120, 15), baselines)
    assert result["passed"] is True
    assert set(result["fastest_pytorch"]["prefill"]["implementations"]) == {"eager"}
    assert set(result["fastest_pytorch"]["decode"]["implementations"]) == {"sdpa"}


def test_all_tied_fastest_pytorch_paths_must_be_stable():
    baselines = {"eager": latency(), "sdpa": latency()}
    baselines["sdpa"]["decode_samples_ms"][-1] = 100
    result = comparison_stability(latency(90, 8), latency(120, 15), baselines)
    assert result["passed"] is False
    assert set(result["fastest_pytorch"]["decode"]["implementations"]) == {"eager", "sdpa"}


def test_slower_unselected_pytorch_phase_does_not_reject_stable_fastest_path():
    baselines = {"eager": latency(), "sdpa": latency(110, 12)}
    baselines["sdpa"]["prefill_samples_ms"][-1] = 1000
    baselines["sdpa"]["decode_samples_ms"][-1] = 1000
    assert comparison_stability(latency(90, 8), latency(120, 15), baselines)["passed"] is True


@pytest.mark.parametrize("baselines", [None, [], {}, {"eager": None},
    {"eager": latency(), "sdpa": {"prefill_p50_ms": float("nan"), "decode_p50_ms": 11}}])
def test_missing_or_invalid_pytorch_baselines_cannot_qualify(baselines):
    assert comparison_stability(latency(90, 8), latency(120, 15), baselines)["passed"] is False


@pytest.mark.parametrize("baseline", [None, {}, [], latency(120, 15) | {"decode_samples_ms": [15] * 4}])
def test_leaf_fp32_repeatability_is_required(baseline):
    assert comparison_stability(latency(90, 8), baseline, {"eager": latency()})["passed"] is False


def test_speed_gates_accept_only_validated_stable_improvement_and_preserve_samples():
    cases, baseline = speed_comparison()
    samples = copy.deepcopy(cases["8"]["latency"])
    apply_speed_gates(cases, baseline)
    assert cases["8"]["eligible_for_automatic_selection"] is True
    assert cases["8"]["latency_stability"]["passed"] is True
    assert cases["8"]["latency_workload_match"]["passed"] is True
    assert cases["32"]["eligible_for_automatic_selection"] is False
    assert cases["8"]["decode_speedup_vs_fastest_pytorch"] == pytest.approx(10 / 9)
    assert cases["8"]["latency"]["decode_samples_ms"] == samples["decode_samples_ms"]
    assert cases["8"]["latency"]["prefill_samples_ms"] == samples["prefill_samples_ms"]


@pytest.mark.parametrize("target", ["baseline", "eager", "sdpa", "leaf", "candidate"])
@pytest.mark.parametrize("change", ["missing", "full_logits", "malformed_cache"])
def test_speed_gates_preserve_legacy_measurements_without_qualifying_unmatched_workload(target, change):
    cases, baseline = speed_comparison()
    selected = (baseline if target == "baseline" else baseline["latency"][target]
                if target in ("eager", "sdpa") else cases["32" if target == "leaf" else "8"]["latency"])
    if change == "missing":
        selected.pop("latency_workload")
    else:
        selected["latency_workload"]["logits" if change == "full_logits" else "use_cache"] = (
            "all_tokens" if change == "full_logits" else 1)
    original_candidate = copy.deepcopy(cases["8"]["latency"])
    original_baseline = copy.deepcopy(baseline["latency"])
    apply_speed_gates(cases, baseline)
    assert cases["8"]["eligible_for_automatic_selection"] is False
    assert cases["8"]["latency_workload_match"]["passed"] is False
    assert cases["8"]["latency_stability"]["passed"] is True
    assert cases["8"]["decode_speedup_vs_fastest_pytorch"] == pytest.approx(10 / 9)
    for field in ("prefill_p50_ms", "decode_p50_ms", "prefill_samples_ms", "decode_samples_ms"):
        assert cases["8"]["latency"][field] == original_candidate[field]
        for implementation in ("eager", "sdpa"):
            assert baseline["latency"][implementation][field] == original_baseline[implementation][field]


@pytest.mark.parametrize("prefill,decode,eligible", [(102, 9, True), (102.001, 9, False),
    (100, 9.8, False), (100, 9.79999, True), (100, 12, False)])
def test_no_slowdown_and_two_percent_decode_thresholds(prefill, decode, eligible):
    cases, baseline = speed_comparison(prefill, decode)
    apply_speed_gates(cases, baseline)
    assert cases["8"]["eligible_for_automatic_selection"] is eligible


@pytest.mark.parametrize("prefill,decode,eligible", [(122.4, 11.9, True),
    (122.401, 11.9, False), (120, 12, False)])
def test_leaf_fp32_speed_and_no_slowdown_are_also_required(prefill, decode, eligible):
    cases, baseline = speed_comparison(prefill, decode)
    baseline["latency"] = {"eager": latency(150, 20), "sdpa": latency(155, 21)}
    apply_speed_gates(cases, baseline)
    assert cases["8"]["eligible_for_automatic_selection"] is eligible


@pytest.mark.parametrize("quality_pass", [False, None, 1, "true"])
def test_quality_pass_must_be_explicit_boolean(quality_pass):
    cases, baseline = speed_comparison()
    cases["8"]["quality_gate_passed"] = quality_pass
    apply_speed_gates(cases, baseline)
    assert cases["8"]["eligible_for_automatic_selection"] is False


@pytest.mark.parametrize("unstable", ["candidate", "leaf", "pytorch"])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_speed_gates_reject_instability_in_either_phase(unstable, phase):
    cases, baseline = speed_comparison()
    target = (cases["8"]["latency"] if unstable == "candidate" else
              cases["32"]["latency"] if unstable == "leaf" else baseline["latency"]["eager"])
    target[f"{phase}_samples_ms"][-1] *= 10
    apply_speed_gates(cases, baseline)
    assert cases["8"]["eligible_for_automatic_selection"] is False
    assert cases["8"]["latency_stability"]["passed"] is False


@pytest.mark.parametrize("malformed", [None, [], {}, {"decode_p50_ms": 9}])
def test_speed_gates_fail_closed_for_malformed_candidate_latency(malformed):
    cases, baseline = speed_comparison()
    cases["8"]["latency"] = malformed
    apply_speed_gates(cases, baseline)
    assert cases["8"]["eligible_for_automatic_selection"] is False


def test_speed_gates_cannot_qualify_without_leaf_fp32_case():
    cases, baseline = speed_comparison()
    del cases["32"]
    apply_speed_gates(cases, baseline)
    assert cases["8"]["eligible_for_automatic_selection"] is False


@pytest.mark.parametrize("malformed", [None, [], {}, {"eager": latency(), "sdpa": None}])
def test_speed_gates_fail_closed_for_malformed_pytorch_latencies(malformed):
    cases, baseline = speed_comparison()
    baseline["latency"] = malformed
    apply_speed_gates(cases, baseline)
    assert cases["8"]["eligible_for_automatic_selection"] is False


@pytest.mark.parametrize("mismatch", ["dataset", "text_column", "pinned_cpu"])
def test_native_validation_rejects_changed_baseline_provenance_before_execution(tmp_path, mismatch):
    model, workdir = tmp_path / "model", tmp_path / "work"
    model.mkdir()
    workdir.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"test weights")
    dataset = tmp_path / "heldout.txt"
    dataset.write_text("held-out text")
    reference = {"model_config_sha256": digest(model / "config.json"),
                 "source_weight_sha256": {"model.safetensors": digest(model / "model.safetensors")},
                 "dataset_sha256": digest(dataset), "threads": 1, "pinned_cpu": 2,
                 "sequence_length": 128, "blocks": 8, "text_column": "text"}
    (workdir / "tokens.json").write_text("{}")
    if mismatch == "dataset":
        reference["dataset_sha256"] = "stale hash"
    else:
        reference[mismatch] = "another column" if mismatch == "text_column" else None
    (workdir / "pytorch.json").write_text(json.dumps(reference))
    args = SimpleNamespace(model=model, workdir=workdir, dataset=dataset, threads=1,
                           cpu=2, sequence_length=128, blocks=8, text_column="text")
    with pytest.raises(ValueError, match="held-out dataset" if mismatch == "dataset" else mismatch):
        native(args)


def test_next_token_quality_excludes_unscored_final_rows():
    expected = np.array([[2., 0.], [0., 2.], [2., 0.]], dtype=np.float32)
    actual = expected.copy()
    actual[-1] = [0., 2.]  # no target after this row
    result = quality(actual, expected, [[0, 1, 0]])
    assert result["evaluated_tokens"] == 2
    assert result["next_token_agreement"] == 1.0
    assert result["perplexity_ratio"] == 1.0


@pytest.mark.parametrize("bad", [np.full((2, 2), np.nan), np.zeros((1, 2))])
def test_invalid_baseline_or_row_count_is_rejected(bad):
    with pytest.raises(AssertionError, match="Invalid native logits"):
        quality(np.zeros((2, 2)), bad, [[0, 1]])
