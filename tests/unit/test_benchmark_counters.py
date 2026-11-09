"""Counter diagnostics must preserve samples and fail on missing accounting."""
import copy

import pytest

from tools.benchmark_counters import (summarize_benchmark_counters,
                                      validate_benchmark_counters)


def fixture():
    phase = dict(interval_ms=[11.0, 21.0, 31.0], process_cpu_ms=[0, 46.875, 62.5],
        process_cycles=[100, 200, 300], process_page_faults=[0, 1, 2],
        caller_cpu_ms=[0, 15.625, 31.25], caller_cycles=[50, 100, 150],
        caller_cpu_start=[2, 2, 4], caller_cpu_end=[2, 4, 4],
        caller_group_start=[0, 0, 0], caller_group_end=[0, 0, 0])
    return dict(prefill_samples_ms=[10, 20, 30], decode_samples_ms=[10, 20, 30],
        benchmark_counters=dict(version=1, scope="process includes all workers",
            interval="monotonic span including complete counter reads",
            prefill=copy.deepcopy(phase), decode=copy.deepcopy(phase)))


def test_valid_zero_cpu_times_and_multithread_cpu_greater_than_wall_are_retained():
    metrics = fixture()
    before = copy.deepcopy(metrics)
    validate_benchmark_counters(metrics, 3)
    assert metrics == before


def test_summary_keeps_all_samples_and_separate_wall_and_counter_intervals():
    metrics = fixture()
    result = summarize_benchmark_counters(metrics)
    assert result["prefill"] == dict(sample_count=3, wall_median_ms=20,
        interval_median_ms=21, process_cpu_median_ms=46.875,
        caller_cpu_median_ms=15.625, process_cycles_median=200,
        caller_cycles_median=100, process_page_faults_total=3,
        caller_endpoints=[dict(group=0, cpu=2), dict(group=0, cpu=4)])
    assert result["decode"] == result["prefill"]
    assert "frequency" not in str(result) and "off_cpu" not in str(result)


@pytest.mark.parametrize("value", [None, [], True, "1"])
def test_counter_object_is_required(value):
    metrics = fixture()
    metrics["benchmark_counters"] = value
    with pytest.raises(ValueError, match="benchmark_counters object"):
        validate_benchmark_counters(metrics, 3)


@pytest.mark.parametrize("value", [None, False, True, 1.0, "1", 0, 2])
def test_counter_version_must_be_integer_one(value):
    metrics = fixture()
    metrics["benchmark_counters"]["version"] = value
    with pytest.raises(ValueError, match="version"):
        validate_benchmark_counters(metrics, 3)


@pytest.mark.parametrize("field", ["scope", "interval"])
@pytest.mark.parametrize("value", [None, "", " ", False, 2])
def test_schema_descriptions_are_required(field, value):
    metrics = fixture()
    metrics["benchmark_counters"][field] = value
    with pytest.raises(ValueError, match=field):
        validate_benchmark_counters(metrics, 3)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("field", ["interval_ms", "process_cpu_ms", "caller_cpu_ms",
    "process_cycles", "caller_cycles", "process_page_faults", "caller_cpu_start",
    "caller_cpu_end", "caller_group_start", "caller_group_end"])
def test_partial_arrays_are_rejected(phase, field):
    metrics = fixture()
    metrics["benchmark_counters"][phase][field].pop()
    with pytest.raises(ValueError, match=field):
        validate_benchmark_counters(metrics, 3)


@pytest.mark.parametrize("field", ["interval_ms", "process_cpu_ms", "caller_cpu_ms"])
@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1, "2", None, 10**500])
def test_numeric_counter_samples_must_be_finite_nonnegative(field, value):
    metrics = fixture()
    metrics["benchmark_counters"]["prefill"][field][0] = value
    with pytest.raises(ValueError, match=field):
        validate_benchmark_counters(metrics, 3)


@pytest.mark.parametrize("field,maximum", [("process_cycles", 2**64 - 1),
    ("caller_cycles", 2**64 - 1), ("process_page_faults", 2**32 - 1),
    ("caller_cpu_start", 255), ("caller_cpu_end", 255),
    ("caller_group_start", 65535), ("caller_group_end", 65535)])
@pytest.mark.parametrize("kind", ["bool", "float", "negative", "overflow"])
def test_integer_counters_keep_native_ranges(field, maximum, kind):
    metrics = fixture()
    values = dict(bool=True, float=2.0, negative=-1, overflow=maximum + 1)
    metrics["benchmark_counters"]["decode"][field][0] = values[kind]
    with pytest.raises(ValueError, match=field):
        validate_benchmark_counters(metrics, 3)


def test_raw_cycles_above_float_integer_precision_remain_exact():
    metrics = fixture()
    metrics["benchmark_counters"]["prefill"]["process_cycles"] = [2**63, 2**63 + 1, 2**63 + 2]
    validate_benchmark_counters(metrics, 3)
    assert summarize_benchmark_counters(metrics)["prefill"]["process_cycles_median"] == 2**63 + 1


def test_shorter_bracket_is_rejected_without_modifying_samples():
    metrics = fixture()
    metrics["benchmark_counters"]["prefill"]["interval_ms"][1] = 19
    before = copy.deepcopy(metrics)
    with pytest.raises(ValueError, match="shorter"):
        validate_benchmark_counters(metrics, 3)
    assert metrics == before


def test_decimal_rounding_between_wall_and_bracket_is_tolerated():
    metrics = fixture()
    metrics["benchmark_counters"]["prefill"]["interval_ms"][1] = 20 - 1e-7
    validate_benchmark_counters(metrics, 3)


@pytest.mark.parametrize("value", [None, False, 0, -1, 3.0, "3"])
def test_expected_sample_count_is_explicit_positive_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        validate_benchmark_counters(fixture(), value)


def test_mismatched_forward_sample_count_is_rejected():
    metrics = fixture()
    metrics["decode_samples_ms"].append(5)
    with pytest.raises(ValueError, match="decode_samples_ms"):
        validate_benchmark_counters(metrics, 3)


@pytest.mark.parametrize("value", [0, True, float("nan"), float("inf"), -1])
def test_forward_wall_samples_must_be_finite_and_positive(value):
    metrics = fixture()
    metrics["prefill_samples_ms"][0] = value
    with pytest.raises(ValueError, match="prefill_samples_ms"):
        validate_benchmark_counters(metrics, 3)


def test_summarizer_rejects_a_missing_phase_instead_of_reporting_zeros():
    metrics = fixture()
    del metrics["benchmark_counters"]["decode"]
    with pytest.raises(ValueError, match="decode"):
        summarize_benchmark_counters(metrics)
