import pytest

from tools.benchmark_core_scaling import validate_cores, validate_smoothed_provenance


TOPOLOGY = [
    dict(group=0, logical_cpu=0, core=0, efficiency_class=1),
    dict(group=0, logical_cpu=1, core=0, efficiency_class=1),
    dict(group=0, logical_cpu=2, core=2, efficiency_class=1),
    dict(group=0, logical_cpu=3, core=3, efficiency_class=0),
]


def test_distinct_physical_cores_keep_requested_order():
    assert [item["logical_cpu"] for item in validate_cores([2, 0], TOPOLOGY)] == [2, 0]


@pytest.mark.parametrize("cores, message", [([], "distinct"), ([0, 0], "distinct"),
    ([0, 1], "SMT"), ([0, 3], "mix"), ([4], "unavailable")])
def test_incomparable_core_budgets_are_rejected(cores, message):
    with pytest.raises(ValueError, match=message):
        validate_cores(cores, TOPOLOGY)


@pytest.mark.parametrize("field,value", [("weight_bits", 32), ("activation_bits", 32),
    ("calibration_sha256", None), ("model_config_sha256", "other"),
    ("source_weight_sha256", {"weights": "other"})])
def test_quantized_timing_rejects_wrong_precision_or_source(field, value):
    baseline = {"model_config_sha256": "config", "source_weight_sha256": {"weights": "source"}}
    request = dict(baseline, weight_bits=8, activation_bits=8, calibration_sha256="calibration")
    provenance = {"artifact_sha256": "artifact", "request": request}
    validate_smoothed_provenance(provenance, baseline, "artifact")
    request[field] = value
    with pytest.raises(ValueError, match="provenance"):
        validate_smoothed_provenance(provenance, baseline, "artifact")


def test_quantized_timing_rejects_changed_artifact():
    baseline = {"model_config_sha256": "config", "source_weight_sha256": {"weights": "source"}}
    request = dict(baseline, weight_bits=8, activation_bits=8, calibration_sha256="calibration")
    with pytest.raises(ValueError, match="provenance"):
        validate_smoothed_provenance({"artifact_sha256": "old", "request": request}, baseline, "new")
