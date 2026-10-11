import pytest

from tools.benchmark_embedding_competitor import distribution


@pytest.mark.parametrize("values", [[], [0.0], [-1.0], [float("nan")], [float("inf")]])
def test_invalid_samples_cannot_qualify(values):
    with pytest.raises(ValueError, match="Invalid timing samples"):
        distribution(values)


def test_variable_measurements_remain_visible_and_fail_stability():
    result = distribution([10.0] * 8 + [25.0] * 3)
    assert result["samples"] == 11
    assert result["median_ms"] == 10.0
    assert result["p90_over_p10"] == 2.5
    assert not result["stable"]


def test_stable_samples_are_summarized_without_dropping_values():
    result = distribution([10.0, 10.1, 9.9])
    assert result["stable"]
    assert result["samples"] == 3
    assert result["median_ms"] == 10.0
