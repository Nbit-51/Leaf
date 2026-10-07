import pytest

from tools.benchmark_core_scaling import validate_cores


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
