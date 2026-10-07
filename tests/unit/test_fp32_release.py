import pytest

from tools.qualify_fp32_release import paired_gate


def pairs(prefill=.5, decode=.9):
    return [dict(before=dict(prefill_p50_ms=350, decode_p50_ms=30),
                 after=dict(prefill_p50_ms=350 * prefill, decode_p50_ms=30 * decode)) for _ in range(10)]


def test_consistent_release_gain_passes_without_authorizing_precision_selection():
    gate = paired_gate(pairs(), True)
    assert gate["passed"]
    assert gate["prefill"]["median_paired_ratio"] == .5
    assert not gate["automatic_precision_selection_authorized"]


@pytest.mark.parametrize("prefill,decode,quality", [(1, .9, True), (.5, 1.1, True), (.5, .9, False)])
def test_release_requires_prefill_gain_decode_nonregression_and_quality(prefill, decode, quality):
    assert not paired_gate(pairs(prefill, decode), quality)["passed"]


def test_inconsistent_pairs_and_incomplete_runs_do_not_qualify():
    values = pairs()
    for pair in values[:4]:
        pair["after"]["prefill_p50_ms"] = 420
    assert not paired_gate(values, True)["passed"]
    with pytest.raises(ValueError, match="ten"):
        paired_gate(values[:9], True)
