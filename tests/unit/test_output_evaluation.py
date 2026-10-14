import math
import numpy as np
import pytest

from leaf.evaluation import generation_comparison, logit_diagnostics, summarize
from leaf import cli


def row(count, ppl, reference, agreement=1.0):
    return dict(evaluated_tokens=count, perplexity=ppl, pytorch_perplexity=reference,
                next_token_agreement=agreement, leaf_correct_targets=count // 2,
                pytorch_correct_targets=count // 2, max_abs_logit_error=0.001, logits_allclose=True)


def test_corpus_perplexity_uses_token_weighted_log_losses():
    result = summarize([row(1, 4, 4), row(3, 16, 8, 0)])
    assert result["perplexity"] == pytest.approx(math.exp((math.log(4) + 3 * math.log(16)) / 4))
    assert result["pytorch_perplexity"] == pytest.approx(math.exp((math.log(4) + 3 * math.log(8)) / 4))
    assert result["next_token_agreement"] == 0.25
    assert result["evaluated_tokens"] == 4
    assert result["leaf_next_token_accuracy"] == 0.25


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_invalid_perplexities_cannot_produce_a_pass(value):
    with pytest.raises(ValueError):
        summarize([row(5, value, 2)])


def test_empty_measurements_are_not_a_quality_pass():
    with pytest.raises(ValueError):
        summarize([])


@pytest.mark.parametrize("expected,actual,index", [([1, 2], [1, 3], 1), ([1, 2], [1], 1),
                                                  ([1], [1, 2], 1), ([1, 2], [1, 2], None)])
def test_generation_checks_both_token_identity_and_termination(expected, actual, index):
    result = generation_comparison(expected, actual)
    assert result["first_mismatch_index"] == index
    assert result["exact_match"] == (index is None)


def test_quality_mode_does_not_route_through_speed_optimizer(monkeypatch, tmp_path):
    from leaf import evaluation
    seen = []
    monkeypatch.setattr(evaluation, "evaluate", lambda args: seen.append(args))
    monkeypatch.setattr(cli, "optimize", lambda args: pytest.fail("Quality invoked timing benchmark"))
    monkeypatch.setattr(cli.sys, "argv", ["leaf", "evaluate", "example/model", "--dataset", "heldout.txt",
                                        "--dataset-source", "test split", "--output", str(tmp_path / "report")])
    assert cli.main() == 0
    assert seen[0].blocks == 32
    assert seen[0].sequence_length == 128


def test_logits_corruption_changes_accuracy_and_fidelity():
    from tools.decoder_validation import quality
    expected = np.array([[8., 0.], [0., 8.], [8., 0.]], dtype=np.float32)
    changed = expected[:, ::-1].copy()
    result = quality(changed, expected, [[0, 0, 1]])
    assert result["next_token_agreement"] == 0
    assert result["perplexity_ratio"] > 100


def test_probability_diagnostics_are_invariant_to_common_logit_offset():
    expected = np.array([[2., 0., -3.], [1., 2., 3.]], dtype=np.float32)
    actual = expected + 16
    result = logit_diagnostics(actual, expected, [[0, 1]])
    assert result["logit_tolerance_violations"] == 6
    assert result["max_logit_tolerance_ratio"] > 1
    assert result["max_probability_error"] == 0
    assert result["max_total_variation"] == 0
    assert result["mean_kl_reference_to_leaf"] == 0
    assert result["argmax_disagreements"] == []


def test_probability_metrics_match_independent_two_class_calculation():
    expected = np.array([[0., 0.], [0., 0.]], dtype=np.float64)
    actual = np.array([[math.log(3), 0.], [0., 0.]], dtype=np.float64)
    result = logit_diagnostics(actual, expected, [[0, 1]])
    # Reference [1/2,1/2], Leaf [3/4,1/4].
    assert result["scored_rows"] == 1
    assert result["max_probability_error"] == pytest.approx(0.25)
    assert result["max_total_variation"] == pytest.approx(0.25)
    assert result["mean_kl_reference_to_leaf"] == pytest.approx(0.5 * math.log(4 / 3))
    assert result["max_kl_reference_to_leaf"] == result["mean_kl_reference_to_leaf"]


def test_near_tie_disagreement_is_visible_even_when_all_logits_are_close():
    expected = np.array([[1., 1.0001], [0., 0.]])
    actual = np.array([[1.0002, 1.0001], [0., 0.]])
    result = logit_diagnostics(actual, expected, [[0, 1]])
    assert result["logit_tolerance_violations"] == 0
    assert len(result["argmax_disagreements"]) == 1
    assert result["argmax_disagreements"][0]["reference_top2_margin"] == pytest.approx(0.0001)


def test_corpus_diagnostics_pool_kl_by_scored_tokens_and_keep_failures():
    first, second = row(1, 4, 4), row(3, 4, 4)
    for measured, kl, violations in ((first, .1, 0), (second, .3, 2)):
        measured['diagnostics'] = dict(mean_kl_reference_to_leaf=kl, max_kl_reference_to_leaf=kl,
                                      max_probability_error=kl, max_total_variation=kl,
                                      logit_tolerance_violations=violations)
    second['logits_allclose'] = False
    result = summarize([first, second])
    assert result['diagnostics']['mean_kl_reference_to_leaf'] == pytest.approx(.25)
    assert result['diagnostics']['max_total_variation'] == .3
    assert result['diagnostics']['logit_tolerance_violations'] == 2
    assert result['all_logits_close'] is False


def test_diagnostics_exclude_unscored_tails_and_keep_global_row_indices():
    expected = np.tile([0., 1., -10000.], (11, 1))
    actual = expected.copy()
    actual[2] = actual[10] = [100., -100., 0.]  # Sequence tails: unscored.
    actual[8] = [1.00001, 1., -10000.]
    result = logit_diagnostics(actual, expected, [[0] * 3, [0] * 8])
    assert result["scored_rows"] == 9
    assert result["logit_tolerance_violations"] > 1  # Tails still count here.
    assert result["argmax_disagreements"] == [dict(row=8, reference_token=1,
                                                   leaf_token=0, reference_top2_margin=1.0)]
    assert 0 < result["max_total_variation"] < 1
    import json
    json.dumps(result, allow_nan=False)
    assert np.all(expected[:, 0] == 0)  # Inputs stay unchanged.


@pytest.mark.parametrize("actual,expected,sequences", [
    (np.zeros((2, 3)), np.zeros((3, 3)), [[0, 1]]),
    (np.zeros((2, 1)), np.zeros((2, 1)), [[0, 1]]),
    (np.zeros((2, 3)), np.zeros((2, 3)), [[0], [1]]),
    (np.zeros((0, 3)), np.zeros((0, 3)), []),
    (np.full((2, 3), np.nan), np.zeros((2, 3)), [[0, 1]]),
    (np.zeros((2, 3)), np.full((2, 3), np.inf), [[0, 1]]),
])
def test_diagnostic_metrics_reject_invalid_or_unscored_inputs(actual, expected, sequences):
    with pytest.raises(ValueError):
        logit_diagnostics(actual, expected, sequences)
