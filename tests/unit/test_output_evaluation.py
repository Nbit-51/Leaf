import math
import numpy as np
import pytest

from leaf.evaluation import generation_comparison, summarize
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
