import copy
import math

import numpy as np
import pytest

from tools.audit_decoder_experiment import independent_loss, replay
from tools.benchmark_decoder_comparison import native_experiment_gate, stage_summary
from tools.decoder_validation import benchmark_stability, quality


def test_independent_ce_shifts_each_block_and_handles_large_logits():
    # Boundary rows are deliberately extreme and must never score the next block.
    values = np.array([[1000, 1000], [1000, -1000], [0, math.log(3)], [-1000, 1000]], dtype=np.float64)
    sequences = [[0, 1], [1, 0]]
    oracle = independent_loss(values, sequences)
    assert oracle["evaluated_tokens"] == 2
    assert oracle["block_mean_nll"] == pytest.approx([math.log(2), math.log(4)])
    assert oracle["perplexity"] == pytest.approx(math.sqrt(8))
    assert quality(values, values, sequences)["perplexity"] == pytest.approx(oracle["perplexity"])


def saved_comparison():
    passes = []
    for name in ("before", "after", "after", "before"):
        value = 100.0 if name == "before" else 90.0
        metrics = {"threads": 1, "activation_bits": 32, "prefill_samples_ms": [value] * 7,
                   "decode_samples_ms": [10.0] * 7, "prefill_p50_ms": value, "decode_p50_ms": 10.0}
        passes.append({"stage": name, "latency": metrics, "stability": benchmark_stability(metrics), "native_policy": {}})
    stages = {name: stage_summary([p["latency"] for p in passes if p["stage"] == name]) for name in ("before", "after")}
    return {"runs_per_pass": 7, "threads": 1, "native_policy_by_stage": {"before": {}, "after": {}},
            "native": {"32": {"activation_bits": 32, "quality_gate_passed": True,
                "timing": {"order": [p["stage"] for p in passes], "passes": passes, **stages},
                "gate": native_experiment_gate(stages["before"], stages["after"], True)}}}


def test_replay_recomputes_without_mutating_record():
    record = saved_comparison()
    original = copy.deepcopy(record)
    assert replay(record)["32"]["raw_sample_replay_passed"] is True
    assert record == original


@pytest.mark.parametrize("change", ["sample", "verdict", "stage", "order", "count", "policy"])
def test_replay_rejects_inconsistent_saved_evidence(change):
    record = saved_comparison()
    case = record["native"]["32"]
    if change == "sample":
        case["timing"]["passes"][0]["latency"]["decode_samples_ms"][0] = 1000
    elif change == "verdict":
        case["gate"]["accepted_native_experiment"] = False
    elif change == "stage":
        case["timing"]["before"]["prefill_p50_ms"] = 1
    elif change == "order":
        case["timing"]["order"] = ["after"] * 4
    elif change == "count":
        record["runs_per_pass"] = 8
    else:
        case["timing"]["passes"][0]["native_policy"] = {"unexpected": True}
    with pytest.raises(ValueError):
        replay(record)
