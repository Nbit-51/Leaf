"""Protect diagnostic normalization and keep noise visible in shape experiments."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import decoder_validation
from tools.benchmark_token_panels import summarize
from tools.profile_decoder import profile_policy, summarize_profile


def profile():
    return {"leaf_decoder_profile": 1, "timings": {
        "prefill": {"forward": {"calls": 4, "tokens": 252, "milliseconds": 40},
                    "linear_fp32": {"calls": 292, "tokens": 0, "milliseconds": 32}},
        "decode": {"forward": {"calls": 4, "tokens": 4, "milliseconds": 8},
                   "linear_fp32": {"calls": 292, "tokens": 0, "milliseconds": 7}}}}


def test_phase_cost_is_per_forward_and_inclusive_time_is_not_double_counted():
    result = summarize_profile(profile(), 4, 63)
    assert result["prefill"]["linear_fp32"] == {"mean_ms_per_forward": 8, "percent_of_forward": 80}
    assert result["prefill"]["unattributed"]["mean_ms_per_forward"] == 2
    assert result["decode"]["linear_fp32"]["mean_ms_per_forward"] == 1.75


@pytest.mark.parametrize("forwards,tokens", [(3, 63), (4, 64)])
def test_profile_rejects_workload_mismatch(forwards, tokens):
    with pytest.raises(ValueError, match="counts"):
        summarize_profile(profile(), forwards, tokens)


def test_profile_policy_restores_parent_after_exception(monkeypatch):
    monkeypatch.setenv("LEAF_EXPERIMENTAL_FLOAT_GEMV", "original")
    monkeypatch.setenv("LEAF_DECODER_PROFILE", "parent")
    monkeypatch.setenv("LEAF_DISABLE_VNNI", "1")
    before = dict(os.environ)
    with pytest.raises(RuntimeError):
        with profile_policy(True):
            assert os.environ["LEAF_EXPERIMENTAL_FLOAT_TILES"] == "1"
            assert "LEAF_DISABLE_VNNI" not in os.environ
            assert "LEAF_EXPERIMENTAL_FLOAT_GEMV" not in os.environ
            raise RuntimeError()
    assert dict(os.environ) == before


@pytest.mark.parametrize("stderr,valid", [
    (json.dumps(profile(), separators=(",", ":")), True),
    ("", False),
    ((json.dumps(profile(), separators=(",", ":")) + "\n") * 2, False),
])
def test_native_profile_capture_requires_exactly_one_record(monkeypatch, stderr, valid):
    def run(command, **kwargs):
        Path(command[3]).write_bytes(np.array([1], dtype="<f4").tobytes())
        Path(command[4]).write_text("{}")
        return SimpleNamespace(returncode=0, stderr=stderr)
    monkeypatch.setattr(decoder_validation.subprocess, "run", run)
    args = (Path("native"), Path("artifact"), [[1, 2]])
    if valid:
        _, result = decoder_validation.run_native(*args, capture_profile=True)
        assert result["diagnostic_profile"] == profile()
    else:
        with pytest.raises(ValueError, match="exactly one"):
            decoder_validation.run_native(*args, capture_profile=True)
    # The existing protocol does not gain diagnostic fields implicitly.
    _, result = decoder_validation.run_native(*args)
    assert result == {}


def test_shape_summary_rejects_noise_and_between_pass_drift_without_dropping_samples():
    passes = [{"stage": stage, "samples_ms": [value] * 21}
              for stage, value in [("full_k", 10), ("kblocked", 8), ("kblocked", 8), ("full_k", 10)]]
    assert summarize(passes)["stable"] is True
    assert summarize(passes)["ratio_kblocked_full_k"] == 0.8
    noisy = copy.deepcopy(passes)
    noisy[1]["samples_ms"][-8:] = [16] * 8
    saved = copy.deepcopy(noisy)
    assert summarize(noisy)["stable"] is False
    assert noisy == saved
    passes[-1]["samples_ms"] = [13] * 21
    assert summarize(passes)["stable"] is False
