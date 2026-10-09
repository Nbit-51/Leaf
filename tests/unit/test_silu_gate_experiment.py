import hashlib

import pytest

from tools.benchmark_default_abba import validate_silu_quality
from tools.verify_cached_decoder import require_silu_execution


@pytest.mark.parametrize("metrics", [{}, {"experimental_silu_gate_build": True},
    {"experimental_silu_gate_build": True, "vector_silu_gate_calls": 0},
    {"experimental_silu_gate_build": True, "vector_silu_gate_calls": True},
    {"experimental_silu_gate_build": False, "vector_silu_gate_calls": 22}])
def test_missing_or_inactive_dispatch_is_not_a_candidate(metrics):
    with pytest.raises(ValueError, match="did not execute"):
        require_silu_execution(metrics)


def test_quality_must_bind_the_actual_timed_binary_and_inputs(tmp_path):
    files = [tmp_path / name for name in ("decoder.exe", "model.leaf", "tokens.json")]
    for path in files:
        path.write_bytes(path.name.encode())
    record = {"complete": True, "passed": True, "required_silu_gate": True,
        "cases": {"fp32": {"passed": True, "tokenwise_matches_full": True,
            "execution": {"experimental_silu_gate_build": True, "vector_silu_gate_calls": 22}}},
        "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    validate_silu_quality(record, *files, 32)
    for path in files:
        original = path.read_bytes()
        path.write_bytes(b"different")
        with pytest.raises(ValueError, match="does not match"):
            validate_silu_quality(record, *files, 32)
        path.write_bytes(original)
    record["cases"]["fp32"]["tokenwise_matches_full"] = False
    with pytest.raises(ValueError, match="cache gate"):
        validate_silu_quality(record, *files, 32)


@pytest.mark.parametrize("field", ["complete", "passed", "required_silu_gate"])
def test_incomplete_or_failed_quality_cannot_start_timing(field, tmp_path):
    record = {"complete": True, "passed": True, "required_silu_gate": True}
    record[field] = False
    with pytest.raises(ValueError, match="completed passing"):
        validate_silu_quality(record, tmp_path / 'binary', tmp_path / 'artifact', tmp_path / 'tokens', 8)
