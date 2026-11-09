"""Policy cost diagnostics retain failures and require bound native evidence."""
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from tools.benchmark_counters import validate_benchmark_counters
from tools.benchmark_fp32_math import comparison_summary, validate_quality_binding
from tools.decoder_validation import run_native
from tools.validate_decoder import digest


def passes():
    result = []
    for pair, stages in enumerate((("before", "after"), ("after", "before"))):
        for stage in stages:
            value = 20.0 if stage == "before" else 10.0
            result.append(dict(pair=pair, stage=stage, metrics=dict(
                prefill_samples_ms=[value] * 5, decode_samples_ms=[value / 2] * 5,
                prefill_p50_ms=value, decode_p50_ms=value / 2)))
    return result


def test_complete_abba_summary_describes_ratios_without_accepting_a_claim():
    records = passes()
    before = copy.deepcopy(records)
    summary = comparison_summary(records)
    assert records == before
    for phase in ("prefill", "decode"):
        assert summary[phase]["pooled_ratio"] == 0.5
        assert summary[phase]["median_paired_ratio"] == 0.5
        assert summary[phase]["stable"] is True
        assert "passed" not in summary[phase]
        assert "timing_qualified" not in summary[phase]


def test_unstable_pass_is_kept_even_when_pooled_ratio_is_favorable():
    records = passes()
    records[1]["metrics"]["decode_samples_ms"] = [1, 5, 5, 5, 100]
    before = copy.deepcopy(records)
    summary = comparison_summary(records)
    assert records == before
    assert summary["decode"]["pooled_ratio"] < 1
    assert summary["decode"]["stable"] is False
    assert summary["decode"]["stability"]["after"]["individual"][0]["passed"] is False
    assert summary["prefill"]["stable"] is True


@pytest.mark.parametrize("kind", ["empty", "partial", "unbalanced", "duplicate_pair_stage", "wrong_order"])
def test_incomplete_or_unbalanced_pairings_are_rejected(kind):
    records = passes()
    if kind == "empty":
        records = []
    elif kind == "partial":
        records.pop()
    elif kind == "unbalanced":
        records[0]["stage"] = "after"
    elif kind == "duplicate_pair_stage":
        records[2]["pair"] = 0
    else:
        records[0], records[1] = records[1], records[0]
    with pytest.raises(ValueError):
        comparison_summary(records)


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_summary_rejects_invalid_raw_latency_samples(value):
    records = passes()
    records[0]["metrics"]["prefill_samples_ms"][0] = value
    with pytest.raises(ValueError):
        comparison_summary(records)


def bound_proof(tmp_path):
    binary, artifact = tmp_path / "native.exe", tmp_path / "model.leaf"
    binary.write_bytes(b"runtime contents")
    artifact.write_bytes(b"model contents")
    proof = dict(complete=True, passed=True,
        hashes={str(p.resolve()): digest(p) for p in (binary, artifact)})
    return proof, binary, artifact


def test_full_quality_report_must_bind_both_runtime_and_model(tmp_path):
    proof, binary, artifact = bound_proof(tmp_path)
    validate_quality_binding(proof, binary, artifact)


@pytest.mark.parametrize("which", ["binary", "artifact"])
def test_changed_inputs_cannot_reuse_old_quality_proof(tmp_path, which):
    proof, binary, artifact = bound_proof(tmp_path)
    (binary if which == "binary" else artifact).write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not bind"):
        validate_quality_binding(proof, binary, artifact)


@pytest.mark.parametrize("field", ["complete", "passed"])
@pytest.mark.parametrize("value", [False, None, "true", 1])
def test_quality_proof_requires_literal_complete_pass(tmp_path, field, value):
    proof, binary, artifact = bound_proof(tmp_path)
    proof[field] = value
    with pytest.raises(ValueError, match="complete passing"):
        validate_quality_binding(proof, binary, artifact)


@pytest.mark.parametrize("policy", ["fast", "accurate"])
@pytest.mark.parametrize("has_counters", [False, True])
def test_requested_native_counters_are_forwarded_and_validated(monkeypatch, tmp_path, policy, has_counters):
    from test_benchmark_counters import fixture
    measured = fixture() if has_counters else {}
    measured["fp32_math"] = policy
    def invoke(command, **kwargs):
        expected = ["32"] + (["--fp32-math=accurate"] if policy == "accurate" else [])
        assert command[12:] == expected + ["--benchmark-counters"]
        Path(command[3]).write_bytes(np.array([1, 2], dtype="<f4").tobytes())
        Path(command[4]).write_text(json.dumps(measured))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr("tools.decoder_validation.subprocess.run", invoke)
    kwargs = dict(mode="bench", runs=3, benchmark_counters=True, fp32_math=policy)
    if has_counters:
        logits, metrics = run_native(tmp_path / "native", tmp_path / "artifact", [[1, 2]], **kwargs)
        assert logits.tolist() == [1, 2]
        validate_benchmark_counters(metrics, 3)
    else:
        with pytest.raises(ValueError, match="benchmark_counters object"):
            run_native(tmp_path / "native", tmp_path / "artifact", [[1, 2]], **kwargs)


@pytest.mark.parametrize("mode", ["verify", "chunked", "generate"])
def test_nonbenchmark_counter_requests_fail_before_launch(monkeypatch, tmp_path, mode):
    monkeypatch.setattr("tools.decoder_validation.subprocess.run", lambda *a, **k: pytest.fail("Must reject before launch"))
    with pytest.raises(ValueError, match="require bench mode"):
        run_native(tmp_path / "native", tmp_path / "artifact", [[1, 2]], mode=mode, benchmark_counters=True)


def native_binary():
    path = os.environ.get("LEAF_TEST_DECODER_BIN")
    if not path:
        pytest.skip("Set LEAF_TEST_DECODER_BIN to a rebuilt native runtime")
    return Path(path)


@pytest.mark.parametrize("mode,tail,message", [
    ("verify", ["--benchmark-counters"], "require bench mode"),
    ("bench", ["--benchmark-counterz"], "unknown decoder option"),
    ("bench", ["--benchmark-counters", "--benchmark-counters"], "duplicate benchmark"),
    ("bench", ["--benchmark-counters", "0,1"], "must precede named"),
])
def test_actual_native_bad_flags_fail_before_reading_inputs(tmp_path, mode, tail, message):
    binary = native_binary()
    command = [str(binary), "missing-model", "missing-request", str(tmp_path / "out"),
        str(tmp_path / "metrics"), mode, "1", "1", "0", "1", "0", "1", "32", *tail]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and message in result.stderr
    assert "cannot open token request" not in result.stderr


@pytest.mark.parametrize("policy", ["fast", "accurate"])
def test_real_native_counters_cover_requests_workers_and_exclude_warmups(tmp_path, policy):
    binary = native_binary()
    if os.name != "nt":
        pytest.skip("Optional CPU accounting currently requires Windows")
    from test_decoder_generation_format import fixture_model
    artifact, _ = fixture_model(tmp_path / "model", None)
    info = json.loads(subprocess.check_output([str(binary), "--runtime-info"], text=True))
    assert info["benchmark_counters_supported"] is True
    assert info["benchmark_counter_version"] == 1
    sequences = [[1, 2, 3], [2, 3, 4, 5]]
    _, metrics = run_native(binary, artifact, sequences, mode="bench", threads=2,
        runs=5, warmup=2, fp32_math=policy, benchmark_counters=True)
    validate_benchmark_counters(metrics, 10)
    assert metrics["fp32_math"] == policy and metrics["threads"] == 2
    for phase in ("prefill", "decode"):
        assert len(metrics[phase + "_samples_ms"]) == 10
        assert all(len(values) == 10 for values in metrics["benchmark_counters"][phase].values())
    _, ordinary = run_native(binary, artifact, sequences, mode="bench", threads=2,
        runs=2, warmup=1, fp32_math=policy)
    assert "benchmark_counters" not in ordinary


def mock_main_setup(monkeypatch, tmp_path):
    from tools import benchmark_fp32_math as benchmark
    binary, artifact = tmp_path / "runtime.exe", tmp_path / "model.leaf"
    binary.write_bytes(b"runtime")
    artifact.write_bytes(b"model")
    tokens, proof = tmp_path / "tokens.json", tmp_path / "quality.json"
    tokens.write_text(json.dumps(dict(benchmark_ids=[1, 2, 3])))
    proof.write_text("{}")
    args = SimpleNamespace(binary=binary, artifact=artifact, tokens=tokens,
        quality_report=proof, directory=tmp_path / "diagnostic", before_math="fast",
        after_math="accurate", threads=1, worker_cpus=None, without_counters=True,
        runs=5, warmup=1, pairs=2, settle_seconds=0, required_power_mode=None)
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(benchmark, "validate_quality_binding", lambda *a: None)
    monkeypatch.setattr(benchmark.subprocess, "check_output", lambda *a, **k:
        json.dumps(dict(fp32_math_modes=["fast", "accurate"])))
    monkeypatch.setattr(benchmark, "keep_awake", lambda **k: nullcontext())
    def power_guard(record, **kwargs):
        record["passed"] = True
        return nullcontext(record)
    monkeypatch.setattr(benchmark, "benchmark_power_mode_guard", power_guard)
    # main uses local scratch paths; register the originals so pytest teardown
    # restores direct changes made by the harness as well as monkeypatches.
    for name in ("TEMP", "TMP"):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for name in list(os.environ):
        if name.startswith("LEAF_EXPERIMENTAL_") or name in (
                "LEAF_DECODER_PROFILE", "LEAF_DISABLE_VNNI", "LEAF_DIAGNOSTIC_START_GATE"):
            monkeypatch.delenv(name)
    return benchmark, args.directory / "comparison.json"


def test_completed_raw_pass_survives_a_postrun_ac_failure(monkeypatch, tmp_path):
    benchmark, report = mock_main_setup(monkeypatch, tmp_path)
    metrics = dict(fp32_math="fast", threads=1, activation_bits=32, benchmark_ac_checked=True,
        prefill_samples_ms=[10] * 5, decode_samples_ms=[5] * 5,
        prefill_p50_ms=10, decode_p50_ms=5)
    monkeypatch.setattr(benchmark, "run_native", lambda *a, **k: (np.array([]), metrics))
    calls = []
    def ac():
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("AC disconnected after measured pass")
        return dict(ac_line_status=1)
    monkeypatch.setattr(benchmark, "require_benchmark_ac", ac)
    with pytest.raises(RuntimeError, match="AC disconnected"):
        benchmark.main()
    record = json.loads(report.read_text())
    assert record["complete"] is False
    assert record["timing_qualified"] is False
    assert "AC disconnected" in record["error"]
    assert len(record["passes"]) == 1
    retained = record["passes"][0]
    assert retained["metrics"] == metrics
    assert retained["stage"] == "before" and retained["fp32_math"] == "fast"
    assert retained.get("validated") is not True


def test_failed_native_launch_retains_the_attempted_pass(monkeypatch, tmp_path):
    benchmark, report = mock_main_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(benchmark, "require_benchmark_ac", lambda: dict(ac_line_status=1))
    def fail(*args, **kwargs):
        raise RuntimeError("native execution interrupted")
    monkeypatch.setattr(benchmark, "run_native", fail)
    with pytest.raises(RuntimeError, match="native execution interrupted"):
        benchmark.main()
    record = json.loads(report.read_text())
    assert record["complete"] is False and record["passes"] == []
    assert "native execution interrupted" in record["error"]
    assert record["current_pass"]["stage"] == "before"
    assert record["current_pass"]["fp32_math"] == "fast"
    assert record["current_pass"]["pair"] == 0


def test_distinct_storage_artifacts_are_bound_and_forwarded_per_stage(monkeypatch, tmp_path):
    benchmark, report = mock_main_setup(monkeypatch, tmp_path)
    args = benchmark.parse_args()
    args.before_binary, args.before_artifact = tmp_path/'before.exe', tmp_path/'before.leaf'
    args.before_quality_report = tmp_path/'before-quality.json'
    args.before_binary.write_bytes(b'prior runtime')
    args.before_artifact.write_bytes(b'original source values')
    args.before_quality_report.write_text('{}')
    bound, invoked = [], []
    monkeypatch.setattr(benchmark, 'validate_quality_binding', lambda proof, binary, artifact:
                        bound.append((binary, artifact)))
    monkeypatch.setattr(benchmark, 'require_benchmark_ac', lambda: dict(ac_line_status=1))
    def run(binary, artifact, sequences, **kwargs):
        invoked.append((binary, artifact))
        return np.array([]), dict(fp32_math=kwargs['fp32_math'], threads=1,
            activation_bits=32, benchmark_ac_checked=True,
            prefill_samples_ms=[10.0]*5, decode_samples_ms=[5.0]*5,
            prefill_p50_ms=10.0, decode_p50_ms=5.0)
    monkeypatch.setattr(benchmark, 'run_native', run)
    benchmark.main()
    before, after = (args.before_binary, args.before_artifact), (args.binary, args.artifact)
    assert bound == [after, before]
    assert invoked == [before, after, after, before]
    result = json.loads(report.read_text())
    assert result['artifacts'] == dict(before=str(args.before_artifact), after=str(args.artifact))
    assert result['hashes'][str(args.before_artifact)] == benchmark.digest(args.before_artifact)
    assert result['complete'] is True and result['timing_qualified'] is False
