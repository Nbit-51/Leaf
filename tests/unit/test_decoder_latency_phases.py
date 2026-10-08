"""Timing-only refreshes must reuse, never recreate, validated model quality."""
from __future__ import annotations

import copy
from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace

import numpy as np
import numpy.testing  # Import before subprocess.run is mocked in native tests.
import pytest

from tools import validate_decoder as validation


OLD_TIME = "2026-10-01T12:00:00+00:00"
NEW_TIME = "2026-10-02T12:00:00+00:00"
NATIVE_POLICY_NAMES = ("LEAF_DISABLE_VNNI", "LEAF_EXPERIMENTAL_FLOAT_TILES", "LEAF_EXPERIMENTAL_FLOAT_GEMV")


def latency(prefill=100, decode=10):
    return {"prefill_p50_ms": prefill, "decode_p50_ms": decode,
            "prefill_samples_ms": [prefill] * 7, "decode_samples_ms": [decode] * 7,
            "threads": 1, "activation_bits": 32,
            "latency_workload": dict(validation.LATENCY_WORKLOAD)}


@pytest.fixture
def fake_torch(monkeypatch):
    """Exercise workload semantics without importing Torch or doing CPU inference."""
    module = SimpleNamespace(tensor=lambda values: np.asarray(values), inference_mode=nullcontext,
                             __version__="mock")
    monkeypatch.setitem(sys.modules, "torch", module)
    ticks = iter(index / 1000 for index in range(1000))
    monkeypatch.setattr(validation.time, "perf_counter", lambda: next(ticks))
    return module


class LastTokenModel:
    def __init__(self):
        self.config = SimpleNamespace(vocab_size=8)
        self.implementation = None
        self.calls = []

    def set_attn_implementation(self, implementation):
        self.implementation = implementation

    def forward(self, ids, past_key_values=None, use_cache=False, logits_to_keep=0):
        self.calls.append({"implementation": self.implementation, "ids": ids.tolist(),
                           "past_key_values": past_key_values, "use_cache": use_cache,
                           "logits_to_keep": logits_to_keep})
        length = logits_to_keep or ids.shape[1]
        return SimpleNamespace(logits=np.zeros((1, length, 8), dtype=np.float32),
                               past_key_values=object() if use_cache else None)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


@pytest.mark.parametrize("implementations", [("eager", "sdpa"), ("sdpa",)])
def test_pytorch_latency_requests_last_token_logits_and_prefill_kv_cache(fake_torch, implementations):
    model, args = LastTokenModel(), SimpleNamespace(warmup=1, runs=5)
    measured = validation.measure_pytorch_latency(model, [1, 2, 3, 4], args, implementations)
    assert set(measured) == set(implementations)
    assert len(model.calls) == 12 * len(implementations)
    for implementation, timing in measured.items():
        assert validation.latency_workload_matches(timing)
        assert timing["prefill_samples_ms"] == pytest.approx([1] * 5)
        assert timing["decode_samples_ms"] == pytest.approx([1] * 5)
        assert timing["stability"]["passed"] is True
        calls = [call for call in model.calls if call["implementation"] == implementation]
        for prefix, decode in zip(calls[::2], calls[1::2]):
            assert prefix["ids"] == [[1, 2, 3]] and prefix["past_key_values"] is None
            assert decode["ids"] == [[4]] and decode["past_key_values"] is not None
            assert prefix["use_cache"] is decode["use_cache"] is True
            assert prefix["logits_to_keep"] == decode["logits_to_keep"] == 1
        # Each measured request starts with a fresh cache, not the prior decode's.
        assert len({id(call["past_key_values"]) for call in calls[1::2]}) == args.warmup + args.runs


@pytest.mark.parametrize("kind", ["missing", "kwargs_only", "positional_only", "variadic_named"])
def test_pytorch_latency_rejects_unverifiable_last_token_option_before_inference(fake_torch, kind):
    def missing(ids, use_cache=False):
        pytest.fail("unsupported model must not be called")

    def kwargs_only(ids, **kwargs):
        pytest.fail("generic kwargs do not prove last-token support")

    def positional_only(ids, logits_to_keep=0, /, use_cache=False):
        pytest.fail("last-token control must be keyword compatible")

    def variadic_named(ids, **logits_to_keep):
        pytest.fail("variadic kwargs are not an explicit last-token parameter")

    model = SimpleNamespace(forward={"missing": missing, "kwargs_only": kwargs_only,
                                     "positional_only": positional_only, "variadic_named": variadic_named}[kind])
    with pytest.raises(ValueError, match="explicitly support logits_to_keep"):
        validation.measure_pytorch_latency(model, [1, 2, 3], SimpleNamespace(warmup=1, runs=5))


@pytest.mark.parametrize("shape", [(1, 3, 8), (2, 1, 8), (1, 1, 0), (1, 1, 7), (1, 8)])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_pytorch_latency_rejects_ignored_keyword_or_invalid_logits_shape(fake_torch, shape, phase):
    class WrongShapeModel(LastTokenModel):
        def forward(self, ids, past_key_values=None, use_cache=False, logits_to_keep=0):
            output = super().forward(ids, past_key_values, use_cache, logits_to_keep)
            if (past_key_values is None) == (phase == "prefill"):
                output.logits = np.zeros(shape)
            return output

    with pytest.raises(ValueError, match=r"last-token logits \[1, 1, vocabulary\]"):
        validation.measure_pytorch_latency(WrongShapeModel(), [1, 2, 3, 4], SimpleNamespace(warmup=1, runs=5))


def test_pytorch_latency_cannot_label_uncached_model_as_cached_decode(fake_torch):
    class NoCacheModel(LastTokenModel):
        def forward(self, ids, past_key_values=None, use_cache=False, logits_to_keep=0):
            output = super().forward(ids, past_key_values, use_cache, logits_to_keep)
            output.past_key_values = None
            return output

    with pytest.raises(ValueError, match="return a KV cache"):
        validation.measure_pytorch_latency(NoCacheModel(), [1, 2, 3], SimpleNamespace(warmup=1, runs=5))


def test_pytorch_latency_requires_prefix_and_decode_token(fake_torch):
    with pytest.raises(ValueError, match="prefix and a cached final token"):
        validation.measure_pytorch_latency(LastTokenModel(), [1], SimpleNamespace(warmup=1, runs=5))


@pytest.mark.parametrize("record", [None, [], {}, {"latency_workload": None},
    {"latency_workload": {**validation.LATENCY_WORKLOAD, "use_cache": 1}},
    {"latency_workload": {**validation.LATENCY_WORKLOAD, "batch_size": True}},
    {"latency_workload": {**validation.LATENCY_WORKLOAD, "prefill": "full_logits"}},
    {"latency_workload": {**validation.LATENCY_WORKLOAD, "unknown": False}}])
def test_latency_workload_requires_exact_canonical_policy(record):
    assert validation.latency_workload_matches(record) is False
    assert validation.latency_workload_matches({"latency_workload": dict(validation.LATENCY_WORKLOAD)}) is True


@pytest.fixture
def cached_run(tmp_path, monkeypatch):
    for name in NATIVE_POLICY_NAMES:
        monkeypatch.delenv(name, raising=False)
    model, workdir = tmp_path / "model", tmp_path / "work"
    model.mkdir()
    workdir.mkdir()
    (model / "config.json").write_text(json.dumps(dict(model_type="llama", hidden_size=16, intermediate_size=32,
                                      num_attention_heads=2, num_hidden_layers=1, vocab_size=8)))
    (model / "model.safetensors").write_bytes(b"trained source weights")
    dataset = tmp_path / "heldout.txt"
    dataset.write_text("held-out corpus")
    executable = tmp_path / "decoder.exe"
    executable.write_bytes(b"validated native executable")
    args = SimpleNamespace(model=model, workdir=workdir, dataset=dataset, threads=1,
                           cpu=2, sequence_length=3, blocks=2, text_column="text",
                           executable=executable, output=tmp_path / "validation.json",
                           warmup=3, runs=7, plan=None, calibration=None)
    data = {"sequences": [[1, 2, 3], [4, 5, 6]], "benchmark_ids": [1, 2, 3, 4],
            "generation_ids": [2, 3], "generated_tokens": [4, 5]}
    (workdir / "tokens.json").write_text(json.dumps(data))
    np.save(workdir / "reference.npy", np.zeros((6, 8), dtype=np.float32))
    baseline = {"model_config_sha256": validation.digest(model / "config.json"),
                "source_weight_sha256": {"model.safetensors": validation.digest(model / "model.safetensors")},
                "dataset_sha256": validation.digest(dataset), "dataset_source": "fixed corpus source",
                "threads": 1, "pinned_cpu": 2, "sequence_length": 3, "blocks": 2,
                "text_column": "text", "generation_prompt": "original prompt",
                "generated_tokens": [4, 5], "generated_text": "original generation",
                "quality_gate_thresholds": copy.deepcopy(validation.GATES),
                "latency_workload": dict(validation.LATENCY_WORKLOAD),
                "measured_at_utc": OLD_TIME, "warmup": 1, "runs": 5,
                "latency": {"eager": latency(), "sdpa": latency(105, 11)}}
    validation.write_record(workdir / "pytorch.json", baseline)
    cases = {}
    plan, plan_sha256 = validation.decoder_plan_identity(args)
    for bits in (32, 8):
        artifact = workdir / f"decoder-{bits}.leaf"
        artifact.write_bytes(f"validated {bits}-bit weights".encode())
        cases[str(bits)] = {"weight_bits": bits, "activation_bits": 32, "artifact": artifact.name,
                            "export_provenance": validation.export_request(baseline, plan, plan_sha256, bits, 32, [], 0),
                            "artifact_sha256": validation.digest(artifact),
                            "quality": {"perplexity_ratio": 1.0, "evaluated_tokens": 4},
                            "quality_gate_passed": True, "generation_exact_match": True,
                            "generated_tokens": [4, 5], "execution": {"quality_mode": "verify"},
                            "latency": latency(1000, 100), "eligible_for_automatic_selection": False}
    result = {"pytorch": copy.deepcopy(baseline), "native": cases,
              "latency_workload": dict(validation.LATENCY_WORKLOAD),
              "native_executable_sha256": validation.digest(executable), "measured_at_utc": OLD_TIME,
              "calibration": {"dataset_sha256": "independent train split", "tokens": 512}}
    validation.write_record(args.output, result)
    monkeypatch.setattr(validation, "utc_now", lambda: NEW_TIME)
    return SimpleNamespace(args=args, data=data, baseline=baseline, result=result)


def forbidden(*args, **kwargs):
    pytest.fail("Timing-only validation must not regenerate quality or export weights")


@pytest.mark.parametrize("flag", NATIVE_POLICY_NAMES)
def test_timing_only_rejects_new_experimental_or_isa_policy_before_any_native_or_export(cached_run, monkeypatch, flag):
    args = cached_run.args
    original = args.output.read_bytes()
    monkeypatch.setenv(flag, "1")
    monkeypatch.setattr(validation, "run_native", forbidden)
    monkeypatch.setattr(validation.subprocess, "run", forbidden)
    monkeypatch.setattr(validation, "decoder_plan_identity", forbidden)
    with pytest.raises(ValueError, match="policy changed"):
        validation.native_latency(args)
    assert args.output.read_bytes() == original


@pytest.mark.parametrize("flag", NATIVE_POLICY_NAMES)
def test_timing_only_rejects_recorded_experiment_after_flag_is_removed(cached_run, monkeypatch, flag):
    args, result = cached_run.args, copy.deepcopy(cached_run.result)
    result["native_policy"] = {name: name == flag for name in NATIVE_POLICY_NAMES}
    validation.write_record(args.output, result)
    original = args.output.read_bytes()
    monkeypatch.setenv(flag, "1")
    monkeypatch.delenv(flag)
    monkeypatch.setattr(validation, "run_native", forbidden)
    monkeypatch.setattr(validation.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="policy changed"):
        validation.native_latency(args)
    assert args.output.read_bytes() == original


@pytest.mark.parametrize("policy", [None, {}, [], False, 0, "default",
    {"LEAF_DISABLE_VNNI": False},
    {**{name: False for name in NATIVE_POLICY_NAMES}, "unknown_flag": False},
    {**{name: False for name in NATIVE_POLICY_NAMES}, "LEAF_EXPERIMENTAL_FLOAT_TILES": 0},
    {**{name: False for name in NATIVE_POLICY_NAMES}, "LEAF_EXPERIMENTAL_FLOAT_GEMV": "false"}])
def test_timing_only_rejects_malformed_native_policy_without_execution_or_exports(cached_run, monkeypatch, policy):
    args, result = cached_run.args, copy.deepcopy(cached_run.result)
    result["native_policy"] = policy
    validation.write_record(args.output, result)
    original = args.output.read_bytes()
    monkeypatch.setattr(validation, "run_native", forbidden)
    monkeypatch.setattr(validation.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="policy"):
        validation.native_latency(args)
    assert args.output.read_bytes() == original


def test_timing_only_legacy_missing_policy_uses_default_false_flags(cached_run, monkeypatch):
    assert "native_policy" not in cached_run.result
    calls = []

    def measure(*args, **kwargs):
        calls.append(kwargs["mode"])
        return np.zeros(0), latency()

    monkeypatch.setattr(validation, "run_native", measure)
    monkeypatch.setattr(validation.subprocess, "run", forbidden)
    validation.native_latency(cached_run.args)
    assert calls == ["bench", "bench"]


@pytest.mark.parametrize("flag", NATIVE_POLICY_NAMES)
def test_timing_only_matching_explicit_experimental_policy_remains_an_explicit_experiment(cached_run, monkeypatch, flag):
    result = copy.deepcopy(cached_run.result)
    result["native_policy"] = {name: name == flag for name in NATIVE_POLICY_NAMES}
    validation.write_record(cached_run.args.output, result)
    monkeypatch.setenv(flag, "1")
    monkeypatch.setattr(validation, "run_native", lambda *args, **kwargs: (np.zeros(0), latency()))
    monkeypatch.setattr(validation.subprocess, "run", forbidden)
    validation.native_latency(cached_run.args)
    refreshed = json.loads(cached_run.args.output.read_text())
    assert refreshed["native_policy"] == result["native_policy"]


@pytest.mark.parametrize("enabled", [(), ("LEAF_EXPERIMENTAL_FLOAT_TILES",), ("LEAF_EXPERIMENTAL_FLOAT_GEMV",),
                                    ("LEAF_EXPERIMENTAL_FLOAT_TILES", "LEAF_EXPERIMENTAL_FLOAT_GEMV"),
                                    ("LEAF_DISABLE_VNNI",)])
def test_fresh_native_validation_records_exact_boolean_runtime_policy(cached_run, monkeypatch, enabled):
    args = cached_run.args
    args.bits, args.generate = [32], 2
    for flag in enabled:
        monkeypatch.setenv(flag, "1")
    reference = np.load(args.workdir / "reference.npy")

    def execute(executable, artifact, sequences, mode="verify", **options):
        if mode == "generate":
            return np.zeros(0), {"generated_tokens": cached_run.data["generated_tokens"]}
        if mode == "bench":
            return np.zeros(0), latency()
        return (reference if mode == "verify" else reference[:args.sequence_length]).ravel().copy(), {}

    monkeypatch.setattr(validation, "run_native", execute)
    monkeypatch.setattr(validation, "cached_export_matches", lambda *args: True)
    monkeypatch.setattr(validation.subprocess, "run", forbidden)
    validation.native(args)
    result = json.loads(args.output.read_text())
    assert result["native_policy"] == {name: name in enabled for name in NATIVE_POLICY_NAMES}
    assert all(isinstance(value, bool) for value in result["native_policy"].values())
    assert validation.latency_workload_matches(result)
    assert validation.latency_workload_matches(result["native"]["32"]["latency"])


def test_baseline_latency_preserves_quality_and_cache_files(cached_run, monkeypatch):
    run, args = cached_run, cached_run.args
    original_cache = {name: (args.workdir / name).read_bytes() for name in ("reference.npy", "tokens.json")}
    calls = []
    model = object()
    monkeypatch.setattr(validation, "load_pytorch_model", lambda received: (model, 0.5))

    def measure(received_model, ids, received_args):
        calls.append((received_model, ids, received_args.warmup, received_args.runs))
        return {"eager": latency(90, 9), "sdpa": latency(95, 10)}

    monkeypatch.setattr(validation, "measure_pytorch_latency", measure)
    monkeypatch.setattr(validation, "quality", forbidden)
    validation.baseline_latency(args)
    refreshed = json.loads((args.workdir / "pytorch.json").read_text())
    assert calls == [(model, run.data["benchmark_ids"], 3, 7)]
    assert refreshed["quality_measured_at_utc"] == OLD_TIME
    assert refreshed["latency_measured_at_utc"] == NEW_TIME
    assert refreshed["measured_at_utc"] == OLD_TIME
    assert refreshed["warmup"] == 3 and refreshed["runs"] == 7
    assert refreshed["latency_load_seconds"] == 0.5
    assert validation.latency_workload_matches(refreshed)
    assert all(validation.latency_workload_matches(item) for item in refreshed["latency"].values())
    assert validation.baseline_quality_identity(refreshed) == validation.baseline_quality_identity(run.baseline)
    assert refreshed["quality_cache_sha256"] == validation.quality_cache_hashes(args.workdir)
    for name, content in original_cache.items():
        assert (args.workdir / name).read_bytes() == content


def test_fresh_baseline_keeps_full_logits_quality_but_times_last_token_only(cached_run, fake_torch, monkeypatch):
    args = cached_run.args
    args.dataset_source, args.generate = "held-out corpus source", 2

    class BaselineModel(LastTokenModel):
        def generate(self, ids, **kwargs):
            return np.concatenate((ids, [[4, 5]]), axis=1)

        def parameters(self):
            return []

        def forward(self, ids, past_key_values=None, use_cache=False, logits_to_keep=0):
            output = super().forward(ids, past_key_values, use_cache, logits_to_keep)
            if not use_cache:
                # Mimic Tensor.numpy without importing actual Torch.
                class QualityLogits:
                    def __getitem__(self, index):
                        return SimpleNamespace(numpy=lambda: output.quality_array[index])
                output.quality_array = output.logits
                output.logits = QualityLogits()
            return output

    model = BaselineModel()
    tokenizer = SimpleNamespace(chat_template=None, eos_token_id=0, encode=lambda prompt: [1, 2],
                                decode=lambda tokens: "mock generated text")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        cpu_count=lambda: 2, virtual_memory=lambda: SimpleNamespace(total=1024),
        Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=64))))
    monkeypatch.setattr(validation, "token_prefix", lambda *args: [1, 2, 3, 4, 5, 6])
    monkeypatch.setattr(validation, "load_pytorch_model", lambda args: (model, 0.1))
    validation.baseline(args)
    record = json.loads((args.workdir / "pytorch.json").read_text())
    assert validation.latency_workload_matches(record)
    assert all(validation.latency_workload_matches(item) for item in record["latency"].values())
    assert record["quality_gate_thresholds"] == validation.GATES
    assert np.load(args.workdir / "reference.npy").shape == (6, 8)
    quality_calls = [call for call in model.calls if not call["use_cache"]]
    assert [call["ids"] for call in quality_calls] == [[[1, 2, 3]], [[4, 5, 6]]]
    assert all(call["logits_to_keep"] == 0 for call in quality_calls)
    assert all(call["logits_to_keep"] == 1 for call in model.calls if call["use_cache"])


@pytest.mark.parametrize("unsupported", [True, False])
def test_failed_matched_baseline_refresh_preserves_legacy_raw_record(cached_run, fake_torch, monkeypatch, unsupported):
    baseline = copy.deepcopy(cached_run.baseline)
    baseline.pop("latency_workload")
    for timing in baseline["latency"].values():
        timing.pop("latency_workload")
    filename = cached_run.args.workdir / "pytorch.json"
    validation.write_record(filename, baseline)
    original = filename.read_bytes()

    class IgnoredKeywordModel(LastTokenModel):
        def forward(self, ids, past_key_values=None, use_cache=False, logits_to_keep=0):
            return super().forward(ids, past_key_values, use_cache, logits_to_keep=0)

    model = (SimpleNamespace(forward=lambda ids, use_cache=False: None) if unsupported
             else IgnoredKeywordModel())
    monkeypatch.setattr(validation, "load_pytorch_model", lambda args: (model, 0.1))
    with pytest.raises(ValueError, match="logits_to_keep|last-token logits"):
        validation.baseline_latency(cached_run.args)
    assert filename.read_bytes() == original


def test_legacy_workloads_need_both_matched_baseline_and_native_refresh_for_selection(cached_run, monkeypatch):
    args = cached_run.args
    baseline, result = copy.deepcopy(cached_run.baseline), copy.deepcopy(cached_run.result)
    baseline.pop("latency_workload")
    for timing in baseline["latency"].values():
        timing.pop("latency_workload")
    result["pytorch"] = copy.deepcopy(baseline)
    result.pop("latency_workload")
    for case in result["native"].values():
        case["latency"].pop("latency_workload")
    validation.write_record(args.workdir / "pytorch.json", baseline)
    validation.write_record(args.output, result)

    def measure_native(executable, artifact, *positional, **options):
        return np.zeros(0), latency(120, 12) if "32" in artifact.name else latency(90, 8)

    monkeypatch.setattr(validation, "run_native", measure_native)
    monkeypatch.setattr(validation, "quality", forbidden)
    monkeypatch.setattr(validation.subprocess, "run", forbidden)
    validation.native_latency(args)
    native_first = json.loads(args.output.read_text())
    for implementation, timing in baseline["latency"].items():
        for field in ("prefill_p50_ms", "decode_p50_ms", "prefill_samples_ms", "decode_samples_ms"):
            assert native_first["pytorch"]["latency"][implementation][field] == timing[field]
        assert "latency_workload" not in native_first["pytorch"]["latency"][implementation]
    assert native_first["native"]["8"]["latency_stability"]["passed"] is True
    assert native_first["native"]["8"]["latency_workload_match"]["passed"] is False
    assert native_first["native"]["8"]["eligible_for_automatic_selection"] is False

    monkeypatch.setattr(validation, "load_pytorch_model", lambda args: (object(), 0.1))
    monkeypatch.setattr(validation, "measure_pytorch_latency", lambda *args: {
        "eager": latency(), "sdpa": latency(110, 11)})
    validation.baseline_latency(args)
    # The baseline refresh cannot rewrite the old native record itself.
    assert json.loads(args.output.read_text()) == native_first
    validation.native_latency(args)
    refreshed = json.loads(args.output.read_text())
    assert refreshed["native"]["8"]["eligible_for_automatic_selection"] is True
    assert refreshed["native"]["8"]["latency_workload_match"]["passed"] is True
    assert validation.baseline_quality_identity(refreshed["pytorch"]) == validation.baseline_quality_identity(baseline)
    for key, case in refreshed["native"].items():
        assert case["quality"] == result["native"][key]["quality"]


@pytest.mark.parametrize("missing", ["reference.npy", "tokens.json", "pytorch.json"])
def test_latency_requires_existing_full_baseline(cached_run, monkeypatch, missing):
    (cached_run.args.workdir / missing).unlink()
    monkeypatch.setattr(validation, "load_pytorch_model", forbidden)
    with pytest.raises(ValueError, match=missing.replace(".", r"\.")):
        validation.baseline_latency(cached_run.args)


@pytest.mark.parametrize("changed", ["config", "weights", "dataset", "text_column", "blocks",
                                      "sequence_length", "threads", "cpu"])
def test_baseline_latency_checks_provenance_before_model_load(cached_run, monkeypatch, changed):
    args = cached_run.args
    if changed in ("config", "weights", "dataset"):
        path = args.model / ("config.json" if changed == "config" else "model.safetensors")
        if changed == "dataset":
            path = args.dataset
        path.write_bytes(b"changed content")
    else:
        setattr(args, changed, "other" if changed == "text_column" else 9)
    monkeypatch.setattr(validation, "load_pytorch_model", forbidden)
    with pytest.raises(ValueError, match="provenance|settings"):
        validation.baseline_latency(args)


@pytest.mark.parametrize("changed", ["reference.npy", "tokens.json"])
def test_sealed_quality_cache_rejects_changed_files(cached_run, changed):
    args = cached_run.args
    baseline = copy.deepcopy(cached_run.baseline)
    baseline["quality_cache_sha256"] = validation.quality_cache_hashes(args.workdir)
    validation.write_record(args.workdir / "pytorch.json", baseline)
    if changed == "tokens.json":
        data = copy.deepcopy(cached_run.data)
        data["benchmark_ids"] = [7, 6, 5, 4]
        (args.workdir / changed).write_text(json.dumps(data))
    else:
        np.save(args.workdir / changed, np.ones((6, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="quality cache hashes changed"):
        validation.load_baseline_cache(args)


@pytest.mark.parametrize("change", ["wrong_blocks", "wrong_length", "bad_ids", "empty_benchmark",
                                    "missing_generation", "changed_generation"])
def test_cached_token_dimensions_and_generation_are_checked(cached_run, change):
    data = copy.deepcopy(cached_run.data)
    if change == "wrong_blocks":
        data["sequences"].pop()
    elif change == "wrong_length":
        data["sequences"][0].pop()
    elif change == "bad_ids":
        data["sequences"][0][0] = True
    elif change == "empty_benchmark":
        data["benchmark_ids"] = []
    elif change == "missing_generation":
        data.pop("generation_ids")
    else:
        data["generated_tokens"] = [7]
    (cached_run.args.workdir / "tokens.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="token cache"):
        validation.load_baseline_cache(cached_run.args)


def test_native_latency_only_retimes_existing_candidates_and_preserves_quality(cached_run, monkeypatch):
    calls = []

    def execute(executable, artifact, sequences, **options):
        calls.append((artifact.name, sequences, options))
        measured = latency(120, 12) if "32" in artifact.name else latency(90, 8)
        return np.zeros(0), measured

    monkeypatch.setattr(validation, "run_native", execute)
    monkeypatch.setattr(validation, "quality", forbidden)
    monkeypatch.setattr(validation, "load_pytorch_model", forbidden)
    monkeypatch.setattr(validation.subprocess, "run", forbidden)
    validation.native_latency(cached_run.args)
    refreshed = json.loads(cached_run.args.output.read_text())
    assert len(calls) == 2
    for _, sequences, options in calls:
        assert sequences == [cached_run.data["benchmark_ids"]]
        assert options == {"mode": "bench", "threads": 1, "runs": 7, "warmup": 3, "activation_bits": 32}
    assert refreshed["quality_measured_at_utc"] == OLD_TIME
    assert refreshed["latency_measured_at_utc"] == NEW_TIME
    assert refreshed["measured_at_utc"] == NEW_TIME
    assert refreshed["calibration"] == cached_run.result["calibration"]
    assert refreshed["native"]["8"]["eligible_for_automatic_selection"] is True
    assert refreshed["native"]["8"]["latency_stability"]["passed"] is True
    assert refreshed["native"]["8"]["latency_workload_match"]["passed"] is True
    assert validation.latency_workload_matches(refreshed)
    for key, prior in cached_run.result["native"].items():
        current = refreshed["native"][key]
        for field in ("quality", "quality_gate_passed", "execution", "generation_exact_match", "generated_tokens"):
            assert current[field] == prior[field]
        assert current["quality_measured_at_utc"] == OLD_TIME
        assert current["latency_measured_at_utc"] == NEW_TIME


@pytest.mark.parametrize("changed", ["executable", "32", "8"])
def test_every_native_hash_is_checked_before_any_timing(cached_run, monkeypatch, changed):
    args = cached_run.args
    path = args.executable if changed == "executable" else args.workdir / f"decoder-{changed}.leaf"
    path.write_bytes(b"changed native code or weights")
    monkeypatch.setattr(validation, "run_native", forbidden)
    with pytest.raises(ValueError, match="changed; rerun full validation"):
        validation.native_latency(args)


@pytest.mark.parametrize("changed", ["executable", "32", "8"])
@pytest.mark.parametrize("phase", ["native", "native-latency"])
def test_mid_run_native_replacement_cannot_relabel_quality_or_write_new_record(cached_run, monkeypatch, changed, phase):
    args = cached_run.args
    args.bits, args.generate = [32, 8], 2
    reference = np.load(args.workdir / "reference.npy")
    original = args.output.read_bytes()
    changed_path = args.executable if changed == "executable" else args.workdir / f"decoder-{changed}.leaf"
    trigger_name = "decoder-32.leaf" if changed == "executable" else changed_path.name
    replacement_done = False

    def execute(executable, artifact, sequences, mode="verify", **options):
        nonlocal replacement_done
        trigger_mode = "verify" if phase == "native" else "bench"
        if mode == trigger_mode and artifact.name == trigger_name and not replacement_done:
            changed_path.write_bytes(b"concurrent replacement")
            replacement_done = True
        if mode == "generate":
            return np.zeros(0), {"generated_tokens": cached_run.data["generated_tokens"]}
        if mode == "bench":
            return np.zeros(0), latency(120, 12) if "32" in artifact.name else latency(90, 8)
        return (reference if mode == "verify" else reference[:args.sequence_length]).ravel().copy(), {}

    monkeypatch.setattr(validation, "run_native", execute)
    monkeypatch.setattr(validation, "cached_export_matches", lambda *args: True)
    monkeypatch.setattr(validation.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="changed during.*rerun full validation"):
        (validation.native if phase == "native" else validation.native_latency)(args)
    assert replacement_done
    assert args.output.read_bytes() == original


def test_native_latency_requires_existing_record(cached_run):
    cached_run.args.output.unlink()
    with pytest.raises(ValueError, match="record missing"):
        validation.native_latency(cached_run.args)


@pytest.mark.parametrize("changed", ["generated_tokens", "quality_timestamp", "quality_hashes", "thresholds"])
def test_native_latency_checks_embedded_baseline_quality_provenance(cached_run, monkeypatch, changed):
    result = copy.deepcopy(cached_run.result)
    if changed == "generated_tokens":
        result["pytorch"]["generated_tokens"] = [7]
    elif changed == "quality_timestamp":
        result["pytorch"]["measured_at_utc"] = "another completed quality run"
    elif changed == "quality_hashes":
        result["pytorch"]["quality_cache_sha256"] = {"reference.npy": "stale", "tokens.json": "stale"}
    else:
        # Both records still agree on the old threshold, but it is not today's
        # validation policy; timing alone cannot promote old quality decisions.
        result["pytorch"]["quality_gate_thresholds"] = {}
        baseline = copy.deepcopy(cached_run.baseline)
        baseline["quality_gate_thresholds"] = {}
        validation.write_record(cached_run.args.workdir / "pytorch.json", baseline)
    validation.write_record(cached_run.args.output, result)
    monkeypatch.setattr(validation, "run_native", forbidden)
    with pytest.raises(ValueError, match="quality|thresholds"):
        validation.native_latency(cached_run.args)


def test_native_artifact_path_cannot_escape_validation_workdir(cached_run, monkeypatch):
    args = cached_run.args
    external = args.workdir.parent / "external.leaf"
    external.write_bytes(b"external weights")
    result = copy.deepcopy(cached_run.result)
    result["native"]["8"].update(artifact="../external.leaf", artifact_sha256=validation.digest(external))
    validation.write_record(args.output, result)
    monkeypatch.setattr(validation, "run_native", forbidden)
    with pytest.raises(ValueError, match="artifact 8 changed"):
        validation.native_latency(args)


def test_refreshed_baseline_latency_can_be_used_without_new_quality(cached_run, monkeypatch):
    monkeypatch.setattr(validation, "load_pytorch_model", lambda args: (object(), 0.2))
    monkeypatch.setattr(validation, "measure_pytorch_latency", lambda *args: {"eager": latency(), "sdpa": latency(110, 11)})
    validation.baseline_latency(cached_run.args)
    monkeypatch.setattr(validation, "run_native", lambda *args, **kwargs: (np.zeros(0), latency(90, 8)))
    validation.native_latency(cached_run.args)
    result = json.loads(cached_run.args.output.read_text())
    assert result["pytorch"]["quality_measured_at_utc"] == OLD_TIME
    assert result["pytorch"]["latency_measured_at_utc"] == NEW_TIME
    assert result["pytorch"]["warmup"] == 3 and result["pytorch"]["runs"] == 7


def test_latency_outer_phase_uses_separate_baseline_and_native_processes(cached_run, monkeypatch):
    calls = []
    args = cached_run.args
    monkeypatch.setattr(validation.sys, "argv", ["validate_decoder.py", "--model", str(args.model),
                                                  "--workdir", str(args.workdir), "--phase", "latency",
                                                  "--runs", "7", "--warmup", "3"])
    monkeypatch.setattr(validation, "pin_cpu", lambda cpu: None)
    monkeypatch.setattr(validation.subprocess, "run", lambda command, **options: calls.append((command, options)))
    validation.main()
    assert len(calls) == 2
    assert calls[0][0][-2:] == ["--phase", "baseline-latency"]
    assert calls[1][0][-2:] == ["--phase", "native-latency"]
    for command, options in calls:
        assert command[0] == validation.sys.executable
        assert options == {"check": True}
        assert "--runs" in command and "--warmup" in command


def test_legacy_native_quality_cannot_be_sealed_by_timing_only(cached_run, monkeypatch):
    result = copy.deepcopy(cached_run.result)
    result["native"]["8"].pop("export_provenance")
    validation.write_record(cached_run.args.output, result)
    monkeypatch.setattr(validation, "run_native", forbidden)
    monkeypatch.setattr(validation.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="provenance.*missing.*phase native"):
        validation.native_latency(cached_run.args)


@pytest.mark.parametrize("changed", ["plan", "weight_bits", "activation_bits", "int8_group_size", "keep_fp32_tensors"])
def test_timing_only_rejects_changed_export_semantics_before_any_native_call(cached_run, monkeypatch, changed):
    args, result = cached_run.args, copy.deepcopy(cached_run.result)
    if changed == "plan":
        plan, _ = validation.decoder_plan_identity(args)
        plan["config"]["epsilon"] *= 2
        args.plan = args.workdir / "custom-plan.json"
        args.plan.write_text(json.dumps(plan))
    else:
        result["native"]["8"][changed] = {"weight_bits": 4, "activation_bits": 8, "int8_group_size": 16,
                                          "keep_fp32_tensors": ["lm_head.weight"]}[changed]
        validation.write_record(args.output, result)
    original = args.output.read_bytes()
    monkeypatch.setattr(validation, "run_native", forbidden)
    monkeypatch.setattr(validation.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="export provenance.*changed"):
        validation.native_latency(args)
    assert args.output.read_bytes() == original


@pytest.mark.parametrize("changed", ["arrays", "split", "missing_argument", "wrong_model"])
def test_timing_only_rejects_changed_calibration_without_reexporting(cached_run, monkeypatch, changed):
    args, result = cached_run.args, copy.deepcopy(cached_run.result)
    args.calibration = args.workdir / "calibration.npz"
    metadata = {"format": "leaf-activation-calibration-v1", "dataset_sha256": "1" * 64,
                "model_config_sha256": cached_run.baseline["model_config_sha256"],
                "source_weight_sha256": cached_run.baseline["source_weight_sha256"]}
    np.savez(args.calibration, input_max=np.ones(16), __metadata__=json.dumps(metadata))
    plan, plan_sha256 = validation.decoder_plan_identity(args)
    case = copy.deepcopy(result["native"]["8"])
    case["activation_bits"] = 8
    case["export_provenance"] = validation.export_request(cached_run.baseline, plan, plan_sha256, 8, 8, [], 0,
                                                        validation.digest(args.calibration))
    result["native"]["8-smooth"] = case
    validation.write_record(args.output, result)
    if changed == "arrays":
        np.savez(args.calibration, input_max=np.full(16, 5.0), __metadata__=json.dumps(metadata))
    elif changed == "split":
        metadata["dataset_sha256"] = "3" * 64
        np.savez(args.calibration, input_max=np.ones(16), __metadata__=json.dumps(metadata))
    elif changed == "missing_argument":
        args.calibration = None
    else:
        metadata["model_config_sha256"] = "0" * 64
        np.savez(args.calibration, input_max=np.ones(16), __metadata__=json.dumps(metadata))
    original = args.output.read_bytes()
    monkeypatch.setattr(validation, "run_native", forbidden)
    monkeypatch.setattr(validation.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="provenance|requires its recorded"):
        validation.native_latency(args)
    assert args.output.read_bytes() == original
