"""Snapshot EOS policies must reach every path without dropping terminators."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from leaf import cli
from tools.decoder_plan import (MAX_EOS_TOKENS, eos_token_ids, generation_policy_identity,
                                greedy_reference_tokens, load_snapshot_plan, make_plan,
                                validate_generation_config, validate_snapshot_generation)
from tools.export_decoder import export_decoder
from tools.validate_decoder import decoder_plan_identity


def source(eos=2):
    return dict(model_type="llama", hidden_size=16, intermediate_size=32, num_attention_heads=2,
                num_hidden_layers=1, vocab_size=32, eos_token_id=eos)


@pytest.fixture
def snapshot(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(source()))
    return model


@pytest.mark.parametrize("generation", [{}, {"max_length": 2048}, {"eos_token_id": 2}, {"eos_token_id": [2]}])
def test_matching_or_inherited_eos_is_accepted(generation):
    validate_generation_config(generation, make_plan(source()))


def test_missing_generation_file_inherits_plan(snapshot):
    validate_snapshot_generation(snapshot, make_plan(source()))


@pytest.mark.parametrize("generation", [{}, {"eos_token_id": None}])
def test_no_eos_null_matches_native_sentinel(generation):
    plan = make_plan(source(None))
    assert plan["config"]["eos"] == 0xffffffff
    validate_generation_config(generation, plan)


@pytest.mark.parametrize("value", [[2, 3], [2, 2], [], [None], [[2]], True, False, "2", 2.0, -1, 32, {}, 0xffffffff])
def test_mismatched_or_malformed_eos_declarations_are_rejected(value):
    with pytest.raises(ValueError, match="EOS"):
        validate_generation_config({"eos_token_id": value}, make_plan(source()))


@pytest.mark.parametrize("generation", [None, [], "config"])
def test_generation_config_requires_an_object(generation):
    with pytest.raises(ValueError, match="object"):
        validate_generation_config(generation, make_plan(source()))


@pytest.mark.parametrize("planned,declared", [(2, 3), (2, None), (None, 2), (2, [3])])
def test_scalar_and_null_termination_overrides_are_rejected(planned, declared):
    with pytest.raises(ValueError, match="differs from the decoder plan"):
        validate_generation_config({"eos_token_id": declared}, make_plan(source(planned)))


@pytest.mark.parametrize("bits", [32, 8, 4])
def test_export_rejects_malformed_generation_before_weight_access(snapshot, tmp_path, bits):
    (snapshot / "generation_config.json").write_text('{"eos_token_id":[2,true]}')
    artifact = tmp_path / "cached.leaf"
    artifact.write_bytes(b"previous artifact")
    with pytest.raises(ValueError, match="EOS"):
        export_decoder(snapshot, artifact, bits)
    assert artifact.read_bytes() == b"previous artifact"
    assert not artifact.with_suffix(".leaf.partial").exists()


@pytest.mark.parametrize("operation", ["run", "optimize"])
def test_cli_rejects_malformed_generation_before_using_any_cached_runtime(snapshot, monkeypatch, operation):
    (snapshot / "generation_config.json").write_text('{"eos_token_id":[2,true]}')
    monkeypatch.setattr(cli, "cache_root", lambda: snapshot.parent / "cache")
    monkeypatch.setattr(cli, "executable", lambda *args: pytest.fail("Unsupported EOS reached cached runtime selection"))
    monkeypatch.setattr(cli, "prepare", lambda *args: pytest.fail("Unsupported EOS reached cached artifact preparation"))
    args = SimpleNamespace(model=str(snapshot), offline=True, plan=None, cpu=None)
    with pytest.raises(ValueError, match="EOS"):
        getattr(cli, operation)(args)


def test_validation_plan_identity_rejects_changed_generation_metadata(snapshot):
    (snapshot / "generation_config.json").write_text('{"eos_token_id":"2"}')
    with pytest.raises(ValueError, match="EOS"):
        decoder_plan_identity(SimpleNamespace(model=snapshot, plan=None))


def test_custom_plan_eos_must_match_snapshot_generation_metadata(snapshot, tmp_path):
    (snapshot / "generation_config.json").write_text('{"eos_token_id":2}')
    plan = make_plan(source(3))
    filename = tmp_path / "custom-plan.json"
    filename.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="differs from the decoder plan"):
        cli.validate_model_generation(snapshot, filename)


def test_hub_generation_metadata_is_checked_before_large_weight_download(snapshot, monkeypatch):
    (snapshot / "generation_config.json").write_text('{"eos_token_id":[2,true]}')
    downloads = []

    def download(model, **options):
        downloads.append(options)
        return str(snapshot)

    import sys
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    with pytest.raises(ValueError, match="EOS"):
        cli.resolve_model("organization/model", snapshot.parent / "cache", offline=True)
    assert len(downloads) == 1
    assert downloads[0]["allow_patterns"] == ["config.json", "generation_config.json"]


def test_generation_metadata_changes_invalidate_source_cache(snapshot):
    cache = snapshot.parent / "cache"
    original = cli.model_directory(snapshot, cache)
    filename = snapshot / "generation_config.json"
    filename.write_text('{"eos_token_id":2}')
    explicit = cli.model_directory(snapshot, cache)
    assert explicit != original
    filename.write_text('{"eos_token_id":2,"max_length":128}')
    assert cli.model_directory(snapshot, cache) != explicit


@pytest.mark.parametrize("value,expected", [([2, 3], [2, 3]), ([3, 0], [3, 0]),
                                            (3, [3]), ([0], [0]), (None, [])])
def test_snapshot_plan_honors_complete_generation_override(snapshot, value, expected):
    (snapshot / "generation_config.json").write_text(json.dumps({"eos_token_id": value,
        "do_sample": True, "repetition_penalty": 1.1, "temperature": 0.7}))
    plan = load_snapshot_plan(snapshot)
    assert eos_token_ids(plan["config"]) == expected
    assert decoder_plan_identity(SimpleNamespace(model=snapshot, plan=None))[0] == plan
    cli.validate_model_generation(snapshot)


@pytest.mark.parametrize("value", [[2, True], [2, 2], [], [None], [[2]], -1, 32, 0xffffffff])
def test_snapshot_resolution_rejects_malformed_eos_instead_of_overriding(snapshot, value):
    (snapshot / "generation_config.json").write_text(json.dumps({"eos_token_id": value}))
    with pytest.raises(ValueError, match="EOS"):
        load_snapshot_plan(snapshot)


def test_multi_eos_from_architecture_config_is_preserved(snapshot):
    (snapshot / "config.json").write_text(json.dumps(source([2, 3])))
    assert eos_token_ids(load_snapshot_plan(snapshot)["config"]) == [2, 3]


def test_custom_plan_cannot_discard_secondary_snapshot_terminator(snapshot, tmp_path):
    (snapshot / "generation_config.json").write_text('{"eos_token_id":[2,3]}')
    custom = tmp_path / "custom-plan.json"
    custom.write_text(json.dumps(make_plan(source())))
    with pytest.raises(ValueError, match="differs from the decoder plan"):
        load_snapshot_plan(snapshot, custom)
    custom.write_text(json.dumps(make_plan(source([2, 3]))))
    assert eos_token_ids(load_snapshot_plan(snapshot, custom)["config"]) == [2, 3]


def test_eos_list_is_bounded_even_with_large_vocabulary():
    config = source(list(range(MAX_EOS_TOKENS + 1)))
    config["vocab_size"] = MAX_EOS_TOKENS + 2
    with pytest.raises(ValueError, match="EOS list"):
        make_plan(config)


def test_secondary_terminator_changes_plan_and_generation_policy_identity(snapshot):
    generation = snapshot / "generation_config.json"
    generation.write_text('{"eos_token_id":[2,3]}')
    first, first_digest = decoder_plan_identity(SimpleNamespace(model=snapshot, plan=None))
    generation.write_text('{"eos_token_id":[2,4]}')
    second, second_digest = decoder_plan_identity(SimpleNamespace(model=snapshot, plan=None))
    assert second_digest != first_digest
    assert generation_policy_identity(first, 16) != generation_policy_identity(second, 16)


@pytest.mark.parametrize("eos,expected", [([2, 3], [3]), (None, [3, 3, 3])])
def test_reference_uses_full_prefix_argmax_without_hub_generation_defaults(snapshot, monkeypatch, eos, expected):
    import sys
    import numpy as np
    (snapshot / "generation_config.json").write_text('{"eos_token_id":[2,3],"do_sample":true,'
        '"temperature":0.7,"repetition_penalty":1.1,"forced_eos_token_id":7}')
    generation = json.loads((snapshot / "generation_config.json").read_text())
    generation["eos_token_id"] = eos
    (snapshot / "generation_config.json").write_text(json.dumps(generation))
    calls = []
    class Model:
        generation_config = generation
        def __call__(self, ids, use_cache):
            assert use_cache is False
            calls.append(ids.tolist())
            logits = np.zeros((1, ids.shape[1], 32), dtype=np.float32)
            logits[0, -1, 3] = 10
            return SimpleNamespace(logits=logits)
        def generate(self, *args, **kwargs):
            pytest.fail("Reference inherited the Hub generate policy")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(tensor=lambda ids: np.asarray(ids)))
    assert greedy_reference_tokens(Model(), [1, 4], load_snapshot_plan(snapshot), 3) == expected
    expected_calls = [[[1, 4]]] if len(expected) == 1 else [[[1, 4]], [[1, 4, 3]], [[1, 4, 3, 3]]]
    assert calls == expected_calls


def test_hub_multi_eos_resolution_still_pins_weights_to_metadata_revision(snapshot, monkeypatch):
    (snapshot / "generation_config.json").write_text('{"eos_token_id":[2,3]}')
    downloads = []
    def download(model, **options):
        downloads.append(options)
        return str(snapshot)
    import sys
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    assert cli.resolve_model("organization/model", snapshot.parent / "cache", offline=True) == snapshot
    assert len(downloads) == 2
    assert downloads[1]["revision"] == snapshot.name


def test_reference_can_predict_one_final_token_at_exact_context_limit(monkeypatch):
    import sys
    import numpy as np
    config = dict(source(None), max_position_embeddings=4)
    plan = make_plan(config)
    calls = []
    def model(ids, use_cache):
        calls.append(ids.tolist())
        return SimpleNamespace(logits=np.zeros((1, ids.shape[1], 32)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(tensor=lambda ids: np.asarray(ids)))
    assert greedy_reference_tokens(model, [1, 2, 3, 4], plan, 1) == [0]
    assert calls == [[[1, 2, 3, 4]]]
    with pytest.raises(ValueError, match="exceeds model context"):
        greedy_reference_tokens(model, [1, 2, 3, 4], plan, 2)


def test_calibration_uses_the_same_effective_multi_eos_plan(snapshot, tmp_path, monkeypatch):
    import sys
    import numpy as np
    from tools import calibrate_decoder
    (snapshot / "generation_config.json").write_text('{"eos_token_id":[2,3]}')
    dataset = tmp_path / "train.txt"
    dataset.write_text("independent calibration split")
    output = tmp_path / "calibration.npz"
    model = object()
    seen = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda threads: None, float32="fp32"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace(eval=lambda: model))))
    monkeypatch.setattr(calibrate_decoder, "token_prefix", lambda *args: [1, 2])
    def collect(actual_model, sequences, plan):
        assert actual_model is model and sequences == [[1, 2]]
        seen.append(eos_token_ids(plan["config"]))
        return {"observed": np.ones(2, dtype=np.float32)}
    monkeypatch.setattr(calibrate_decoder, "collect", collect)
    monkeypatch.setattr(sys, "argv", ["calibrate_decoder", "--model", str(snapshot), "--dataset", str(dataset),
                                     "--output", str(output), "--blocks", "1", "--sequence-length", "2"])
    calibrate_decoder.main()
    assert seen == [[2, 3]] and output.is_file()
