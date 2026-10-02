"""Generation metadata cannot silently override the native single-EOS policy."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from leaf import cli
from tools.decoder_plan import make_plan, validate_generation_config, validate_snapshot_generation
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
def test_multiple_or_malformed_eos_declarations_are_rejected(value):
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
def test_export_rejects_unsupported_generation_before_weight_access(snapshot, tmp_path, bits):
    (snapshot / "generation_config.json").write_text('{"eos_token_id":[2,3]}')
    artifact = tmp_path / "cached.leaf"
    artifact.write_bytes(b"previous artifact")
    with pytest.raises(ValueError, match="EOS"):
        export_decoder(snapshot, artifact, bits)
    assert artifact.read_bytes() == b"previous artifact"
    assert not artifact.with_suffix(".leaf.partial").exists()


@pytest.mark.parametrize("operation", ["run", "optimize"])
def test_cli_rejects_generation_override_before_using_any_cached_runtime(snapshot, monkeypatch, operation):
    (snapshot / "generation_config.json").write_text('{"eos_token_id":[2,3]}')
    monkeypatch.setattr(cli, "cache_root", lambda: snapshot.parent / "cache")
    monkeypatch.setattr(cli, "executable", lambda *args: pytest.fail("Unsupported EOS reached cached runtime selection"))
    monkeypatch.setattr(cli, "prepare", lambda *args: pytest.fail("Unsupported EOS reached cached artifact preparation"))
    args = SimpleNamespace(model=str(snapshot), offline=True, plan=None, cpu=None)
    with pytest.raises(ValueError, match="EOS"):
        getattr(cli, operation)(args)


def test_validation_plan_identity_rejects_changed_generation_metadata(snapshot):
    (snapshot / "generation_config.json").write_text('{"eos_token_id":null}')
    with pytest.raises(ValueError, match="differs from the decoder plan"):
        decoder_plan_identity(SimpleNamespace(model=snapshot, plan=None))


def test_custom_plan_eos_must_match_snapshot_generation_metadata(snapshot, tmp_path):
    (snapshot / "generation_config.json").write_text('{"eos_token_id":2}')
    plan = make_plan(source(3))
    filename = tmp_path / "custom-plan.json"
    filename.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="differs from the decoder plan"):
        cli.validate_model_generation(snapshot, filename)


def test_hub_generation_metadata_is_checked_before_large_weight_download(snapshot, monkeypatch):
    (snapshot / "generation_config.json").write_text('{"eos_token_id":[2,3]}')
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
