import json
import sys
from types import SimpleNamespace

import pytest

from leaf import cli


def test_inspect_remote_unsupported_model_downloads_metadata_only(tmp_path, monkeypatch, capsys):
    metadata = tmp_path / "snapshot"
    metadata.mkdir()
    (metadata / "config.json").write_text('{"model_type":"unsupported"}')
    requests = []
    def download(*args, **kwargs):
        requests.append(kwargs)
        return str(metadata)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    cli.inspect_model(SimpleNamespace(model="owner/model", offline=False, output=None))
    report = json.loads(capsys.readouterr().out)
    assert report["configuration_supported"] is False
    assert report["weights_verified"] is False
    assert len(requests) == 1
    assert requests[0]["allow_patterns"] == ["config.json", "generation_config.json"]


def test_mistral_config_check_reports_memory_without_claiming_weight_parity(tmp_path, capsys):
    config = dict(model_type="mistral", hidden_size=4096, intermediate_size=14336,
                  num_attention_heads=32, num_key_value_heads=8, num_hidden_layers=32,
                  vocab_size=32000, max_position_embeddings=32768, sliding_window=None)
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = SimpleNamespace(model=str(tmp_path), offline=True, output=tmp_path / "report.json")
    cli.inspect_model(args)
    record = json.loads(args.output.read_text())
    assert record["configuration_supported"] is True
    assert 26 < record["fp32_required_weight_gib_lower_bound"] < 28
    assert not record["weights_verified"] and not record["timing_qualified"]
    config["sliding_window"] = 4096
    (tmp_path / "config.json").write_text(json.dumps(config))
    args.output = None
    capsys.readouterr()
    cli.inspect_model(args)
    assert json.loads(capsys.readouterr().out)["configuration_supported"] is True


@pytest.mark.parametrize("revision", [None, "refs/pr/50", "b" * 40])
def test_decoder_download_uses_inspected_revision(tmp_path, monkeypatch, revision):
    root = tmp_path / ("a" * 40)
    root.mkdir()
    (root / "config.json").write_text(json.dumps(dict(model_type="llama", hidden_size=16,
        intermediate_size=32, num_attention_heads=2, num_hidden_layers=1, vocab_size=64)))
    calls = []
    def download(*args, **kwargs):
        calls.append(kwargs)
        return str(root)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    cli.resolve_model("owner/model", tmp_path, revision=revision)
    assert calls[0]["revision"] == revision
    assert calls[1]["revision"] == root.name


def test_revision_rejects_local_snapshot(tmp_path):
    with pytest.raises(ValueError, match="--revision applies to a Hub model ID"):
        cli.resolve_model(str(tmp_path), tmp_path / "cache", revision="main")


@pytest.mark.parametrize("tokenizer_files", [("tokenizer.json",), ("vocab.json", "merges.txt")])
def test_offline_pinned_run_reuses_prepared_snapshot(tmp_path, monkeypatch, tokenizer_files):
    revision = "a" * 40
    snapshot = tmp_path / "hub/models--owner--model/snapshots" / revision
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "model.safetensors", *tokenizer_files):
        (snapshot / name).write_text("{}")
    artifact = cli.model_directory(snapshot, tmp_path) / "decoder-32.leaf"
    artifact.parent.mkdir(parents=True)
    artifact.touch()
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setattr(cli, "validate_model_generation", lambda path, plan: calls.append(path))

    result = cli.resolve_model("owner/model", tmp_path, offline=True, revision=revision,
                               before_weights=lambda: calls.append("runtime"), prefer_prepared_fp32=True)
    assert result == snapshot
    assert calls == [snapshot, "runtime"]


@pytest.mark.parametrize("mode", ["uncached", "changed_weights", "missing_tokenizer", "branch", "online"])
def test_prepared_snapshot_misses_use_hub_resolver(tmp_path, monkeypatch, mode):
    revision = "a" * 40
    snapshot = tmp_path / "hub/models--owner--model/snapshots" / revision
    snapshot.mkdir(parents=True)
    config = dict(model_type="llama", hidden_size=16, intermediate_size=32,
                  num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, vocab_size=64)
    (snapshot / "config.json").write_text(json.dumps(config))
    (snapshot / "tokenizer_config.json").write_text("{}")
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    if mode != "uncached":
        artifact = cli.model_directory(snapshot, tmp_path) / "decoder-32.leaf"
        artifact.parent.mkdir(parents=True)
        artifact.touch()
    if mode == "changed_weights":
        (snapshot / "model.safetensors").write_bytes(b"changed weights")
    if mode == "missing_tokenizer":
        (snapshot / "tokenizer.json").unlink()
    calls = []

    def download(model, **options):
        calls.append((model, options))
        return str(snapshot)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    result = cli.resolve_model("owner/model", tmp_path, offline=mode != "online",
                               revision="main" if mode == "branch" else revision, prefer_prepared_fp32=True)
    assert result == snapshot
    assert len(calls) == 2
    assert calls[0][1]["local_files_only"] is (mode != "online")
    assert calls[1][1]["revision"] == revision


def test_inspection_counts_tied_payload_once_without_reading_weights(tmp_path, capsys):
    config = dict(model_type="llama", hidden_size=16, intermediate_size=32,
                  num_attention_heads=2, num_hidden_layers=1, vocab_size=32,
                  tie_word_embeddings=False)
    filename = tmp_path / "config.json"
    args = SimpleNamespace(model=str(tmp_path), offline=True, output=None)
    filename.write_text(json.dumps(config))
    cli.inspect_model(args)
    untied = json.loads(capsys.readouterr().out)
    config["tie_word_embeddings"] = True
    filename.write_text(json.dumps(config))
    cli.inspect_model(args)
    tied = json.loads(capsys.readouterr().out)
    assert untied["fp32_required_weight_bytes_lower_bound"] - tied["fp32_required_weight_bytes_lower_bound"] == 32 * 16 * 4
    assert tied["weights_verified"] is False and tied["timing_qualified"] is False
    assert not list(tmp_path.glob("*.safetensors"))


@pytest.mark.parametrize("different", ["transform", "source_alias"])
def test_inspection_does_not_merge_distinct_payload_mappings(tmp_path, monkeypatch, capsys, different):
    from tools import decoder_plan
    source = dict(model_type="llama", hidden_size=16, intermediate_size=32,
                  num_attention_heads=2, num_hidden_layers=1, vocab_size=32,
                  tie_word_embeddings=True)
    (tmp_path / "config.json").write_text(json.dumps(source))
    plan = decoder_plan.make_plan(source)
    shared = plan["tensors"]["lm_head.weight"]
    if different == "transform":
        shared["transforms"] = [{"reshape": [32, 16]}]
    else:
        shared.update(source="other.weight", source_aliases=["model.embed_tokens.weight"])
    monkeypatch.setattr(decoder_plan, "make_plan", lambda source: plan)
    cli.inspect_model(SimpleNamespace(model=str(tmp_path), offline=True, output=None))
    record = json.loads(capsys.readouterr().out)
    import math
    assert record["fp32_required_weight_bytes_lower_bound"] == sum(
        math.prod(shape) * 4 for shape in decoder_plan.tensor_shapes(plan["config"]).values())
