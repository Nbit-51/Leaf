import json
import sys
from types import SimpleNamespace

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


def test_decoder_download_uses_inspected_revision(tmp_path, monkeypatch):
    root = tmp_path / ("a" * 40)
    root.mkdir()
    (root / "config.json").write_text(json.dumps(dict(model_type="llama", hidden_size=16,
        intermediate_size=32, num_attention_heads=2, num_hidden_layers=1, vocab_size=64)))
    calls = []
    def download(*args, **kwargs):
        calls.append(kwargs)
        return str(root)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    cli.resolve_model("owner/model", tmp_path)
    assert calls[1]["revision"] == root.name
