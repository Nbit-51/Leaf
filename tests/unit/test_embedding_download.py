"""Hub resolution contracts: reject unsupported metadata before downloading weights."""
import json
from fnmatch import fnmatch
from types import SimpleNamespace

import huggingface_hub
from huggingface_hub.utils import GatedRepoError, LocalEntryNotFoundError, RepositoryNotFoundError
import pytest

from leaf.embedding import resolve_model
from tools.verify_embeddinggemma_reference import MODULES


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "snapshots" / ("a" * 40)
    root.mkdir(parents=True)
    config = dict(model_type="gemma3_text", use_bidirectional_attention=True,
                  hidden_activation="gelu_pytorch_tanh", hidden_size=32, intermediate_size=64,
                  num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
                  vocab_size=128, max_position_embeddings=96, head_dim=16,
                  layer_types=["sliding_attention", "full_attention"], sliding_window=16,
                  rms_norm_eps=1e-6, query_pre_attn_scalar=16)
    files = {"config.json": config,
             "modules.json": [dict(idx=i, path=path, type="sentence_transformers.models." + kind)
                              for i, (path, kind) in enumerate(MODULES)],
             "1_Pooling/config.json": dict(word_embedding_dimension=32, include_prompt=True,
                                            pooling_mode_mean_tokens=True),
             "2_Dense/config.json": dict(in_features=32, out_features=48, bias=False,
                                          activation_function="torch.nn.modules.linear.Identity"),
             "3_Dense/config.json": dict(in_features=48, out_features=768, bias=False,
                                          activation_function="torch.nn.modules.linear.Identity")}
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    return root


@pytest.mark.parametrize("offline", [False, True])
def test_model_id_downloads_supporting_files_at_one_revision(snapshot, tmp_path, monkeypatch, offline):
    calls = []
    def download(model, **kwargs):
        calls.append((model, kwargs))
        return str(snapshot)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert resolve_model("google/embeddinggemma-300m", tmp_path, offline) == snapshot
    assert len(calls) == 2
    metadata, weights = [kwargs for _, kwargs in calls]
    assert all(call[1]["local_files_only"] is offline for call in calls)
    assert weights["revision"] == snapshot.name
    assert not any("safetensors" in pattern for pattern in metadata["allow_patterns"])
    for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json", "modules.json",
                 "1_Pooling/config.json", "2_Dense/model.safetensors", "3_Dense/model.safetensors"):
        assert any(fnmatch(name, pattern) for pattern in weights["allow_patterns"])


@pytest.mark.parametrize("mutation", ["backbone", "pooling", "missing_head", "activation"])
def test_unsupported_model_does_not_download_weights(snapshot, tmp_path, monkeypatch, mutation):
    if mutation == "missing_head":
        (snapshot / "3_Dense/config.json").unlink()
    else:
        file = snapshot / ("1_Pooling/config.json" if mutation == "pooling" else "config.json")
        value = json.loads(file.read_text())
        field, replacement = {"backbone": ("model_type", "llama"),
                              "pooling": ("pooling_mode_cls_token", True),
                              "activation": ("hidden_activation", "silu")}[mutation]
        value[field] = replacement
        file.write_text(json.dumps(value))
    calls = []
    def download(*args, **kwargs):
        calls.append(kwargs)
        return str(snapshot)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    with pytest.raises(ValueError, match="not supported"):
        resolve_model("owner/model", tmp_path)
    assert len(calls) == 1


@pytest.mark.parametrize("error,message", [(GatedRepoError("denied", response=SimpleNamespace(headers={}, request=None)), "hf auth login"),
    (LocalEntryNotFoundError("missing"), "Run once online"),
    (RepositoryNotFoundError("missing", response=SimpleNamespace(headers={}, request=None)), "Check the model ID")])
def test_access_failures_are_actionable(tmp_path, monkeypatch, error, message):
    def download(*args, **kwargs):
        raise error
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    with pytest.raises(ValueError, match=message):
        resolve_model("owner/model", tmp_path)


def test_local_and_alias_paths_need_no_hub(snapshot, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Local snapshot must not contact the Hub")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    (tmp_path / "aliases.json").write_text(json.dumps({"my-embedder": str(snapshot)}))
    assert resolve_model(str(snapshot), tmp_path, True) == snapshot.resolve()
    assert resolve_model("my-embedder", tmp_path, True) == snapshot.resolve()


def test_missing_local_folder_has_local_error(tmp_path):
    with pytest.raises(ValueError, match="folder does not exist"):
        resolve_model(str(tmp_path / "missing"), tmp_path)


def test_embed_cli_forwards_model_id_and_offline_flag(monkeypatch):
    import sys
    from leaf import cli, embedding
    captured = []
    monkeypatch.setattr(embedding, "run", lambda args: captured.append(args))
    monkeypatch.setattr(sys, "argv", ["leaf", "embed", "owner/model", "--offline",
                                     "--text", "hello", "--output", "result.npy"])
    assert cli.main() == 0
    assert captured[0].model == "owner/model" and captured[0].offline is True
