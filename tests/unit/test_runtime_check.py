import json
from types import SimpleNamespace

import pytest

from leaf import cli, runtime_check


@pytest.mark.parametrize("stdout,returncode", [('{"runtime":"leaf_decoder","artifact_versions":[2]}', 0),
                                               ('{"runtime":"leaf_embed","artifact_versions":[1]}', 0),
                                               ('not json', 1), ('[]', 0)])
def test_doctor_rejects_stale_wrong_and_broken_binaries(tmp_path, monkeypatch, stdout, returncode):
    monkeypatch.setattr(runtime_check.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=stdout, returncode=returncode))
    with pytest.raises(RuntimeError, match="Reinstall"):
        runtime_check.check(tmp_path / "runtime", "leaf_decoder")


def test_doctor_accepts_matching_native_capabilities(tmp_path, monkeypatch):
    info = dict(runtime="leaf_decoder", artifact_versions=[2, 3])
    monkeypatch.setattr(runtime_check.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(info), returncode=0))
    assert runtime_check.check(tmp_path / "runtime", "leaf_decoder")["artifact_versions"] == [2, 3]


def test_invalid_override_fails_before_model_download(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAF_DECODER_BIN", str(tmp_path / "missing.exe"))
    monkeypatch.setattr(cli, "resolve_model", lambda *a, **k: pytest.fail("Downloaded before runtime check"))
    args = SimpleNamespace(cpu=None)
    with pytest.raises(FileNotFoundError, match="Remove-Item Env:LEAF_DECODER_BIN"):
        cli.run(args)


def test_embedding_invalid_override_fails_before_model_download(tmp_path, monkeypatch):
    from leaf import embedding
    monkeypatch.setenv("LEAF_EMBED_BIN", str(tmp_path / "missing.exe"))
    monkeypatch.setattr(embedding, "resolve_model", lambda *a, **k: pytest.fail("Downloaded before runtime check"))
    with pytest.raises(FileNotFoundError, match="Remove-Item Env:LEAF_EMBED_BIN"):
        embedding.run(SimpleNamespace(output=tmp_path / "output.npy", metrics=None))
