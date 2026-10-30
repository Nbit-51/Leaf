"""Broader source storage must be explicit, cache-separated and runtime-gated."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from leaf import cli, runtime_check
from test_source_bf16 import fixture_model


def test_matrix_scope_has_separate_cache_and_f32_fallback(tmp_path):
    before, _, _ = fixture_model(tmp_path/'bf16')
    directory = tmp_path/'cache'
    regular = cli.prepare(before.parent, directory, 32)
    head = cli.prepare(before.parent, directory, 32, source_bf16_head=True)
    matrices = cli.prepare(before.parent, directory, 32, source_bf16=True)
    assert len({regular, head, matrices}) == 3
    assert matrices.read_bytes()[:8] == b'LEAFDC06'
    assert cli.prepare(before.parent, directory, 32, source_bf16=True) == matrices
    assert matrices.stat().st_size < head.stat().st_size < regular.stat().st_size
    f32, _, _ = fixture_model(tmp_path/'f32', dtype='F32')
    fallback = tmp_path/'f32-cache'
    assert cli.prepare(f32.parent, fallback, 32, source_bf16=True) == cli.prepare(f32.parent, fallback, 32)
    assert not (fallback/'decoder-32-source-bf16-matrices.leaf').exists()


@pytest.mark.parametrize('bits,activations', [('8','auto'), ('4','32'), ('32','8')])
def test_matrix_storage_rejects_lower_precision_before_download(monkeypatch, bits, activations):
    monkeypatch.setattr(cli, 'resolve_model', lambda *a, **k: pytest.fail('Downloaded weights'))
    with pytest.raises(ValueError, match='FP32'):
        cli.run(SimpleNamespace(cpu=None, source_bf16=True, bits=bits, activation_bits=activations))


def test_matrix_storage_requires_new_runtime_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'cache_root', lambda: tmp_path)
    monkeypatch.setattr(cli, 'executable', lambda *a: tmp_path/'old.exe')
    monkeypatch.setattr(runtime_check, 'check', lambda *a: dict(
        artifact_versions=[2,3,4,5], source_bf16_head_supported=True))
    def resolve(*args, **kwargs):
        kwargs['before_weights']()
        pytest.fail('Downloaded weights despite missing storage support')
    monkeypatch.setattr(cli, 'resolve_model', resolve)
    with pytest.raises(RuntimeError, match='does not support --source-bf16'):
        cli.run(SimpleNamespace(cpu=None, source_bf16=True, bits='32', activation_bits='32',
            model='owner/model', offline=False, plan=None))


def test_storage_scopes_are_mutually_exclusive_before_download(monkeypatch):
    monkeypatch.setattr(cli, 'resolve_model', lambda *a, **k: pytest.fail('Downloaded weights'))
    with pytest.raises(ValueError, match='one source BF16'):
        cli.run(SimpleNamespace(cpu=None, source_bf16=True, source_bf16_head=True))


def test_real_cli_runs_matrices_with_accurate_math(tmp_path, monkeypatch, capsys):
    import os
    from tokenizers import Tokenizer, models, pre_tokenizers
    binary = os.environ.get('LEAF_TEST_DECODER_BIN')
    if not binary:
        pytest.skip('Set LEAF_TEST_DECODER_BIN to the rebuilt v6 runtime')
    before, _, _ = fixture_model(tmp_path/'model')
    tokenizer = Tokenizer(models.WordLevel({'[UNK]':0, **{f'tok{i}':i for i in range(1,101)}}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(before.parent/'tokenizer.json'))
    monkeypatch.setenv('LEAF_CACHE_DIR', str(tmp_path/'cache'))
    monkeypatch.setenv('LEAF_DECODER_BIN', str(Path(binary).resolve()))
    metrics = tmp_path/'metrics.json'
    monkeypatch.setattr('sys.argv', ['leaf','run',str(before.parent),'--offline','--bits','32',
        '--source-bf16','--fp32-math','accurate','--raw','--prompt','tok5 tok6 tok7',
        '--max-tokens','2','--metrics',str(metrics)])
    cli.main()
    saved = json.loads(metrics.read_text())
    assert saved['source_bf16_tensors'] > 2
    assert saved['weight_bits'] == saved['activation_bits'] == 32
    assert saved['fp32_math'] == 'accurate'
    assert 'FP32 compute, source BF16 storage' in capsys.readouterr().err
