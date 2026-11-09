"""Trained-weight formats retain the operator semantics of the reference models."""
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from tools.decoder_plan import greedy_reference_tokens, load_snapshot_plan, make_plan, tensor_shapes, validate_plan
from tools.decoder_validation import run_native
from tools.export_decoder import HEADER, OPERATOR_METADATA, export_decoder
from tools.validate_source_bf16_matrices import audit_payloads


LLAMA3 = dict(rope_type='llama3', factor=8., original_max_position_embeddings=32,
              low_freq_factor=1., high_freq_factor=4.)


def source(kind, **options):
    return dict(model_type=kind, hidden_size=32, num_attention_heads=4, num_key_value_heads=2,
                num_hidden_layers=2, intermediate_size=64, vocab_size=64,
                max_position_embeddings=256, eos_token_id=[2, 3], **options)


def test_qwen3_plan_has_separate_attention_width_and_required_head_norms():
    plan = make_plan(source('qwen3', head_dim=16))
    validate_plan(plan)
    prefix = 'model.layers.0.self_attn.'
    shapes = tensor_shapes(plan['config'])
    assert shapes[prefix + 'q_proj.weight'] == (64, 32)
    assert shapes[prefix + 'o_proj.weight'] == (32, 64)
    assert shapes[prefix + 'q_norm.weight'] == shapes[prefix + 'k_norm.weight'] == (1, 16)
    del plan['tensors'][prefix + 'k_norm.weight']
    with pytest.raises(ValueError, match='missing required'):
        validate_plan(plan)


def test_scaled_rope_legacy_and_modern_metadata_are_equivalent():
    linear = dict(rope_type='linear', factor=2.)
    first = make_plan(source('llama', rope_scaling=linear))
    second = make_plan(source('llama', rope_parameters=linear))
    assert first == second
    assert first['config']['rope_frequencies'][0] == 0.5
    validate_plan(first)


@pytest.mark.parametrize('rope', [dict(rope_type='dynamic', factor=2.), dict(rope_type='linear', factor=0),
                                 dict(LLAMA3, high_freq_factor=1.), dict(LLAMA3, attention_factor=2.)])
def test_unsupported_or_invalid_rope_fails_closed(rope):
    with pytest.raises(ValueError, match='RoPE'):
        make_plan(source('llama', rope_scaling=rope))


@pytest.fixture(scope='module')
def runtime():
    configured = os.environ.get('LEAF_TEST_DECODER_BIN')
    if not configured:
        pytest.skip('Set LEAF_TEST_DECODER_BIN for native operator parity')
    return Path(configured).resolve()


@pytest.fixture(scope='module', params=['linear', 'llama3', 'qwen3', 'qwen3-linear'])
def operator_model(request, tmp_path_factory):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    torch.set_num_threads(1)
    torch.manual_seed(512)
    case = request.param
    kind = 'qwen3' if case.startswith('qwen3') else 'llama'
    rope = LLAMA3 if case == 'llama3' else dict(rope_type='linear', factor=2.) if 'linear' in case else None
    options = dict(head_dim=16 if kind == 'qwen3' else 8, rope_theta=10000.)
    if rope:
        options['rope_scaling'] = rope
    config = (transformers.Qwen3Config if kind == 'qwen3' else transformers.LlamaConfig)(**source(kind, **options))
    config._attn_implementation = 'eager'
    model = (transformers.Qwen3ForCausalLM if kind == 'qwen3' else transformers.LlamaForCausalLM)(config).eval()
    if kind == 'qwen3':
        with torch.no_grad():
            for layer in model.model.layers:
                layer.self_attn.q_norm.weight.copy_(torch.linspace(0.5, 1.5, 16))
                layer.self_attn.k_norm.weight.copy_(torch.linspace(1.5, 0.5, 16))
    path = tmp_path_factory.mktemp(case)
    model.to(torch.bfloat16).save_pretrained(path, safe_serialization=True)
    model.float()
    plan = load_snapshot_plan(path)
    artifact = path / 'model.leaf'
    export_decoder(path, artifact)
    assert HEADER.unpack_from(artifact.read_bytes())[:2] == (b'LEAFDC07', 7)
    return case, path, artifact, model, plan


def test_native_operator_logits_cache_and_generation(runtime, operator_model):
    import torch

    case, _, artifact, model, plan = operator_model
    ids = list(range(5, 53))  # Llama 3 crosses its original context of 32 positions.
    with torch.inference_mode():
        expected = model(torch.tensor([ids]), use_cache=False).logits[0].numpy().reshape(-1)
        expected_ids = greedy_reference_tokens(model, ids[:17], plan, 8)
    for scalar, mode, chunk, policy in ((True, 'verify', 1, 'accurate'), (False, 'verify', 1, 'fast'),
                                        (False, 'verify', 1, 'accurate'), (False, 'chunked', 7, 'accurate'),
                                        (False, 'chunked', 1, 'accurate')):
        actual, _ = run_native(runtime, artifact, [ids], mode=mode, chunk=chunk, scalar=scalar, fp32_math=policy)
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4, err_msg=f'{case}: {mode}, {chunk}, {scalar}')
    _, metrics = run_native(runtime, artifact, [ids[:17]], mode='generate', generate=8, fp32_math='accurate')
    assert metrics['generated_tokens'] == expected_ids


def test_extended_artifact_composes_with_source_bf16(runtime, operator_model):
    _, path, artifact, _, _ = operator_model
    compact = path / 'compact.leaf'
    exported = export_decoder(path, compact, source_bf16=True)
    assert OPERATOR_METADATA.unpack_from(compact.read_bytes(), HEADER.size)[0] == 6
    assert audit_payloads(artifact, compact, exported)['exact']
    prompt = [list(range(5, 38))]
    baseline, _ = run_native(runtime, artifact, prompt, fp32_math='accurate')
    actual, metrics = run_native(runtime, compact, prompt, fp32_math='accurate')
    assert metrics['source_bf16_tensors'] == len(exported['source_bf16_tensors'])
    np.testing.assert_array_equal(actual.view('<u4'), baseline.view('<u4'))


def test_native_rejects_corrupt_operator_metadata(runtime, operator_model):
    _, path, artifact, _, _ = operator_model
    corrupt = bytearray(artifact.read_bytes())
    struct.pack_into('<I', corrupt, HEADER.size + 8, 0xffffffff)
    target = path / 'corrupt.leaf'
    target.write_bytes(corrupt)
    with pytest.raises(RuntimeError, match='invalid decoder operator metadata'):
        run_native(runtime, target, [[5]])


def test_public_cli_runs_and_evaluates_extended_artifact(runtime, operator_model, tmp_path, monkeypatch):
    from leaf import cli
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    _, path, artifact, _, _ = operator_model
    tokens = {'[UNK]': 0, **{f'tok{i}': i for i in range(1, 64)}}
    tokenizer = Tokenizer(models.WordLevel(tokens, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]').save_pretrained(path)
    monkeypatch.setenv('LEAF_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setenv('LEAF_DECODER_BIN', str(runtime))
    common = [str(path), '--offline', '--source-bf16', '--fp32-math', 'accurate']
    metrics_path = tmp_path / 'metrics.json'
    monkeypatch.setattr(cli.sys, 'argv', ['leaf', 'run', *common, '--raw', '--prompt', 'tok5 tok6 tok7',
                                        '--max-tokens', '4', '--metrics', str(metrics_path)])
    assert cli.main() == 0
    metrics = json.loads(metrics_path.read_text())
    _, reference = run_native(runtime, artifact, [[5, 6, 7]], mode='generate', generate=4, fp32_math='accurate')
    assert metrics['source_bf16_tensors'] > 0 and metrics['generated_tokens'] == reference['generated_tokens']

    dataset = tmp_path / 'heldout.txt'
    dataset.write_text(' '.join(f'tok{i}' for i in range(5, 41)), encoding='utf-8')
    output = tmp_path / 'quality'
    monkeypatch.setattr(cli.sys, 'argv', ['leaf', 'evaluate', *common, '--dataset', str(dataset),
                                        '--dataset-source', 'synthetic operator fixture', '--sequence-length', '16',
                                        '--blocks', '1', '--generate', '4', '--output', str(output)])
    assert cli.main() == 0
    report = json.loads((output / 'quality.json').read_text())
    assert report['passed']
    assert report['storage'] == dict(requested='source_bf16', effective='source_bf16', artifact_format='LEAFDC07')
