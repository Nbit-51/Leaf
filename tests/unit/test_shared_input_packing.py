"""Shared packing must reuse only promised inputs and preserve numerical bits."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from test_source_bf16 import fixture_model
from tools.decoder_plan import load_snapshot_plan, tensor_shapes
from tools.decoder_validation import run_native
from tools.export_decoder import export_decoder
from tools.validate_decoder import digest


@pytest.fixture
def native():
    value = os.environ.get('LEAF_TEST_DECODER_BIN')
    if not value:
        pytest.skip('Set LEAF_TEST_DECODER_BIN to the rebuilt packing runtime')
    return Path(value).resolve()


@pytest.fixture
def baseline():
    value = os.environ.get('LEAF_TEST_PACK_BASELINE_BIN')
    if not value:
        pytest.skip('Set LEAF_TEST_PACK_BASELINE_BIN for old/new numerical regression')
    return Path(value).resolve()


def profile(monkeypatch, native, artifact, *, policy='fast', scalar=False):
    monkeypatch.setenv('LEAF_DECODER_PROFILE', '1')
    _, metrics = run_native(native, artifact, [list(range(5, 15))], mode='bench',
        runs=2, warmup=1, scalar=scalar, fp32_math=policy, activation_bits=32,
        capture_profile=True)
    return metrics, metrics['diagnostic_profile']['linear_details']


@pytest.mark.parametrize('policy', ['fast', 'accurate'])
def test_profile_counts_only_four_fresh_packs_and_three_reuses_per_prefill(
        native, tmp_path, monkeypatch, policy):
    artifact, _, _ = fixture_model(tmp_path/'model', dtype='F32')
    monkeypatch.delenv('LEAF_EXPERIMENTAL_FLOAT_TILES', raising=False)
    metrics, detail = profile(monkeypatch, native, artifact, policy=policy)
    # A one-layer gated block has Q/K/V, output, up/gate and down projections.
    # The last-row vocabulary head uses GEMV, not a packed token panel.
    packed = metrics['avx2'] and (policy == 'accurate' or metrics['optimized_fp32_active'])
    assert detail['prefill']['packs'] == (12 if packed else 0)
    assert detail['prefill']['reuses'] == (9 if packed else 0)
    assert detail['decode']['packs'] == detail['decode']['reuses'] == 0
    # Native profiling includes the warmup; it is not a latency acceptance run.
    assert metrics['diagnostic_profile']['timings']['prefill']['forward']['calls'] == 3


@pytest.mark.parametrize('policy', ['fast', 'accurate'])
def test_forced_scalar_never_reuses_packed_input(native, tmp_path, monkeypatch, policy):
    artifact, _, _ = fixture_model(tmp_path/'model', dtype='F32')
    metrics, detail = profile(monkeypatch, native, artifact, policy=policy, scalar=True)
    assert metrics['avx2'] is False
    for phase in ('prefill', 'decode'):
        assert detail[phase]['packs'] == detail[phase]['reuses'] == 0


@pytest.mark.parametrize('dtype', ['F32', 'BF16'])
@pytest.mark.parametrize('policy', ['fast', 'accurate'])
@pytest.mark.parametrize('mode', ['verify', 'chunked'])
@pytest.mark.parametrize('threads', [1, 2])
def test_old_new_logits_exact_across_changed_requests_cache_workers_and_source_storage(
        native, baseline, tmp_path, monkeypatch, dtype, policy, mode, threads):
    before, stored, _ = fixture_model(tmp_path/'model', dtype=dtype)
    monkeypatch.delenv('LEAF_DECODER_PROFILE', raising=False)
    monkeypatch.delenv('LEAF_EXPERIMENTAL_FLOAT_TILES', raising=False)
    # Repeated lengths with different tokens exercise reused source buffers.
    # A nine-token cache chunk also keeps the packed path active across chunks.
    sequences = [list(range(5, 14)), list(range(45, 62)), list(range(83, 74, -1))]
    artifacts = [before, stored] if dtype == 'BF16' else [before]
    for artifact in artifacts:
        kwargs = dict(mode=mode, threads=threads, fp32_math=policy, chunk=9)
        reference, _ = run_native(baseline, artifact, sequences, **kwargs)
        result, _ = run_native(native, artifact, sequences, **kwargs)
        assert np.array_equal(reference.view('<u4'), result.view('<u4'))


def calibration(snapshot, path):
    plan = load_snapshot_plan(snapshot)
    arrays = {name: np.linspace(0.25, 1.5, shape[1], dtype=np.float32)
              for name, shape in tensor_shapes(plan['config']).items()
              if shape[0] > 1 and not name.startswith(('model.embed_tokens.', 'model.position_embeddings.'))}
    metadata = dict(format='leaf-activation-calibration-v1', dataset_sha256='1'*64,
        model_config_sha256=digest(snapshot/'config.json'),
        source_weight_sha256={p.name: digest(p) for p in snapshot.glob('*.safetensors')})
    np.savez(path, **arrays, __metadata__=json.dumps(metadata))


@pytest.mark.parametrize('smoothed', [False, True])
def test_protected_fp32_inside_mixed_artifacts_keeps_old_packing_dispatch(
        native, baseline, tmp_path, monkeypatch, smoothed):
    original, _, _ = fixture_model(tmp_path/'model', dtype='F32', tied=False)
    artifact = tmp_path/'mixed.leaf'
    kept = ['model.layers.0.mlp.up_proj.weight', 'model.layers.0.mlp.gate_proj.weight']
    options = {}
    if smoothed:
        path = tmp_path/'calibration.npz'
        calibration(original.parent, path)
        options['calibration_path'] = path
    export_decoder(original.parent, artifact, bits=8, keep_fp32_tensors=kept, **options)
    # Explicit research dispatch exercises protected FP32 panels even though
    # this mixed artifact does not qualify for the ordinary optimized FP32 stack.
    monkeypatch.setenv('LEAF_EXPERIMENTAL_FLOAT_TILES', '1')
    metrics, detail = profile(monkeypatch, native, artifact)
    assert metrics['optimized_fp32_active'] is False
    assert detail['prefill']['packs'] == (6 if metrics['avx2'] else 0)
    assert detail['prefill']['reuses'] == 0
    assert detail['decode']['packs'] == detail['decode']['reuses'] == 0
    sequences = [list(range(5, 14)), list(range(45, 62)), list(range(83, 74, -1))]
    a, _ = run_native(baseline, artifact, sequences, activation_bits=32)
    b, _ = run_native(native, artifact, sequences, activation_bits=32)
    assert np.array_equal(a.view('<u4'), b.view('<u4'))
