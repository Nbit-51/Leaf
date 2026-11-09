"""Original BF16 matrix storage preserves source bits, aliases and native outputs."""
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from test_source_bf16 import descriptors, fixture_model
from tools.decoder_plan import load_snapshot_plan, tensor_shapes
from tools.decoder_validation import run_native
from tools.export_decoder import (HEADER, export_decoder, source_bf16_available,
                                  tensor_index, transform_array)


def source_arrays(root):
    filename = root/'model.safetensors'
    base, entries = tensor_index(filename)
    result = {}
    for name, entry in entries.items():
        dtype = {'BF16': '<u2', 'F16': '<f2', 'F32': '<f4'}[entry['dtype']]
        start, end = entry['data_offsets']
        with filename.open('rb') as stream:
            stream.seek(base+start)
            raw = stream.read(end-start)
        result[name] = (entry['dtype'], np.frombuffer(raw, dtype=dtype).reshape(entry['shape']).copy())
    return result


def write_sources(root, arrays):
    entries, chunks, offset = {}, [], 0
    for name, (dtype, array) in arrays.items():
        payload = np.ascontiguousarray(array).tobytes()
        entries[name] = dict(dtype=dtype, shape=list(array.shape), data_offsets=[offset, offset+len(payload)])
        chunks.append(payload)
        offset += len(payload)
    header = json.dumps(entries).encode()
    (root/'model.safetensors').write_bytes(struct.pack('<Q', len(header))+header+b''.join(chunks))


def fp32(dtype, values):
    return ((values.astype('<u4') << 16).view('<f4') if dtype == 'BF16'
            else values.astype('<f4'))


def decoded(blob, descriptor):
    _, rows, columns, bits, _, offset, _, _, _ = descriptor
    base = HEADER.unpack_from(blob)[3]
    values = np.frombuffer(blob, dtype='<u2' if bits == 16 else '<f4',
                           count=rows*columns, offset=base+offset)
    return (values.astype('<u4') << 16).view('<f4') if bits == 16 else values


def assert_export_values(before, after, compact):
    original, stored = before.read_bytes(), after.read_bytes()
    old, current = descriptors(original), descriptors(stored)
    for name, (_, row) in current.items():
        assert row[3] == (16 if name in compact else 32)
        assert row[4] == row[-1] == 0
        assert np.array_equal(decoded(stored, row).view('<u4'), decoded(original, old[name][1]).view('<u4'))


@pytest.mark.parametrize('tied', [False, True])
def test_all_original_bf16_matrices_compact_but_norms_stay_fp32(tmp_path, tied):
    before, head, head_report = fixture_model(tmp_path/'model', tied=tied)
    after = tmp_path/'matrices.leaf'
    report = export_decoder(before.parent, after, source_bf16=True)
    shapes = tensor_shapes(load_snapshot_plan(before.parent)['config'])
    compact = {name for name, shape in shapes.items() if shape[0] > 1}
    assert HEADER.unpack_from(after.read_bytes())[:2] == (b'LEAFDC06', 6)
    assert set(report['source_bf16_tensors']) == compact
    assert report['compute_bits'] == report['weight_bits'] == 32
    assert report['source_bf16_scope'] == 'matrices'
    assert source_bf16_available(before.parent)
    assert_export_values(before, after, compact)
    assert after.stat().st_size < head.stat().st_size < before.stat().st_size
    # The existing narrower API still has exactly its original v5 scope.
    assert head_report['format'] == 'leaf-decoder-v5'
    repeated = tmp_path/'head-again.leaf'
    export_decoder(before.parent, repeated, source_bf16_head=True)
    assert repeated.read_bytes() == head.read_bytes()
    if tied:
        rows = descriptors(after.read_bytes())
        assert rows['lm_head.weight'][1][5:] == rows['model.embed_tokens.weight'][1][5:]


@pytest.mark.parametrize('dtype', ['F32', 'F16'])
def test_other_source_dtypes_never_acquire_bf16_storage_or_duplicate_artifacts(tmp_path, dtype):
    before, _, _ = fixture_model(tmp_path/'model')
    arrays = source_arrays(before.parent)
    arrays = {name: (dtype, fp32(kind, array).astype('<f4' if dtype == 'F32' else '<f2'))
              for name, (kind, array) in arrays.items()}
    write_sources(before.parent, arrays)
    export_decoder(before.parent, before)
    after = tmp_path/'matrix-option.leaf'
    report = export_decoder(before.parent, after, source_bf16=True)
    assert not source_bf16_available(before.parent)
    assert report['source_bf16_tensors'] == []
    assert before.read_bytes() == after.read_bytes()



def test_norm_only_bf16_source_is_not_a_matrix_candidate(tmp_path):
    before, _, _ = fixture_model(tmp_path/'model')
    arrays = source_arrays(before.parent)
    for name, shape in tensor_shapes(load_snapshot_plan(before.parent)['config']).items():
        if shape[0] <= 1:
            continue
        source = load_snapshot_plan(before.parent)['tensors'][name]['source']
        kind, raw = arrays[source]
        arrays[source] = ('F32', fp32(kind, raw))
    write_sources(before.parent, arrays)
    assert any(kind == 'BF16' for kind, _ in arrays.values())
    assert not source_bf16_available(before.parent)
    export_decoder(before.parent, before)
    after = tmp_path/'norm-only.leaf'
    report = export_decoder(before.parent, after, source_bf16=True)
    assert report['source_bf16_tensors'] == []
    assert before.read_bytes() == after.read_bytes()


def test_mixed_source_dtypes_preserve_unrepresentable_fp32_values(tmp_path):
    before, _, _ = fixture_model(tmp_path/'model', tied=False)
    arrays = source_arrays(before.parent)
    q = 'model.layers.0.self_attn.q_proj.weight'
    up = 'model.layers.0.mlp.up_proj.weight'
    for name, dtype in [(q, 'F32'), (up, 'F16')]:
        kind, raw = arrays[name]
        values = fp32(kind, raw)
        if dtype == 'F32':
            values = values + np.float32(0.000013)
            assert np.any(values.view('<u4') & 0xffff)
        arrays[name] = (dtype, values.astype('<f4' if dtype == 'F32' else '<f2'))
    write_sources(before.parent, arrays)
    export_decoder(before.parent, before)
    after = tmp_path/'mixed-source.leaf'
    report = export_decoder(before.parent, after, source_bf16=True)
    compact = set(report['source_bf16_tensors'])
    assert q not in compact and up not in compact
    assert 'model.layers.0.mlp.down_proj.weight' in compact
    assert_export_values(before, after, compact)


def test_protection_covers_identical_transformed_aliases(tmp_path):
    before, _, _ = fixture_model(tmp_path/'model')
    plan = load_snapshot_plan(before.parent)
    q, k = ('model.layers.0.self_attn.'+name+'.weight' for name in ('q_proj', 'k_proj'))
    source = plan['tensors'][k]['source']
    plan['tensors'][q] = dict(source='not-physically-stored', source_aliases=[source])
    plan_path = tmp_path/'plan.json'
    plan_path.write_text(json.dumps(plan))
    baseline = tmp_path/'alias-baseline.leaf'
    export_decoder(before.parent, baseline, plan_path=plan_path)
    stored = tmp_path/'aliases.leaf'
    report = export_decoder(before.parent, stored, plan_path=plan_path, source_bf16=True)
    rows = descriptors(stored.read_bytes())
    assert rows[q][1][5:] == rows[k][1][5:]
    assert_export_values(baseline, stored, set(report['source_bf16_tensors']))
    protected = tmp_path/'protected.leaf'
    report = export_decoder(before.parent, protected, plan_path=plan_path, source_bf16=True,
                            keep_fp32_tensors=[q, 'model.embed_tokens.weight'])
    assert q not in report['source_bf16_tensors'] and k not in report['source_bf16_tensors']
    assert 'lm_head.weight' not in report['source_bf16_tensors']
    assert 'model.embed_tokens.weight' not in report['source_bf16_tensors']
    assert_export_values(baseline, protected, set(report['source_bf16_tensors']))


def test_all_protected_matrices_keep_existing_artifact_bytes(tmp_path):
    before, _, _ = fixture_model(tmp_path/'model')
    names = list(tensor_shapes(load_snapshot_plan(before.parent)['config']))
    after = tmp_path/'protected-all.leaf'
    report = export_decoder(before.parent, after, source_bf16=True, keep_fp32_tensors=names)
    assert report['source_bf16_tensors'] == []
    assert after.read_bytes() == before.read_bytes()


@pytest.mark.parametrize('operation', ['transpose', 'reshape_slice'])
def test_declarative_transforms_preserve_raw_bf16_bits(tmp_path, operation):
    before, _, _ = fixture_model(tmp_path/'model')
    plan = load_snapshot_plan(before.parent)
    name = 'model.layers.0.mlp.up_proj.weight'
    source = plan['tensors'][name]['source']
    arrays = source_arrays(before.parent)
    _, raw = arrays[source]
    if operation == 'transpose':
        arrays[source] = ('BF16', raw.T.copy())
        transforms = [dict(transpose=[1, 0])]
    else:
        arrays[source] = ('BF16', np.stack([raw, raw ^ np.uint16(0x8000)]))
        transforms = [dict(slice=[0, 0, 1]), dict(reshape=list(raw.shape))]
    write_sources(before.parent, arrays)
    plan['tensors'][name]['transforms'] = transforms
    plan_path = tmp_path/'transformed-plan.json'
    plan_path.write_text(json.dumps(plan))
    baseline, after = tmp_path/'transformed-baseline.leaf', tmp_path/'transformed.leaf'
    export_decoder(before.parent, baseline, plan_path=plan_path)
    report = export_decoder(before.parent, after, plan_path=plan_path, source_bf16=True)
    assert source_bf16_available(before.parent, plan_path)
    assert_export_values(baseline, after, set(report['source_bf16_tensors']))
    blob = after.read_bytes()
    d = descriptors(blob)[name][1]
    base = HEADER.unpack_from(blob)[3]
    packed = np.frombuffer(blob, dtype='<u2', count=d[1]*d[2], offset=base+d[5])
    assert packed.tobytes() == transform_array(arrays[source][1], transforms).tobytes()


@pytest.mark.parametrize('options', [dict(bits=4), dict(bits=8), dict(calibration_path=Path('unused.npz')),
    dict(source_bf16_head=True), dict(source_bf16=1), dict(source_bf16='yes')])
def test_invalid_storage_requests_fail_before_loading_weights(tmp_path, options):
    arguments = dict(source_bf16=True)
    arguments.update(options)
    with pytest.raises(ValueError, match='Source BF16'):
        export_decoder(tmp_path, tmp_path/'invalid.leaf', **arguments)


@pytest.mark.parametrize('mutation', ['nonfinite', 'bad_range', 'wrong_shape'])
def test_raw_bf16_validation_rejects_invalid_values_and_shape(tmp_path, mutation):
    before, _, _ = fixture_model(tmp_path/'model')
    arrays = source_arrays(before.parent)
    name = 'model.layers.0.mlp.up_proj.weight'
    plan = load_snapshot_plan(before.parent)
    source = plan['tensors'][name]['source']
    dtype, raw = arrays[source]
    if mutation == 'nonfinite':
        raw.flat[0] = 0x7f80
    elif mutation == 'wrong_shape':
        arrays[source] = (dtype, raw.reshape(-1))
    write_sources(before.parent, arrays)
    if mutation == 'bad_range':
        filename = before.parent/'model.safetensors'
        base, entries = tensor_index(filename)
        blob = filename.read_bytes()
        entries[source]['data_offsets'][1] -= 2
        header = json.dumps(entries).encode()
        filename.write_bytes(struct.pack('<Q', len(header))+header+blob[base:])
    target = tmp_path/'bad.leaf'
    with pytest.raises(ValueError):
        export_decoder(before.parent, target, source_bf16=True)
    assert not target.exists() and not target.with_suffix('.leaf.partial').exists()


@pytest.fixture(scope='module')
def native():
    value = os.environ.get('LEAF_TEST_DECODER_BIN')
    if not value:
        pytest.skip('Set LEAF_TEST_DECODER_BIN for matrix-storage native parity')
    return Path(value).resolve()


@pytest.mark.parametrize('policy', ['fast', 'accurate'])
@pytest.mark.parametrize('scalar,threads', [(False, 1), (False, 2), (True, 1)])
@pytest.mark.parametrize('width', [24, 28])
@pytest.mark.parametrize('mode', ['verify', 'chunked', 'generate'])
def test_native_full_logits_cache_and_generation_match_expanded_fp32(native, tmp_path, policy, scalar, threads, width, mode):
    before, _, _ = fixture_model(tmp_path/'model', width=width)
    after = tmp_path/'all-matrices.leaf'
    report = export_decoder(before.parent, after, source_bf16=True)
    sequences = [list(range(3, 20)), list(range(23, 32))]
    kwargs = dict(mode=mode, fp32_math=policy, scalar=scalar, threads=threads, chunk=9, generate=4)
    a, am = run_native(native, before, sequences, **kwargs)
    b, bm = run_native(native, after, sequences, **kwargs)
    assert np.array_equal(a.view('<u4'), b.view('<u4'))
    assert am['generated_tokens'] == bm['generated_tokens']
    assert bm['source_bf16_tensors'] == len(report['source_bf16_tensors'])
    assert bm['source_bf16_linear_calls'] > 0


def test_previous_v5_runtime_rejects_broader_storage(tmp_path):
    value = os.environ.get('LEAF_TEST_PACK_BASELINE_BIN')
    if not value:
        pytest.skip('Set LEAF_TEST_PACK_BASELINE_BIN to verify explicit v6 rejection')
    before, _, _ = fixture_model(tmp_path/'model')
    after = tmp_path/'v6.leaf'
    export_decoder(before.parent, after, source_bf16=True)
    with pytest.raises(RuntimeError, match='unsupported artifact magic'):
        run_native(Path(value).resolve(), after, [[5, 6, 7]])


@pytest.mark.parametrize('mutation,message', [
    ('norm_bf16', 'invalid tensor'),
    ('matrix_int8', 'cannot mix quantized weights'),
    ('group', 'cannot contain quantization metadata'),
    ('scale_count', 'cannot contain quantization metadata'),
    ('input_scale_name', 'cannot contain calibrated input scales'),
    ('downgraded_v5', 'invalid tensor'),
])
def test_v6_reader_rejects_invalid_storage_semantics(native, tmp_path, mutation, message):
    before, _, _ = fixture_model(tmp_path/'model')
    after = tmp_path/'invalid-v6.leaf'
    export_decoder(before.parent, after, source_bf16=True)
    blob = bytearray(after.read_bytes())
    rows = descriptors(blob)
    matrix, _ = rows['model.layers.0.mlp.up_proj.weight']
    norm_name = 'model.norm.weight'
    norm, _ = rows[norm_name]
    if mutation == 'norm_bf16':
        struct.pack_into('<I', blob, norm+12, 16)
    elif mutation == 'matrix_int8':
        struct.pack_into('<I', blob, matrix+12, 8)
    elif mutation == 'group':
        struct.pack_into('<I', blob, matrix+16, 8)
    elif mutation == 'scale_count':
        struct.pack_into('<Q', blob, matrix+44, 1)
    elif mutation == 'input_scale_name':
        suffix = '.input_scale'
        replacement = 'x'*(len(norm_name)-len(suffix))+suffix
        assert len(replacement) == len(norm_name)
        blob[norm-len(norm_name):norm] = replacement.encode()
    else:
        blob[:8] = b'LEAFDC05'
        struct.pack_into('<I', blob, 8, 5)
    after.write_bytes(blob)
    with pytest.raises(RuntimeError, match=message):
        run_native(native, after, [[5, 6, 7]])


def test_v6_reader_requires_fp32_activations(native, tmp_path):
    before, _, _ = fixture_model(tmp_path/'model')
    after = tmp_path/'fp32-activation-only.leaf'
    export_decoder(before.parent, after, source_bf16=True)
    with pytest.raises(RuntimeError, match='requires FP32 activations'):
        run_native(native, after, [[5, 6, 7]], activation_bits=8)
