"""The independent payload audit rejects corrupt or mismatched storage evidence."""
import struct

import pytest

from test_source_bf16 import descriptors, fixture_model
from tools.export_decoder import HEADER, export_decoder
from tools.validate_source_bf16_matrices import audit_payloads


@pytest.fixture
def payloads(tmp_path):
    before, _, _ = fixture_model(tmp_path/'model')
    after = tmp_path/'all-matrices.leaf'
    report = export_decoder(before.parent, after, source_bf16=True)
    return before, after, report


def test_audit_checks_all_values_and_configuration_without_mutating_artifacts(payloads):
    before, after, exported = payloads
    old, stored = before.read_bytes(), after.read_bytes()
    result = audit_payloads(before, after, exported)
    assert result['exact'] is True
    assert result['values'] == 5840
    assert result['source_bf16_tensors'] == 9
    assert result['tensors'] == 12
    assert result['vocabulary'] == 101
    assert result['config_and_eos_exact'] is True
    assert set(result['bf16_names']) == set(exported['source_bf16_tensors'])
    assert result['before_bytes'] == len(old) and result['after_bytes'] == len(stored)
    assert 0 < result['bytes_ratio'] < 1
    assert before.read_bytes() == old and after.read_bytes() == stored


@pytest.mark.parametrize('kind', ['bf16_payload', 'fp32_payload', 'config', 'eos', 'missing_descriptor',
                                  'report_omission', 'report_extra', 'old_storage_version'])
def test_audit_rejects_tampered_payload_or_metadata(payloads, kind):
    before, after, exported = payloads
    blob = bytearray(after.read_bytes())
    rows = descriptors(blob)
    if kind in ('bf16_payload', 'fp32_payload'):
        name = ('model.layers.0.mlp.up_proj.weight' if kind == 'bf16_payload' else 'model.norm.weight')
        row = rows[name][1]
        base = HEADER.unpack_from(blob)[3]
        blob[base+row[5]] ^= 1
    elif kind == 'config':
        struct.pack_into('<I', blob, 24, 17)
    elif kind == 'eos':
        struct.pack_into('<I', blob, HEADER.size+4, 3)
    elif kind == 'missing_descriptor':
        count = HEADER.unpack_from(blob)[2]
        struct.pack_into('<I', blob, 12, count-1)
    elif kind == 'report_omission':
        exported['source_bf16_tensors'] = exported['source_bf16_tensors'][1:]
    elif kind == 'report_extra':
        exported['source_bf16_tensors'] = [*exported['source_bf16_tensors'], 'model.norm.weight']
    else:
        blob[:8] = b'LEAFDC05'
        struct.pack_into('<I', blob, 8, 5)
    after.write_bytes(blob)
    with pytest.raises(ValueError):
        audit_payloads(before, after, exported)


def test_audit_rejects_missing_tensor_payload_bytes(payloads):
    before, after, exported = payloads
    after.write_bytes(after.read_bytes()[:-1])
    with pytest.raises(ValueError, match='payload range'):
        audit_payloads(before, after, exported)
