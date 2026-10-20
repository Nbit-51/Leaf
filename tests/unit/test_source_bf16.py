"""Source-preserving storage, version rejection and real decoder parity."""
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from tools.decoder_plan import load_snapshot_plan, tensor_shapes
from tools.decoder_validation import run_native
from tools.export_decoder import DESCRIPTOR, HEADER, export_decoder


def fixture_model(root, dtype='BF16', tied=True, eos=2, window=False, width=16):
    root.mkdir()
    config = dict(model_type='mistral' if window else 'llama', hidden_size=width,
        intermediate_size=width*2, num_attention_heads=2, num_hidden_layers=1,
        vocab_size=101, max_position_embeddings=128, eos_token_id=eos, tie_word_embeddings=tied)
    if window:
        config['sliding_window'] = 3
    (root/'config.json').write_text(json.dumps(config))
    plan = load_snapshot_plan(root)
    random = np.random.default_rng(591)
    entries, chunks, offset = {}, [], 0
    for name, shape in tensor_shapes(plan['config']).items():
        source = plan['tensors'][name]['source']
        if source in entries:
            continue
        values = random.uniform(-0.3,0.3,shape).astype('<f4')
        if 'norm.weight' in name:
            values = (values/10+1).astype('<f4')
        if dtype == 'BF16':
            payload = (values.view('<u4') >> 16).astype('<u2').tobytes()
        else:
            payload = values.tobytes()
        entries[source] = dict(dtype=dtype, shape=list(shape), data_offsets=[offset,offset+len(payload)])
        chunks.append(payload)
        offset += len(payload)
    header = json.dumps(entries).encode()
    (root/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b''.join(chunks))
    before, after = root/'before.leaf', root/'after.leaf'
    export_decoder(root,before)
    report = export_decoder(root,after,source_bf16_head=True)
    return before, after, report


def descriptors(blob):
    version, count = HEADER.unpack_from(blob)[1:3]
    position = HEADER.size
    if version >= 4:
        position += 4 + 4*struct.unpack_from('<I',blob,position)[0]
    result = {}
    for _ in range(count):
        size = struct.unpack_from('<I',blob,position)[0]
        position += 4
        name = blob[position:position+size].decode()
        position += size
        result[name] = (position,DESCRIPTOR.unpack_from(blob,position))
        position += DESCRIPTOR.size
    return result


@pytest.mark.parametrize('tied',[False,True])
def test_export_compacts_source_bf16_and_shares_tied_payload(tmp_path,tied):
    before, after, report = fixture_model(tmp_path/'model',tied=tied)
    original, compact = before.read_bytes(), after.read_bytes()
    assert HEADER.unpack_from(compact)[:2] == (b'LEAFDC05',5)
    rows = descriptors(compact)
    names = ['lm_head.weight','model.embed_tokens.weight'] if tied else ['lm_head.weight']
    assert set(report['source_bf16_tensors']) == set(names)
    assert report['weight_bits'] == report['compute_bits'] == 32
    base = HEADER.unpack_from(compact)[3]
    old_base = HEADER.unpack_from(original)[3]
    old = descriptors(original)
    for name, (_,d) in rows.items():
        expected_bits = 16 if name in names else 32
        assert d[3] == expected_bits and d[4] == 0 and d[-1] == 0
        decoded = np.frombuffer(compact,dtype='<u2' if expected_bits==16 else '<f4',
                                count=d[1]*d[2],offset=base+d[5])
        if expected_bits==16:
            decoded = (decoded.astype('<u4')<<16).view('<f4')
        expected = np.frombuffer(original,dtype='<f4',count=d[1]*d[2],offset=old_base+old[name][1][5])
        assert np.array_equal(decoded.view('<u4'),expected.view('<u4'))
    assert len(compact) < len(original)
    if tied:
        assert rows['lm_head.weight'][1][5:] == rows['model.embed_tokens.weight'][1][5:]


def test_f32_sources_keep_exact_existing_artifact(tmp_path):
    before,after,report=fixture_model(tmp_path/'model',dtype='F32')
    assert before.read_bytes() == after.read_bytes()
    assert report['source_bf16_tensors'] == []


def test_protected_tied_alias_prevents_duplicate_storage(tmp_path):
    before,_,_=fixture_model(tmp_path/'model')
    after=tmp_path/'protected.leaf'
    report=export_decoder(before.parent,after,source_bf16_head=True,
                          keep_fp32_tensors=['model.embed_tokens.weight'])
    assert report['source_bf16_tensors'] == []
    assert before.read_bytes() == after.read_bytes()


@pytest.mark.parametrize('bits',[4,8])
def test_storage_option_rejects_quantized_compute(tmp_path,bits):
    with pytest.raises(ValueError,match='FP32 computation'):
        export_decoder(tmp_path,tmp_path/'bad.leaf',bits=bits,source_bf16_head=True)


@pytest.fixture(scope='module')
def native():
    value=os.environ.get('LEAF_TEST_DECODER_BIN')
    if not value:
        pytest.skip('Set LEAF_TEST_DECODER_BIN for native storage checks')
    return Path(value).resolve()


@pytest.mark.parametrize('tokens',[1,2,3,4,7,8,9,17])
@pytest.mark.parametrize('scalar,threads',[(False,1),(False,2),(True,1)])
@pytest.mark.parametrize('width',[24,28])
def test_native_storage_changes_no_logit_bits(native,tmp_path,tokens,scalar,threads,width):
    before,after,_=fixture_model(tmp_path/'model',width=width)
    ids=[3+i%19 for i in range(tokens)]
    a,am=run_native(native,before,[ids],scalar=scalar,threads=threads)
    b,bm=run_native(native,after,[ids],scalar=scalar,threads=threads)
    assert np.array_equal(a.view('<u4'),b.view('<u4'))
    assert am['source_bf16_tensors']==0 and bm['source_bf16_tensors']==2
    assert bm['source_bf16_linear_calls']>0
    assert am['optimized_fp32_active']==bm['optimized_fp32_active']


@pytest.mark.parametrize('eos,window',[(None,False),(2,False),([2,3],False),([2,3],True)])
def test_generation_chunked_cache_and_eos_compose(native,tmp_path,eos,window):
    before,after,_=fixture_model(tmp_path/'model',eos=eos,window=window)
    for mode in ('generate','chunked'):
        a,am=run_native(native,before,[[5,6,7,8,9,10,11,12,13]],mode=mode,generate=5,chunk=3)
        b,bm=run_native(native,after,[[5,6,7,8,9,10,11,12,13]],mode=mode,generate=5,chunk=3)
        assert am['generated_tokens']==bm['generated_tokens']
        assert np.array_equal(a.view('<u4'),b.view('<u4'))


@pytest.mark.parametrize('mutation',['old_magic','quantized','group','scales','size','offset','wrong_tensor','eos'])
def test_reader_rejects_invalid_storage_metadata(native,tmp_path,mutation):
    _,after,_=fixture_model(tmp_path/'model')
    blob=bytearray(after.read_bytes())
    rows=descriptors(blob)
    p,_=rows['lm_head.weight']
    if mutation=='old_magic':
        blob[:8]=b'LEAFDC04'; struct.pack_into('<I',blob,8,4)
    elif mutation=='quantized': struct.pack_into('<I',blob,p+12,8)
    elif mutation=='group': struct.pack_into('<I',blob,p+16,16)
    elif mutation=='scales': struct.pack_into('<Q',blob,p+44,1)
    elif mutation=='size': struct.pack_into('<Q',blob,p+28,1)
    elif mutation=='offset': struct.pack_into('<Q',blob,p+20,1)
    elif mutation=='wrong_tensor':
        p,_=rows['model.norm.weight']; struct.pack_into('<I',blob,p+12,16)
    else: struct.pack_into('<I',blob,HEADER.size,0)
    after.write_bytes(blob)
    with pytest.raises(RuntimeError,match='Native decoder failed'):
        run_native(native,after,[[5,6]])


def test_reader_rejects_integer_activation_request(native,tmp_path):
    _,after,_=fixture_model(tmp_path/'model')
    with pytest.raises(RuntimeError,match='FP32 activations'):
        run_native(native,after,[[5,6]],activation_bits=8)


def test_previous_runtime_rejects_new_storage(tmp_path):
    value=os.environ.get('LEAF_TEST_LEGACY_DECODER_BIN')
    if not value:
        pytest.skip('Set LEAF_TEST_LEGACY_DECODER_BIN')
    _,after,_=fixture_model(tmp_path/'model')
    with pytest.raises(RuntimeError,match='unsupported artifact magic'):
        run_native(Path(value).resolve(),after,[[5,6]])


def test_cli_source_storage_is_separate_from_regular_fp32_cache(tmp_path):
    from leaf import cli
    before,_,_=fixture_model(tmp_path/'model')
    directory=tmp_path/'cache'
    regular=cli.prepare(before.parent,directory,32)
    compact=cli.prepare(before.parent,directory,32,source_bf16_head=True)
    assert regular!=compact
    assert regular.read_bytes()==before.read_bytes()
    assert compact.read_bytes()[:8]==b'LEAFDC05'
    assert cli.prepare(before.parent,directory,32,source_bf16_head=True)==compact


def test_cli_f32_fallback_reuses_regular_cache(tmp_path):
    from leaf import cli
    before,_,_=fixture_model(tmp_path/'model',dtype='F32')
    directory=tmp_path/'cache'
    regular=cli.prepare(before.parent,directory,32)
    assert cli.prepare(before.parent,directory,32,source_bf16_head=True)==regular
    assert not (directory/'decoder-32-source-bf16.leaf').exists()


@pytest.mark.parametrize('bits,activations',[('8','auto'),('4','32'),('32','8')])
def test_cli_rejects_incompatible_storage_before_downloading(tmp_path,monkeypatch,bits,activations):
    from types import SimpleNamespace
    from leaf import cli
    monkeypatch.setattr(cli,'resolve_model',lambda *a,**k: pytest.fail('Downloaded before validating storage policy'))
    with pytest.raises(ValueError,match='FP32'):
        cli.run(SimpleNamespace(cpu=None,source_bf16_head=True,bits=bits,activation_bits=activations))


def test_real_cli_prepares_and_runs_source_storage(native,tmp_path,monkeypatch,capsys):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from leaf import cli
    before,_,_=fixture_model(tmp_path/'model')
    tokenizer=Tokenizer(models.WordLevel({'[UNK]':0,**{f'tok{i}':i for i in range(1,101)}},unk_token='[UNK]'))
    tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    tokenizer.save(str(before.parent/'tokenizer.json'))
    monkeypatch.setenv('LEAF_CACHE_DIR',str(tmp_path/'cli-cache'))
    monkeypatch.setenv('LEAF_DECODER_BIN',str(native))
    metrics=tmp_path/'metrics.json'
    monkeypatch.setattr('sys.argv',['leaf','run',str(before.parent),'--offline','--bits','32',
        '--source-bf16-head','--raw','--prompt','tok5 tok6 tok7','--max-tokens','2','--metrics',str(metrics)])
    cli.main()
    saved=json.loads(metrics.read_text())
    assert saved['source_bf16_tensors']==2 and saved['source_bf16_linear_calls']>0
    assert saved['weight_bits']==saved['activation_bits']==32
    assert 'FP32 compute, source BF16 storage' in capsys.readouterr().err


def test_cli_rejects_old_runtime_before_downloading_weights(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from leaf import cli, runtime_check
    monkeypatch.setattr(cli,'cache_root',lambda:tmp_path)
    monkeypatch.setattr(cli,'executable',lambda *a:tmp_path/'old.exe')
    monkeypatch.setattr(runtime_check.subprocess,'run',lambda *a,**k:SimpleNamespace(
        stdout='{"runtime":"leaf_decoder","artifact_versions":[2,3,4]}',returncode=0))
    def resolve(*a,**k):
        k['before_weights']()
        pytest.fail('Downloaded weights despite incompatible runtime')
    monkeypatch.setattr(cli,'resolve_model',resolve)
    with pytest.raises(RuntimeError,match='Runtime check failed'):
        cli.run(SimpleNamespace(cpu=None,source_bf16_head=True,bits='32',activation_bits='32',
                                model='owner/model',offline=False,plan=None))
