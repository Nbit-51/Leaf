"""Accuracy selection must be explicit, early, and honored by the native path."""
import json
import os
from pathlib import Path
import subprocess
import sys
import io
from types import SimpleNamespace

import numpy as np
import pytest

from leaf import cli, runtime_check
from tools.decoder_validation import run_native


def args(**changes):
    values=dict(worker_cpus=None,threads=1,cpu=None,source_bf16_head=False,
                model='owner/model',offline=True,plan=None,bits='auto',activation_bits='auto',fp32_math='accurate')
    values.update(changes)
    return SimpleNamespace(**values)


def test_cli_parser_exposes_explicit_accuracy(monkeypatch):
    captured=[]
    monkeypatch.setattr(cli,'run',captured.append)
    monkeypatch.setattr(sys,'argv',['leaf','run','owner/model','--fp32-math','accurate'])
    assert cli.main()==0 and captured[0].fp32_math=='accurate'


def test_evaluate_parser_exposes_the_same_accuracy_policy(monkeypatch):
    from leaf import evaluation
    captured=[]
    monkeypatch.setattr(evaluation,'evaluate',captured.append)
    monkeypatch.setattr(sys,'argv',['leaf','evaluate','owner/model','--dataset','data.txt',
        '--dataset-source','fixture','--output','unused','--fp32-math','accurate'])
    assert cli.main()==0 and captured[0].fp32_math=='accurate'


def test_cli_accuracy_bypasses_quantized_auto_profiles_and_forwards_policy(monkeypatch,tmp_path):
    captured=[]
    monkeypatch.setattr(cli,'cache_root',lambda:tmp_path)
    monkeypatch.setattr(cli,'executable',lambda *a:tmp_path/'native.exe')
    monkeypatch.setattr(runtime_check,'check',lambda *a:{'fp32_math_modes':['fast','accurate']})
    def resolve(*a,**kw):
        kw['before_weights']()
        return tmp_path
    monkeypatch.setattr(cli,'resolve_model',resolve)
    monkeypatch.setattr(cli,'model_directory',lambda *a:tmp_path)
    monkeypatch.setattr(cli,'prepare',lambda s,d,b,p:captured.append(b) or tmp_path/'model.leaf')
    monkeypatch.setattr(cli,'select_configuration',lambda *a:pytest.fail('Existing quantization profiles cannot select this math policy'))
    tokenizer=SimpleNamespace(decode=lambda *a,**k:'ok')
    monkeypatch.setattr(cli,'encode_prompt',lambda *a:(tokenizer,[1,2]))
    class Process:
        def __init__(self,command,**kwargs):
            assert command[-1]=='--fp32-math=accurate' and command[12]=='32'
            Path(command[4]).write_text(json.dumps({'fp32_math':'accurate','decode_p50_ms':1}))
            self.stdout=io.StringIO('TOKEN 3\n');self.stderr=io.StringIO('')
        def wait(self):return 0
    monkeypatch.setattr(cli.subprocess,'Popen',Process)
    report=tmp_path/'metrics.json'
    cli.run(args(prompt='hello',raw=True,max_tokens=2,metrics=report))
    assert captured==[32] and json.loads(report.read_text())['fp32_math']=='accurate'


@pytest.mark.parametrize('weight,activation',[('8','auto'),('4','32'),('auto','8'),('32','8')])
def test_quantized_requests_fail_before_download_or_build(monkeypatch,weight,activation):
    monkeypatch.setattr(cli,'cache_root',lambda:pytest.fail('Must reject before setup'))
    with pytest.raises(ValueError,match='unquantized'):
        cli.run(args(bits=weight,activation_bits=activation))


@pytest.mark.parametrize('info',[{}, {'fp32_math_modes':['fast']}, {'fp32_math_modes':[]},
    {'fp32_math_modes':None}, {'fp32_math_modes':'accurate'}])
def test_old_binary_rejected_before_weights(monkeypatch,tmp_path,info):
    monkeypatch.setattr(cli,'cache_root',lambda:tmp_path)
    monkeypatch.setattr(cli,'executable',lambda *a:tmp_path/'old.exe')
    monkeypatch.setattr(runtime_check,'check',lambda *a:info)
    def resolve(*a,**kw):
        kw['before_weights']()
        pytest.fail('Weights must not download')
    monkeypatch.setattr(cli,'resolve_model',resolve)
    with pytest.raises(RuntimeError,match='does not support --fp32-math accurate'):
        cli.run(args())


@pytest.mark.parametrize('cpus',[None,[4,2]])
@pytest.mark.parametrize('reported',['accurate','fast',None])
def test_protocol_is_unambiguous_and_policy_is_verified(monkeypatch,tmp_path,cpus,reported):
    monkeypatch.setattr('leaf.affinity.validate_worker_cpus',lambda x,t:list(x))
    def invoke(command,**kwargs):
        expected=['32']+(['4,2'] if cpus else [])+['--fp32-math=accurate']
        assert command[12:]==expected
        Path(command[3]).write_bytes(np.array([1,2],dtype='<f4').tobytes())
        Path(command[4]).write_text(json.dumps({'fp32_math':reported,'worker_cpus':cpus}))
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr('tools.decoder_validation.subprocess.run',invoke)
    kw=dict(fp32_math='accurate',worker_cpus=cpus,threads=2 if cpus else 1)
    if reported=='accurate':
        result,_=run_native(tmp_path/'native',tmp_path/'artifact',[[1,2]],**kw)
        assert result.tolist()==[1,2]
    else:
        with pytest.raises(ValueError,match='math policy differs'):
            run_native(tmp_path/'native',tmp_path/'artifact',[[1,2]],**kw)


@pytest.mark.parametrize('tail',[['--fp32-math=typo'],['--fp32-math=accurate','--fp32-math=fast']])
def test_native_rejects_bad_policy_before_artifact_load(tail,tmp_path):
    binary=os.environ.get('LEAF_TEST_DECODER_BIN')
    if not binary:pytest.skip('Requires rebuilt native decoder')
    command=[binary,'missing-model','missing-request','out','metrics','verify','1','1','0','1','0','1','32',*tail]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode and ('FP32 math' in result.stderr)


def test_real_native_accurate_scalar_cache_and_metadata(tmp_path):
    binary=os.environ.get('LEAF_TEST_DECODER_BIN')
    if not binary:pytest.skip('Requires rebuilt native decoder')
    from test_decoder_generation_format import fixture_model
    artifact,_=fixture_model(tmp_path/'model',None)
    a,m=run_native(Path(binary),artifact,[[1,2,3]],fp32_math='accurate')
    b,n=run_native(Path(binary),artifact,[[1,2,3]],fp32_math='accurate',scalar=True,mode='chunked',chunk=1)
    np.testing.assert_allclose(a,b,rtol=2e-4,atol=2e-4)
    assert m['fp32_math']==n['fp32_math']=='accurate'


def test_native_does_not_silently_apply_accuracy_policy_to_quantized_weights(tmp_path):
    binary=os.environ.get('LEAF_TEST_DECODER_BIN')
    if not binary:pytest.skip('Requires rebuilt native decoder')
    from test_decoder_generation_format import fixture_model
    from tools.export_decoder import export_decoder
    artifact,_=fixture_model(tmp_path/'model',None)
    quantized=tmp_path/'quantized.leaf'
    export_decoder(artifact.parent,quantized,8,group_size=8)
    with pytest.raises(RuntimeError,match='requires FP32 compute and unquantized weights'):
        run_native(Path(binary),quantized,[[1,2]],fp32_math='accurate')
