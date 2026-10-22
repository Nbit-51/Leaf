"""Placement must fail before weight downloads and preserve native protocol."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from leaf import cli, affinity, runtime_check
from tools.decoder_validation import run_native


def test_cli_forwards_ordered_cpu_list(monkeypatch):
    captured=[]
    monkeypatch.setattr(cli,'run',captured.append)
    monkeypatch.setattr(sys,'argv',['leaf','run','owner/model','--threads','2','--worker-cpus','4','2'])
    assert cli.main()==0
    assert captured[0].worker_cpus==[4,2]


def test_old_runtime_is_rejected_before_weights(monkeypatch,tmp_path):
    monkeypatch.setattr(affinity,'validate_worker_cpus',lambda *args:[0])
    monkeypatch.setattr(cli,'cache_root',lambda:tmp_path)
    monkeypatch.setattr(cli,'executable',lambda *args:tmp_path/'old.exe')
    monkeypatch.setattr(runtime_check,'check',lambda *args:{'runtime':'leaf_decoder'})
    def resolve(*args,**kwargs):
        kwargs['before_weights']()
        pytest.fail('Must fail before model weights')
    monkeypatch.setattr(cli,'resolve_model',resolve)
    args=SimpleNamespace(worker_cpus=[0],threads=1,cpu=None,source_bf16_head=False,
        model='owner/model',offline=False,plan=None)
    with pytest.raises(RuntimeError,match='does not support --worker-cpus'):
        cli.run(args)


@pytest.mark.parametrize('requested,reported', [([4,2],[4,2]),([4,2],[2,4]),([4,2],None)])
def test_native_request_places_cpu_list_after_activation_bits(monkeypatch,tmp_path,requested,reported):
    monkeypatch.setattr(affinity,'validate_worker_cpus',lambda cpus,threads:list(cpus))
    def invoke(command,**kwargs):
        assert command[-2:]==['32','4,2']
        Path(command[3]).write_bytes(np.array([1,2],dtype='<f4').tobytes())
        Path(command[4]).write_text(json.dumps({'worker_cpus':reported}))
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr('tools.decoder_validation.subprocess.run',invoke)
    if reported==requested:
        result,metrics=run_native(tmp_path/'native',tmp_path/'model',[[1,2]],threads=2,worker_cpus=requested)
        assert result.tolist()==[1,2] and metrics['worker_cpus']==requested
    else:
        with pytest.raises(ValueError,match='policy differs'):
            run_native(tmp_path/'native',tmp_path/'model',[[1,2]],threads=2,worker_cpus=requested)


@pytest.mark.parametrize('text',['','0,','0,,1','-1','true','0,1x','4294967296'])
def test_real_native_rejects_malformed_cpu_csv_before_artifact_load(tmp_path,text):
    import subprocess
    configured=os.environ.get('LEAF_TEST_DECODER_BIN')
    if not configured: pytest.skip('Set LEAF_TEST_DECODER_BIN for native protocol checks')
    info=json.loads(subprocess.check_output([configured,'--runtime-info'],text=True))
    if not info.get('worker_affinity_supported'): pytest.skip('Explicit placement unsupported')
    result=subprocess.run([configured,str(tmp_path/'missing.leaf'),'tokens','out','metrics',
        'verify','1','1','0','1','0','1','32',text],capture_output=True,text=True)
    assert result.returncode!=0 and 'cannot open artifact' not in result.stderr


@pytest.mark.parametrize('threads,placed', [(1,False),(1,True),(2,False),(2,True)])
def test_real_native_reports_wait_policy_and_preserves_logits(tmp_path,threads,placed):
    import importlib.util
    import subprocess
    import psutil
    configured=os.environ.get('LEAF_TEST_DECODER_BIN')
    if not configured: pytest.skip('Set LEAF_TEST_DECODER_BIN for native policy checks')
    info=json.loads(subprocess.check_output([configured,'--runtime-info'],text=True))
    if placed and not info.get('worker_affinity_supported'):
        pytest.skip('Explicit placement unsupported')
    allowed=psutil.Process().cpu_affinity() if placed else []
    if placed and len(allowed)<threads: pytest.skip('Insufficient allowed CPUs')
    spec=importlib.util.spec_from_file_location('worker_policy_fixture',Path(__file__).with_name('test_source_bf16.py'))
    fixture=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    artifact,_,_=fixture.fixture_model(tmp_path/'model')
    reference,_=run_native(Path(configured),artifact,[[1,3,5]],threads=1)
    cpus=allowed[:threads] if placed else None
    result,metrics=run_native(Path(configured),artifact,[[1,3,5]],threads=threads,worker_cpus=cpus)
    assert np.array_equal(reference.view('<u4'),result.view('<u4'))
    assert metrics['worker_spin_us']==(50 if placed and threads>1 else 0)
