import pytest

from tools import ollama_validation as validation


def test_generation_request_keeps_matched_greedy_cpu_work():
    request = validation.generation_request('local-model','raw input')
    assert request['raw'] is True and request['stream'] is False
    options = request['options']
    assert (options['num_gpu'], options['num_thread'], options['num_predict']) == (0,1,2)
    assert options['temperature'] == 0 and options['repeat_penalty'] == 1
    assert options['stop'] == []


def test_failed_server_policy_setup_still_cleans_owned_process(monkeypatch,tmp_path):
    calls=[]
    class Socket:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def bind(self,address): pass
        def getsockname(self): return ('127.0.0.1',54321)

    class Child:
        pid=12345
        returncode=None
        def poll(self): return self.returncode
        def children(self,**kwargs): return []
        def cpu_affinity(self,value): raise RuntimeError('policy setup failed')
        def terminate(self): self.returncode=0; calls.append('terminate')
        def wait(self,**kwargs): calls.append('wait'); return self.returncode

    child=Child()
    monkeypatch.setattr(validation.socket,'socket',Socket)
    monkeypatch.setattr(validation.subprocess,'Popen',lambda *a,**k:child)
    monkeypatch.setattr(validation.subprocess,'CREATE_NO_WINDOW',0,raising=False)
    monkeypatch.setattr(validation.psutil,'Process',lambda pid:child)
    monkeypatch.setattr(validation.psutil,'wait_procs',lambda *a,**k:([child],[]))
    with pytest.raises(RuntimeError,match='policy setup failed'):
        with validation.local_server('unused.exe',tmp_path/'models',tmp_path/'server',cpu=2):
            pytest.fail('Server setup should fail before yielding')
    assert calls==['terminate','wait']
