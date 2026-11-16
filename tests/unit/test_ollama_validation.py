import pytest

from tools import ollama_validation as validation
from tools.benchmark_ollama_decoder import check_generation_work, check_selected_logprobs


@pytest.mark.parametrize("reported,expected", [
    ([-0.1, -0.2], True),
    ([-0.1], False),
    ([], False),
    ([-0.1, -0.2003], False),
    ([-0.1, float("nan")], False),
])
def test_selected_logprobs_require_every_token_and_unchanged_tolerance(reported, expected):
    leaf = {"selected_logprobs": [-0.1, -0.2]}
    ollama = {"response": "answer", "eval_count": 2, "logprobs": [{"logprob": x} for x in reported]}
    assert check_selected_logprobs(leaf, ollama, "answer")["passed"] is expected


@pytest.mark.parametrize('settings,threads,context', [({}, 1, 256), ({'threads': 4, 'context': 512}, 4, 512)])
def test_generation_request_keeps_matched_greedy_cpu_work(settings, threads, context):
    request = validation.generation_request('local-model','raw input', **settings)
    assert request['raw'] is True and request['stream'] is False
    options = request['options']
    assert (options['num_gpu'], options['num_thread'], options['num_predict']) == (0, threads, 2)
    assert options['num_ctx'] == options['num_batch'] == context
    assert options['temperature'] == 0 and options['repeat_penalty'] == 1
    assert options['stop'] == []


def test_fixed_generation_work_rejects_early_eos_and_reused_prompt():
    leaf = dict(input_tokens=768, cached_prompt_tokens=0, uncached_prompt_tokens=768,
                generated_tokens=[42] * 64, cached_decode_forwards=63)
    ollama = dict(prompt_eval_count=768, prompt_eval_cached_count=0, eval_count=64)
    check_generation_work(leaf, 'leaf', 768, 0, 64)
    check_generation_work(ollama, 'Ollama', 768, 0, 64)
    with pytest.raises(ValueError, match='Unmatched leaf'):
        check_generation_work(dict(leaf, generated_tokens=[42] * 63, cached_decode_forwards=62),
                              'leaf', 768, 0, 64)
    with pytest.raises(ValueError, match='Unmatched Ollama'):
        check_generation_work(dict(ollama, prompt_eval_cached_count=64), 'Ollama', 768, 0, 64)


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
