"""Local Ollama lifecycle and HTTP helpers for controlled CPU comparisons."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

import psutil


def api(base, path, payload=None, timeout=300):
    data=json.dumps(payload).encode() if payload is not None else None
    request=urllib.request.Request(base+path,data=data,
        headers={'Content-Type':'application/json'},method='POST' if data is not None else 'GET')
    try:
        with urllib.request.urlopen(request,timeout=timeout) as response:
            result=json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'Ollama HTTP {error.code}: {error.read().decode()}') from error
    if isinstance(result,dict) and result.get('error'):
        raise RuntimeError(result['error'])
    return result


@contextmanager
def local_server(executable, models, directory, *, cpu=None):
    """Own a separate localhost server; never modify an existing user service."""
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0))
        port=probe.getsockname()[1]
    environment=os.environ.copy()
    environment.update(OLLAMA_HOST=f'127.0.0.1:{port}', OLLAMA_MODELS=str(Path(models).resolve()),
        OLLAMA_NO_CLOUD='1', OLLAMA_NUM_PARALLEL='1', OLLAMA_MAX_LOADED_MODELS='1',
        OLLAMA_KV_CACHE_TYPE='f32', OLLAMA_FLASH_ATTENTION='0', LLAMA_ARG_CACHE_RAM='0')
    base=f'http://127.0.0.1:{port}'
    with (directory/'server.log').open('wb') as log:
        process=subprocess.Popen([str(Path(executable).resolve()),'serve'],env=environment,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW)
        parent=None
        try:
            parent=psutil.Process(process.pid)
            if cpu is not None:
                parent.cpu_affinity([cpu])
                parent.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError(f'Ollama exited; inspect {directory / "server.log"}')
                try:
                    api(base,'/api/version',timeout=1)
                    break
                except (OSError,RuntimeError):
                    time.sleep(0.1)
            else:
                raise TimeoutError('Local Ollama server did not become ready')
            yield base, environment, process
        finally:
            # Only descendants of the process created above are eligible.
            try:
                owned=parent.children(recursive=True)+[parent] if parent is not None else []
            except psutil.NoSuchProcess:
                owned=[]
            for child in owned:
                try: child.terminate()
                except psutil.NoSuchProcess: pass
            _,remaining=psutil.wait_procs(owned,timeout=5)
            for child in remaining:
                try: child.kill()
                except psutil.NoSuchProcess: pass
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)


def generation_request(model,prompt,*,tokens=2,keep_alive='5m',logprobs=False):
    return dict(model=model,prompt=prompt,raw=True,stream=False,keep_alive=keep_alive,
        logprobs=logprobs,top_logprobs=1 if logprobs else 0,
        options=dict(num_gpu=0,num_thread=1,num_ctx=256,num_batch=256,num_predict=tokens,
            temperature=0,repeat_penalty=1.0,top_p=1.0,min_p=0.0,seed=0,stop=[]))
