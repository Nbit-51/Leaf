"""Unmodified one-core binaries for ETW attribution, never speed promotion.

Twelve balanced launches, 31 samples/ten warmups. Record owned child PIDs and
UTC windows so scheduling/power events can be aligned with retained samples.
No PDH, native instrumentation, quantization selection or model download.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import psutil

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from leaf.power import keep_awake,require_benchmark_ac
from tools.decoder_validation import latency_stability
from tools.validate_decoder import digest,utc_now,write_record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    directory=args.directory.resolve()
    if os.name!='nt' or not directory.is_relative_to((ROOT/'build').resolve()) or directory==(ROOT/'build').resolve():
        parser.error('Use a fresh directory inside repository build/ on Windows')
    if (directory/'replay.json').exists():parser.error('Keep existing records; choose a fresh directory')
    base=ROOT/'build/worker-placement-20261022'
    before=ROOT/'build/source-bf16-20261020/default/leaf_decoder.exe'
    after=Path((base/'final-runtime.txt').read_text().strip())
    proof=base/'final-ordinary-regression.json'
    regression=json.loads(proof.read_text())
    if not (regression['complete'] and regression['passed'] and
        regression['binary_sha256']==dict(before=digest(before),after=digest(after))):
        raise ValueError('Unmodified binary identities differ from regression proof')
    prepared=ROOT/'build/source-bf16-20261020/prepared.json'
    row=json.loads(prepared.read_text())['models']['smol']
    artifact=Path(row['before'])
    if digest(artifact)!=row['before_sha256']:raise ValueError('FP32 artifact changed')
    tokens=ROOT/'build/numerics-20261017/decode-silu/smol-work/tokens.json'
    failed_guard=base/'final-default-model/ordinary.json'
    failed=json.loads(failed_guard.read_text())
    if not failed['complete'] or failed['passed']:raise ValueError('Require the retained failed ordinary guard')
    for path in (before,after,prepared,tokens):
        if failed['hashes'][str(path)]!=digest(path):raise ValueError('Inputs differ from the failing comparison')
    ids=json.loads(tokens.read_text())['benchmark_ids']
    if len(ids)!=64 or any(type(x)!=int or x<0 or x>=2**32 for x in ids):raise ValueError('Benchmark IDs changed')
    if any(v for k,v in os.environ.items() if k.startswith('LEAF_EXPERIMENTAL_') or k=='LEAF_DECODER_PROFILE'):
        raise ValueError('Clear experimental/profile environment before tracing')
    for binary in (before,after):
        info=json.loads(subprocess.check_output([str(binary),'--runtime-info'],text=True))
        if not info.get('benchmark_power_guard') or info.get('diagnostic_counter_version'):
            raise ValueError('Require uninstrumented runtimes with native AC guards')
    guard_source=ROOT/'benchmark/experiments/source-bf16/integration.py'
    paths=[before,after,artifact,prepared,proof,tokens,failed_guard,Path(__file__),guard_source,
           ROOT/'leaf/power.py',ROOT/'tools/decoder_validation.py']
    hashes={str(p):digest(p) for p in paths}
    if args.check_only:
        print(json.dumps(dict(ready=True,timing_started=False,model='SmolLM2-360M-Instruct',
            threads=1,cpu=2,passes=12,runs=31,warmup=10,before=str(before),after=str(after))))
        return
    require_benchmark_ac()
    directory.mkdir(parents=True,exist_ok=True)
    scratch=directory/'tmp';scratch.mkdir(exist_ok=True)
    os.environ['TEMP']=os.environ['TMP']=str(scratch)
    tempfile.tempdir=str(scratch)
    os.environ['LEAF_BENCHMARK_REQUIRE_AC']='1'
    spec=importlib.util.spec_from_file_location('etw_replay_guards',guard_source)
    guards=importlib.util.module_from_spec(spec);spec.loader.exec_module(guards)
    me=psutil.Process();affinity,priority=me.cpu_affinity(),me.nice()
    if 2 not in affinity:raise ValueError('Benchmark CPU 2 is outside the caller allowed set')
    report=directory/'replay.json'
    record=dict(complete=False,acceptance_timing=False,scope=__doc__,utc=utc_now(),
        hashes=hashes,cpu=2,threads=1,passes=[])
    write_record(report,record)
    try:
        me.cpu_affinity([2]);me.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        with keep_awake(display=True):
            for index,variant in enumerate(['before','after','after','before']*3):
                binary=before if variant=='before' else after
                with tempfile.TemporaryDirectory(dir=scratch) as temp:
                    work=Path(temp);request=work/'tokens.bin'
                    request.write_bytes(struct.pack('<II',1,len(ids))+struct.pack(f'<{len(ids)}I',*ids))
                    command=[str(binary),str(artifact),str(request),str(work/'logits.bin'),str(work/'metrics.json'),
                             'bench','1','31','10','1','0','1','32']
                    started=utc_now()
                    with guards.power_window() as power:
                        child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                                               creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS)
                        try:
                            owned=psutil.Process(child.pid)
                            if owned.cpu_affinity()!=[2] or owned.nice()!=psutil.ABOVE_NORMAL_PRIORITY_CLASS:
                                raise ValueError('Owned decoder launch policy changed')
                            stdout,stderr=child.communicate()
                        except BaseException:
                            if child.poll() is None:child.kill()
                            child.communicate();raise
                    ended=utc_now()
                    if child.returncode:raise RuntimeError(stderr)
                    m=json.loads((work/'metrics.json').read_text())
                item=dict(index=index,variant=variant,pid=child.pid,started_at_utc=started,ended_at_utc=ended,
                    metrics=m,power=power,stability={p:latency_stability(m[p+'_samples_ms']) for p in ('prefill','decode')})
                record['passes'].append(item);write_record(report,record)
                if not m['benchmark_ac_checked'] or m.get('worker_spin_us',0)!=0 or any(len(m[p+'_samples_ms'])!=31 for p in ('prefill','decode')):
                    raise ValueError('AC, inactive spin or sample-count audit failed')
                print(index+1,variant,m['prefill_p50_ms'],m['decode_p50_ms'],flush=True)
            record['sleep_events']=guards.sleep_events(record['utc'])
        if any(digest(Path(p))!=h for p,h in hashes.items()):raise ValueError('Trace replay inputs changed')
        record['complete']=True
    except BaseException as error:
        record['error']=repr(error);raise
    finally:
        me.cpu_affinity(affinity);me.nice(priority);write_record(report,record)


if __name__=='__main__':main()
