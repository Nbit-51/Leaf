"""Trained exactness and matched four-core acceptance for tile-aligned jobs.

Both paired and pooled ratios: prefill <=0.98, decode <=1.02; every pass and
pooled p90/p10 <=1.25. Same four physical cores and FP32 math/storage policy.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tools.decoder_validation import run_native,latency_stability
from tools.validate_decoder import digest,utc_now,write_record
from leaf.power import keep_awake

SOURCE=ROOT/'build/source-bf16-20261020'
BASE=SOURCE/'accurate-final/leaf_decoder.exe'
spec=importlib.util.spec_from_file_location('storage_guards',ROOT/'benchmark/experiments/source-bf16/integration.py')
guards=importlib.util.module_from_spec(spec); spec.loader.exec_module(guards)


def context(directory,binary):
    prepared=SOURCE/'prepared.json'; qualified=SOURCE/'quality.json'
    models=json.loads(prepared.read_text()); gold=json.loads(qualified.read_text())
    assert models['complete'] and gold['complete'] and gold['passed']
    assert gold['hashes'][str(BASE)]==digest(BASE)
    assert gold['hashes'][str(prepared)]==digest(prepared)
    for row in models['models'].values():
        assert digest(Path(row['after']))==row['after_sha256']
    scratch=directory/'tmp'; scratch.mkdir(exist_ok=True)
    os.environ['TEMP']=os.environ['TMP']=str(scratch)
    import tempfile
    tempfile.tempdir=str(scratch)
    paths=[prepared,qualified,BASE,binary,Path(__file__),ROOT/'leaf/power.py',
           ROOT/'tools/decoder_validation.py',ROOT/'benchmark/experiments/source-bf16/integration.py']
    return models,{str(p):digest(p) for p in paths}


def check(directory,binary,resume=None):
    report=directory/'model-quality.json'
    if report.exists(): raise ValueError('Keep previous quality records; use a fresh directory')
    models,hashes=context(directory,binary)
    cached={}
    if resume:
        old=json.loads(resume.read_text())
        snapshot=resume.parent.parent/'model-first.py'
        assert not old['complete'] and not old['passed']
        assert old['hashes'][str(Path(__file__))]==digest(snapshot)
        for p,h in old['hashes'].items():
            if Path(p)!=Path(__file__): assert digest(Path(p))==h
        assert old['hashes'][str(binary)]==digest(binary)
        hashes.update({str(resume):digest(resume),str(snapshot):digest(snapshot)})
        cached=old['models']
    record=dict(complete=False,passed=False,scope='Full prior trained quality exactly rebound from one thread to tile-aligned four-thread execution',
                hashes=hashes,models={})
    write_record(report,record)
    try:
        for name,row in models['models'].items():
            prior=ROOT/'build/linear-accumulation-20261019-pipeline'/('full-quality-'+name)
            tokens=prior/'tokens.json'; quality=prior/'quality.json'
            gold=json.loads(quality.read_text()); sequences=json.loads(tokens.read_text())['sequences']
            assert gold['complete'] and gold['passed'] and len(sequences)==8 and len(gold['generations'])==6
            assert gold['tokens_sha256']==digest(tokens)
            hashes.update({str(p):digest(p) for p in (tokens,quality)})
            artifact=Path(row['after'])
            cases=record['models'][name]=dict(blocks=[],generations=[],controls=[])
            for index,ids in enumerate(sequences):
                saved=cached.get(name,{}).get('blocks',[])
                if index<len(saved):
                    item=saved[index]
                    assert item['index']==index and item['exact'] and item['aligned_active']
                    cases['blocks'].append(dict(item,reused_from_bound_proof=True))
                    print(name,'block',index+1,'retained exact proof',flush=True)
                    continue
                a,_=run_native(BASE,artifact,[ids],threads=1)
                b,metrics=run_native(binary,artifact,[ids],threads=4)
                exact=bool(np.array_equal(a.view('<u4'),b.view('<u4')))
                active=metrics['tile_aligned_linear_calls']>0
                cases['blocks'].append(dict(index=index,exact=exact,aligned_active=active,logits=int(a.size)))
                write_record(report,record); print(name,'block',index+1,'exact',exact,flush=True)
                if not exact or not active: raise ValueError('Full-logit exactness or job dispatch failed')
            for case in gold['generations']:
                saved=cached.get(name,{}).get('generations',[])
                index=len(cases['generations'])
                if index<len(saved):
                    assert saved[index]['prompt']==case['prompt'] and saved[index]['exact']
                    cases['generations'].append(dict(saved[index],reused_from_bound_proof=True))
                    continue
                _,metrics=run_native(binary,artifact,[case['input_ids']],threads=4,mode='generate',generate=16)
                exact=metrics['generated_tokens']==case['reference_tokens']
                cases['generations'].append(dict(prompt=case['prompt'],exact=exact))
                if not exact: raise ValueError('Reference continuation changed')
            for threads,scalar,mode,chunk in [(1,False,'verify',1),(4,True,'verify',1),(4,False,'chunked',9),(2,False,'verify',1)]:
                ids=sequences[0][:33]
                a,_=run_native(BASE,artifact,[ids],threads=1,scalar=scalar,mode=mode,chunk=chunk)
                b,metrics=run_native(binary,artifact,[ids],threads=threads,scalar=scalar,mode=mode,chunk=chunk)
                exact=bool(np.array_equal(a.view('<u4'),b.view('<u4')))
                cases['controls'].append(dict(threads=threads,scalar=scalar,mode=mode,exact=exact,
                    aligned_calls=metrics['tile_aligned_linear_calls']))
                if not exact: raise ValueError('Scalar/cache/thread control changed')
                if threads==1 or scalar:
                    if metrics['tile_aligned_linear_calls']: raise ValueError('Exclusion dispatch changed')
            write_record(report,record); print(name,'generation and exclusion controls pass',flush=True)
        if any(digest(Path(p))!=h for p,h in hashes.items()): raise ValueError('Quality inputs changed')
        record.update(complete=True,passed=True)
    finally: write_record(report,record)


def timing(directory,binary):
    report=directory/'model-acceptance.json'
    if report.exists(): raise ValueError('Keep previous timing records')
    models,hashes=context(directory,binary)
    quality=directory/'model-quality.json'; q=json.loads(quality.read_text())
    if not q['complete'] or not q['passed'] or q['hashes'][str(binary)]!=digest(binary):
        raise ValueError('Require complete bound trained quality')
    hashes[str(quality)]=digest(quality)
    from tools.benchmark_core_scaling import validate_cores
    from tools.diagnose_windows_benchmark import cpu_sets
    import psutil
    cores=[2,4,6,8]; topology=validate_cores(cores,cpu_sets())
    psutil.Process().cpu_affinity(cores)
    assert psutil.Process().cpu_affinity()==cores
    psutil.Process().nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    os.environ['LEAF_BENCHMARK_REQUIRE_AC']='1'
    if any(v for k,v in os.environ.items() if k.startswith('LEAF_EXPERIMENTAL_') or k=='LEAF_DECODER_PROFILE'):
        raise ValueError('Clear profiling/experimental environment')
    record=dict(complete=False,passed=False,hashes=hashes,utc=utc_now(),threads=4,cores=cores,topology=topology,
                criteria=__doc__,models={})
    write_record(report,record)
    try:
        with keep_awake(display=True):
            for name,row in models['models'].items():
                tokens=ROOT/'build/numerics-20261017/decode-silu'/(name+'-work/tokens.json')
                ids=json.loads(tokens.read_text())['benchmark_ids']; hashes[str(tokens)]=digest(tokens)
                cases=record['models'][name]=dict(passes=[])
                for stage in ['before','after','after','before']*3:
                    started=utc_now()
                    with guards.power_window() as power:
                        _,metrics=run_native(BASE if stage=='before' else binary,Path(row['after']),[ids],
                            threads=4,mode='bench',runs=31,warmup=10,windows_above_normal=True)
                    if not metrics['benchmark_ac_checked']: raise ValueError('Native AC guard missing')
                    if stage=='after' and not metrics['tile_aligned_linear_calls']: raise ValueError('Inactive candidate')
                    cases['passes'].append(dict(stage=stage,started_at_utc=started,ended_at_utc=utc_now(),metrics=metrics,power=power))
                    write_record(report,record)
                    print(name,stage,metrics['prefill_p50_ms'],metrics['decode_p50_ms'],flush=True)
                summary={}
                for phase,limit in [('prefill',0.98),('decode',1.02)]:
                    key=phase+'_samples_ms'
                    pooled={s:[v for p in cases['passes'] if p['stage']==s for v in p['metrics'][key]] for s in ('before','after')}
                    medians={s:statistics.median(v) for s,v in pooled.items()}
                    ordered={s:[statistics.median(p['metrics'][key]) for p in cases['passes'] if p['stage']==s] for s in pooled}
                    paired=statistics.median(a/b for a,b in zip(ordered['after'],ordered['before']))
                    stable=all(latency_stability(p['metrics'][key])['passed'] for p in cases['passes'])
                    stable=stable and all(latency_stability(v)['passed'] for v in pooled.values())
                    ratio=medians['after']/medians['before']
                    summary[phase]=dict(medians_ms=medians,pooled_ratio=ratio,median_paired_ratio=paired,stable=stable,
                                        passed=stable and paired<=limit and ratio<=limit)
                cases['summary']=summary; write_record(report,record); print(name,json.dumps(summary),flush=True)
            record['sleep_events']=guards.sleep_events(record['utc'])
        if any(digest(Path(p))!=h for p,h in hashes.items()): raise ValueError('Timing inputs changed')
        record['complete']=True
        record['passed']=not record['sleep_events'] and all(v['passed'] for row in record['models'].values() for v in row['summary'].values())
    finally: write_record(report,record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('check','timing'))
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--resume-quality',type=Path,help='retain bound exact cases from the interrupted first quality attempt')
    args=parser.parse_args(); directory=args.directory.resolve(); directory.mkdir(parents=True,exist_ok=True)
    if args.stage=='check': check(directory,args.binary.resolve(),args.resume_quality.resolve() if args.resume_quality else None)
    else: timing(directory,args.binary.resolve())


if __name__=='__main__': main()
