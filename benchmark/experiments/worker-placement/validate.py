"""Qualify explicit placement without changing default kernels or arithmetic.

check: rebind full trained quality bit-exactly to the prior qualified FP32 build.
default: accurate arithmetic, old vs new unbound, one core; ratios <=1.02.
placement: same new binary, four allowed cores, off/on balanced pairs; prefill
both ratios <=0.98, decode <=1.02, all individual/pooled p90/p10 <=1.25.
jobs: two accurate binaries with the same four assigned workers; same speed gate.
synchronization: same placement/arithmetic, prefill <=1.02, decode <=0.98.
ordinary: released ordinary arithmetic vs final ordinary build, FP32 artifacts,
one core, both ratios <=1.02; bound relative-regression quality is required.
No samples filtered. Six pairs, 31 samples and ten warmups/pass. AC and awake
guards plus a fail-closed sleep audit. Placement never becomes automatic.
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
from leaf.power import keep_awake
from tools.decoder_validation import run_native,latency_stability
from tools.validate_decoder import digest,utc_now,write_record

SOURCE=ROOT/'build/source-bf16-20261020'
BASE=SOURCE/'accurate-final/leaf_decoder.exe'
PRIOR=ROOT/'build/linear-accumulation-20261019-pipeline'
CORES=[2,4,6,8]  # This machine's checked benchmark topology, never a runtime default.
spec=importlib.util.spec_from_file_location('storage_guards',ROOT/'benchmark/experiments/source-bf16/integration.py')
guards=importlib.util.module_from_spec(spec);spec.loader.exec_module(guards)


def context(directory,binary,baseline=None,baseline_quality=None,ordinary=False):
    prepared=SOURCE/'prepared.json';qualified=SOURCE/'quality.json'
    models=json.loads(prepared.read_text());gold=json.loads(qualified.read_text())
    if not (models['complete'] and gold['complete'] and gold['passed']):
        raise ValueError('Require complete prior storage and trained quality')
    if gold['hashes'][str(BASE)]!=digest(BASE) or gold['hashes'][str(prepared)]!=digest(prepared):
        raise ValueError('Prior qualified inputs changed')
    manifest=binary.parent/'build.json'
    if not ordinary:
        build=json.loads(manifest.read_text())
        if not build['complete'] or build['binary_sha256']['leaf_decoder.exe']!=digest(binary):
            raise ValueError('Require a complete candidate build')
        if any(digest(Path(p))!=h for p,h in build['source_sha256'].items()):
            raise ValueError('Candidate build source changed')
    for row in models['models'].values():
        if digest(Path(row['after']))!=row['after_sha256']:raise ValueError('Artifact changed')
        if ordinary and digest(Path(row['before']))!=row['before_sha256']:
            raise ValueError('Ordinary FP32 artifact changed')
    scratch=directory/'tmp';scratch.mkdir(exist_ok=True)
    os.environ['TEMP']=os.environ['TMP']=str(scratch)
    import tempfile
    tempfile.tempdir=str(scratch)
    paths=[prepared,qualified,BASE,binary,Path(__file__),ROOT/'leaf/power.py',ROOT/'leaf/affinity.py',
           ROOT/'tools/decoder_validation.py',ROOT/'benchmark/experiments/source-bf16/integration.py']
    if not ordinary:paths.append(manifest)
    if baseline is not None:
        prior=json.loads(baseline_quality.read_text())
        if not prior['complete'] or not prior['passed']:raise ValueError('Baseline quality incomplete')
        if ordinary:
            if (prior['binary_sha256']['before']!=digest(baseline) or
                    prior['binary_sha256']['after']!=digest(binary)):
                raise ValueError('Ordinary binaries differ from regression proof')
            for row in models['models'].values():
                if prior['hashes'][row['before']]!=digest(Path(row['before'])):
                    raise ValueError('Ordinary regression artifact changed')
        elif prior['hashes'][str(baseline)]!=digest(baseline):
            raise ValueError('Baseline binary differs from full trained proof')
        if prior['hashes'][str(prepared)]!=digest(prepared):
            raise ValueError('Baseline prepared input changed')
        paths.extend([baseline,baseline_quality])
    return models,{str(p):digest(p) for p in paths}


def check(directory,binary,baseline=None,baseline_quality=None):
    report=directory/'quality.json'
    if report.exists():raise ValueError('Use fresh quality output; retain partial records')
    models,hashes=context(directory,binary,baseline,baseline_quality)
    reference=baseline if baseline is not None else BASE
    reference_threads=4 if baseline is not None else 1
    reference_cpus=CORES if baseline is not None else None
    record=dict(complete=False,passed=False,scope='Bit-exact full trained rebinding; unchanged strict tolerances',
                hashes=hashes,models={})
    write_record(report,record)
    try:
        with keep_awake(display=True):
            for name,row in models['models'].items():
                tokens=PRIOR/('full-quality-'+name)/'tokens.json'
                quality=PRIOR/('full-quality-'+name)/'quality.json'
                gold=json.loads(quality.read_text());sequences=json.loads(tokens.read_text())['sequences']
                if not (gold['complete'] and gold['passed'] and len(sequences)==8 and
                        len(gold['generations'])==6 and gold['tokens_sha256']==digest(tokens)):
                    raise ValueError('Incomplete previous trained proof')
                hashes.update({str(p):digest(p) for p in (tokens,quality)})
                artifact=Path(row['after']);cases=record['models'][name]=dict(blocks=[],generations=[],controls=[])
                for index,ids in enumerate(sequences):
                    a,_=run_native(reference,artifact,[ids],threads=reference_threads,worker_cpus=reference_cpus)
                    b,metrics=run_native(binary,artifact,[ids],threads=4,worker_cpus=CORES)
                    if baseline is not None and metrics.get('tile_aligned_linear_calls',0)<=0:
                        raise ValueError('New job path inactive on a full block')
                    exact=bool(np.array_equal(a.view('<u4'),b.view('<u4')))
                    cases['blocks'].append(dict(index=index,exact=exact,logits=int(a.size)))
                    write_record(report,record);print(name,'block',index+1,'exact',exact,flush=True)
                    if not exact:raise ValueError('Full trained logits changed')
                for case in gold['generations']:
                    _,m=run_native(binary,artifact,[case['input_ids']],threads=4,worker_cpus=CORES,mode='generate',generate=16)
                    exact=m['generated_tokens']==case['reference_tokens']
                    cases['generations'].append(dict(prompt=case['prompt'],exact=exact))
                    if not exact:raise ValueError('Reference continuation changed')
                for threads,scalar,mode,chunk,cpus in [(1,False,'verify',1,CORES[:1]),
                    (4,True,'verify',1,CORES),(2,False,'verify',1,CORES[:2]),
                    (4,False,'chunked',9,CORES),(4,False,'verify',1,None)]:
                    ids=sequences[0][:33]
                    a,_=run_native(reference,artifact,[ids],threads=reference_threads,worker_cpus=reference_cpus,scalar=scalar,mode=mode,chunk=chunk)
                    b,metrics=run_native(binary,artifact,[ids],threads=threads,scalar=scalar,mode=mode,chunk=chunk,worker_cpus=cpus)
                    if baseline is not None and (scalar or threads==1) and metrics['tile_aligned_linear_calls']:
                        raise ValueError('Job-sizing exclusion changed')
                    exact=bool(np.array_equal(a.view('<u4'),b.view('<u4')))
                    cases['controls'].append(dict(threads=threads,scalar=scalar,mode=mode,worker_cpus=cpus,exact=exact))
                    if not exact:raise ValueError('Scalar/cache/default/thread control changed')
                write_record(report,record);print(name,'reference continuations and controls pass',flush=True)
        if any(digest(Path(p))!=h for p,h in hashes.items()):raise ValueError('Quality inputs changed')
        record.update(complete=True,passed=True)
    except BaseException as error:
        record['error']=repr(error);raise
    finally:write_record(report,record)


def timing(directory,binary,stage,baseline=None,baseline_quality=None):
    report=directory/(stage+'.json')
    if report.exists():raise ValueError('Retain previous timing; use fresh output')
    models,hashes=context(directory,binary,baseline,baseline_quality,ordinary=stage=='ordinary')
    quality=baseline_quality if stage=='ordinary' else directory/'quality.json'
    gold=json.loads(quality.read_text())
    if not (gold['complete'] and gold['passed'] and (stage=='ordinary' or gold['hashes'][str(binary)]==digest(binary))):
        raise ValueError('Complete bound trained quality required before timing')
    hashes[str(quality)]=digest(quality)
    from tools.benchmark_core_scaling import validate_cores
    from tools.diagnose_windows_benchmark import cpu_sets
    import psutil
    cores=CORES[:1] if stage in ('default','ordinary') else CORES
    topology=validate_cores(cores,cpu_sets());process=psutil.Process()
    previous_affinity=process.cpu_affinity();previous_priority=process.nice()
    process.cpu_affinity(cores);process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    if process.cpu_affinity()!=cores:raise ValueError('Parent affinity differs from request')
    os.environ['LEAF_BENCHMARK_REQUIRE_AC']='1'
    if any(v for k,v in os.environ.items() if k.startswith('LEAF_EXPERIMENTAL_') or k=='LEAF_DECODER_PROFILE'):
        raise ValueError('Clear profiling and experimental environment first')
    record=dict(complete=False,passed=False,utc=utc_now(),stage=stage,criteria=__doc__,hashes=hashes,
                threads=len(cores),cores=cores,topology=topology,default_placement_changed=False,
                baseline=str(baseline) if baseline is not None else None,models={})
    write_record(report,record)
    try:
        with keep_awake(display=True):
            for name,row in models['models'].items():
                tokens=ROOT/'build/numerics-20261017/decode-silu'/(name+'-work/tokens.json')
                ids=json.loads(tokens.read_text())['benchmark_ids'];hashes[str(tokens)]=digest(tokens)
                cases=record['models'][name]=dict(passes=[])
                for variant in ['before','after','after','before']*3:
                    cpus=CORES if stage in ('jobs','synchronization') or (stage=='placement' and variant=='after') else None
                    executable=(baseline if baseline is not None else BASE) if stage in ('default','ordinary','jobs','synchronization') and variant=='before' else binary
                    started=utc_now()
                    with guards.power_window() as power:
                        _,m=run_native(executable,Path(row['before' if stage=='ordinary' else 'after']),[ids],threads=len(cores),worker_cpus=cpus,
                            mode='bench',runs=31,warmup=10,windows_above_normal=True)
                    if not m['benchmark_ac_checked']:raise ValueError('Native AC guard missing')
                    if stage=='jobs' and variant=='after' and m.get('tile_aligned_linear_calls',0)<=0:
                        raise ValueError('New job path inactive during timing')
                    if stage=='synchronization' and variant=='after' and m.get('experimental_worker_spin_us')!=50:
                        raise ValueError('Bounded-spin path missing during timing')
                    cases['passes'].append(dict(variant=variant,worker_cpus=cpus,started_at_utc=started,
                        ended_at_utc=utc_now(),metrics=m,power=power))
                    write_record(report,record)
                    print(stage,name,variant,m['prefill_p50_ms'],m['decode_p50_ms'],flush=True)
                summary={}
                for phase in ('prefill','decode'):
                    key=phase+'_samples_ms'
                    pooled={s:[v for p in cases['passes'] if p['variant']==s for v in p['metrics'][key]] for s in ('before','after')}
                    medians={s:statistics.median(v) for s,v in pooled.items()}
                    ordered={s:[statistics.median(p['metrics'][key]) for p in cases['passes'] if p['variant']==s] for s in pooled}
                    paired=statistics.median(a/b for a,b in zip(ordered['after'],ordered['before']))
                    stability={s:dict(pooled=latency_stability(pooled[s]),
                        individual=[latency_stability(p['metrics'][key]) for p in cases['passes'] if p['variant']==s]) for s in pooled}
                    stable=all(v['pooled']['passed'] and all(x['passed'] for x in v['individual']) for v in stability.values())
                    limit=0.98 if stage in ('placement','jobs') and phase=='prefill' else 1.02
                    if stage=='synchronization' and phase=='decode':limit=0.98
                    ratio=medians['after']/medians['before']
                    summary[phase]=dict(medians_ms=medians,pooled_ratio=ratio,median_paired_ratio=paired,
                        stability=stability,limit=limit,passed=stable and ratio<=limit and paired<=limit)
                cases['summary']=summary;write_record(report,record)
                print(stage,name,{p:{k:v for k,v in s.items() if k!='stability'} for p,s in summary.items()},flush=True)
            record['sleep_events']=guards.sleep_events(record['utc'])
        if any(digest(Path(p))!=h for p,h in hashes.items()):raise ValueError('Timing inputs changed')
        record['complete']=True
        record['passed']=not record['sleep_events'] and all(s['passed'] for m in record['models'].values() for s in m['summary'].values())
    except BaseException as error:
        record['error']=repr(error);raise
    finally:
        process.cpu_affinity(previous_affinity);process.nice(previous_priority);write_record(report,record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('check','default','placement','jobs','ordinary','synchronization'))
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--baseline-quality',type=Path)
    args=parser.parse_args();directory=args.directory.resolve();directory.mkdir(parents=True,exist_ok=True)
    if (args.baseline is None)!=(args.baseline_quality is None):parser.error('Provide baseline and its bound quality together')
    if args.stage in ('jobs','ordinary','synchronization') and args.baseline is None:parser.error('This stage requires a bound baseline')
    if args.stage in ('default','placement') and args.baseline is not None:parser.error('Use the fixed baseline for this stage')
    baseline=args.baseline.resolve() if args.baseline else None
    quality=args.baseline_quality.resolve() if args.baseline_quality else None
    if args.stage=='check':check(directory,args.binary.resolve(),baseline,quality)
    else:timing(directory,args.binary.resolve(),args.stage,baseline,quality)


if __name__=='__main__':main()
