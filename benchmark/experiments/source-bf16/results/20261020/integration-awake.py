"""Resume-safe full-model storage audit, exact quality binding and timing.

Targets decode: <=0.98 paired decode ratio, <=1.02 prefill ratio, every pass
and pooled p90/p10 <=1.25. No samples filtered. Accuracy limits are unchanged.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from leaf.affinity import pin_cpu
from leaf.evaluation import write_record
from leaf.power import keep_awake, require_benchmark_ac
from tools.decoder_validation import latency_stability, run_native
from tools.export_decoder import HEADER, DESCRIPTOR, export_decoder
from tools.validate_decoder import digest, utc_now

PRIOR = ROOT/'build/linear-accumulation-20261019-pipeline'


def table(path):
    with path.open('rb') as f:
        header=f.read(HEADER.size)
        values=HEADER.unpack(header)
        data_base=values[3]
        blob=header+f.read(data_base-HEADER.size)
    position=HEADER.size
    if values[1]>=4:
        position += 4+4*struct.unpack_from('<I',blob,position)[0]
    result={}
    for _ in range(values[2]):
        size=struct.unpack_from('<I',blob,position)[0]; position+=4
        name=blob[position:position+size].decode(); position+=size
        result[name]=DESCRIPTOR.unpack_from(blob,position); position+=DESCRIPTOR.size
    return data_base,result


def audit(before,after):
    ba,ta=table(before); bb,tb=table(after)
    if ta.keys()!=tb.keys(): raise ValueError('Changed artifact tensor set')
    checks=[]
    for name,a in ta.items():
        b=tb[name]
        if a[1:3]!=b[1:3] or a[3]!=32 or b[3] not in (16,32):
            raise ValueError('Unexpected shape/storage change')
        count=a[1]*a[2]
        original=np.memmap(before,mode='r',dtype='<u4',shape=(count,),offset=ba+a[5])
        saved=np.memmap(after,mode='r',dtype='<u2' if b[3]==16 else '<u4',shape=(count,),offset=bb+b[5])
        for start in range(0,count,262144):
            piece=saved[start:start+262144]
            if b[3]==16: piece=piece.astype('<u4')<<16
            if not np.array_equal(original[start:start+262144],piece):
                raise ValueError('Storage discarded source bits: '+name)
        checks.append(dict(name=name,storage_bits=b[3],exact=True))
        del original,saved
    return checks


def prepare(directory):
    output=directory/'prepared.json'
    if output.exists(): raise ValueError('Prepared record already exists')
    paths=json.loads((ROOT/'build/numerics-20261017/model-paths.json').read_text())
    record=dict(complete=False,hashes={str(Path(__file__)):digest(Path(__file__))},models={})
    write_record(output,record)
    for name in ('smol','qwen'):
        snapshot,before=Path(paths[name]['snapshot']),Path(paths[name]['artifact'])
        after=directory/(name+'-source-bf16.leaf')
        if after.exists(): raise ValueError('Refuse to overwrite a prepared artifact')
        source_hashes={str(p):digest(p) for p in [snapshot/'config.json',*sorted(snapshot.glob('*.safetensors'))]}
        before_hash=digest(before)
        exported=export_decoder(snapshot,after,source_bf16_head=True)
        if not exported['source_bf16_tensors']: raise ValueError('No source-preserving storage selected')
        checks=audit(before,after)
        if digest(before)!=before_hash or any(digest(Path(p))!=h for p,h in source_hashes.items()):
            raise ValueError('Preparation inputs changed')
        record['models'][name]=dict(snapshot=str(snapshot),before=str(before),after=str(after),
            before_sha256=before_hash,after_sha256=digest(after),before_bytes=before.stat().st_size,
            after_bytes=after.stat().st_size,export=exported,tensor_checks=checks,source_hashes=source_hashes)
        write_record(output,record)
        print(name,'all tensors exact, bytes',before.stat().st_size,'->',after.stat().st_size,flush=True)
    record['complete']=True; write_record(output,record)


def context(directory,binary):
    p=directory/'prepared.json'; r=json.loads(p.read_text())
    if not r['complete']: raise ValueError('Preparation incomplete')
    for row in r['models'].values():
        for stage in ('before','after'):
            if digest(Path(row[stage]))!=row[stage+'_sha256']: raise ValueError('Artifact changed')
    os.environ['TEMP']=os.environ['TMP']=str((directory/'tmp').resolve())
    (directory/'tmp').mkdir(exist_ok=True)
    import tempfile
    tempfile.tempdir=os.environ['TEMP']
    return r,{str(p):digest(p),str(binary):digest(binary),str(Path(__file__)):digest(Path(__file__))}


def check(directory,binary):
    report=directory/'quality.json'
    if report.exists(): raise ValueError('Quality record already exists')
    models,hashes=context(directory,binary)
    old=PRIOR/'leaf_decoder.exe'
    hashes[str(old)]=digest(old)
    record=dict(complete=False,passed=False,timing_qualified=False,
        scope='All eight prior fully qualified blocks/model exactly rebound to native source-BF16; six reference generations/model; trained cache/scalar controls',
        hashes=hashes,models={})
    write_record(report,record)
    try:
        for name,model in models['models'].items():
            before,after=Path(model['before']),Path(model['after'])
            quality=PRIOR/('full-quality-'+name)/'quality.json'
            tokens=quality.with_name('tokens.json')
            gold=json.loads(quality.read_text()); fixture=json.loads(tokens.read_text())
            if (not gold['complete'] or not gold['passed'] or len(gold['blocks'])!=8 or
                len(gold['generations'])!=6 or gold['summary']['evaluated_tokens']!=1016 or
                gold['executable_sha256']!=hashes[str(old)] or gold['artifact_sha256']!=model['before_sha256'] or
                gold['tokens_sha256']!=digest(tokens)):
                raise ValueError('Require complete bound prior trained quality')
            hashes.update({str(p):digest(p) for p in (quality,tokens)})
            rows=record['models'][name]=dict(blocks=[],generations=[],controls=[],prior_summary=gold['summary'])
            for i,ids in enumerate(fixture['sequences']):
                a,_=run_native(old,before,[ids])
                b,metrics=run_native(binary,after,[ids])
                exact=np.array_equal(a.view('<u4'),b.view('<u4'))
                rows['blocks'].append(dict(index=i,logits=int(a.size),exact=bool(exact),
                    bf16_tensors=metrics['source_bf16_tensors'],bf16_linear_calls=metrics['source_bf16_linear_calls']))
                if not exact or not metrics['source_bf16_linear_calls']: raise ValueError('Full-block exactness/dispatch failed')
                write_record(report,record); print(name,'block',i+1,'exact',flush=True)
            for case in gold['generations']:
                _,metrics=run_native(binary,after,[case['input_ids']],mode='generate',generate=16)
                exact=metrics['generated_tokens']==case['reference_tokens']
                rows['generations'].append(dict(prompt=case['prompt'],exact=exact,tokens=metrics['generated_tokens']))
                if not exact: raise ValueError('Reference continuation changed')
                write_record(report,record)
            ids=fixture['sequences'][0][:17]
            for mode,scalar,threads in [('verify',True,1),('chunked',False,2),('chunked',True,1)]:
                a,_=run_native(binary,before,[ids],mode=mode,scalar=scalar,threads=threads,chunk=5)
                b,_=run_native(binary,after,[ids],mode=mode,scalar=scalar,threads=threads,chunk=5)
                exact=bool(np.array_equal(a.view('<u4'),b.view('<u4')))
                rows['controls'].append(dict(mode=mode,scalar=scalar,threads=threads,exact=exact))
                if not exact: raise ValueError('Trained scalar/cache/thread exactness failed')
            write_record(report,record); print(name,'reference generation and cache/scalar/thread controls pass',flush=True)
        if any(digest(Path(p))!=h for p,h in hashes.items()): raise ValueError('Quality inputs changed')
        record.update(complete=True,passed=True)
    finally:
        write_record(report,record)


@contextmanager
def power_window():
    stop=threading.Event(); state=dict(polls=0,failures=[])
    def monitor():
        while not stop.is_set():
            try: require_benchmark_ac()
            except Exception as e: state['failures'].append(dict(utc=utc_now(),error=str(e)))
            state['polls']+=1
            stop.wait(0.05)
    state['before']=require_benchmark_ac()
    worker=threading.Thread(target=monitor,daemon=True); worker.start()
    try:
        yield state
    finally:
        stop.set(); worker.join()
        state['after']=require_benchmark_ac()
    if state['failures']: raise ValueError('AC changed within pass; no timing accepted')


def sleep_events(since):
    # Query after measurements, outside every timed region. No-event is valid;
    # inaccessible/failed event-log queries fail closed instead of hiding sleep.
    from datetime import datetime
    stamp=datetime.fromisoformat(since).astimezone().isoformat()
    command = """$ErrorActionPreference='Stop'; try {
      Import-Module (Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Diagnostics/Microsoft.PowerShell.Diagnostics.psd1')
      $events=@(Get-WinEvent -FilterHashtable @{LogName='System';
        ProviderName='Microsoft-Windows-Kernel-Power'; Id=42,107,506,507;
        StartTime=[datetime]::Parse('STAMP')} | Select-Object Id,RecordId,
        @{Name='utc';Expression={$_.TimeCreated.ToUniversalTime().ToString('o')}})
      ConvertTo-Json -InputObject $events -Compress
    } catch { if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') { '[]' }
      else { Write-Error $_; exit 1 } }""".replace('STAMP',stamp)
    result=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',command],
                          check=True,capture_output=True,text=True,timeout=20)
    return json.loads(result.stdout)


def measure(directory,binary,screen,attempt=None):
    name='screen' if screen else 'acceptance'
    if attempt:
        if not attempt.isascii() or not all(c.isalnum() or c=='-' for c in attempt):
            raise ValueError('Attempt must contain only ASCII letters, digits or hyphens')
        name+='-'+attempt
    report=directory/(name+'.json')
    if report.exists(): raise ValueError('Timing record already exists')
    models,hashes=context(directory,binary)
    quality=directory/'quality.json'; q=json.loads(quality.read_text())
    if not q['complete'] or not q['passed'] or q['hashes'][str(binary)]!=digest(binary):
        raise ValueError('Require full exact bound quality before timing')
    old=PRIOR/'leaf_decoder.exe'; hashes.update({str(old):digest(old),str(quality):digest(quality)})
    if not screen:
        s=json.loads((directory/'screen.json').read_text())
        if not s['complete'] or s['hashes'][str(binary)]!=digest(binary): raise ValueError('Require a completed diagnostic screen')
    import psutil
    parent=psutil.Process(); parent.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS); pin_cpu(2)
    os.environ['LEAF_BENCHMARK_REQUIRE_AC']='1'
    if any(v for k,v in os.environ.items() if k.startswith('LEAF_EXPERIMENTAL_') or k=='LEAF_DECODER_PROFILE'):
        raise ValueError('Clear inherited profile/experiment environment')
    runs,warmup=(7,3) if screen else (31,10)
    record=dict(complete=False,release_qualified=False,diagnostic_only=screen,measured_at_utc=utc_now(),
        cpu=2,threads=1,hashes=hashes,policy=dict(decode_ratio_limit=0.98,prefill_ratio_limit=1.02,
        max_p90_p10=1.25,runs=runs,warmup=warmup,pairs=2 if screen else 6),models={})
    write_record(report,record)
    try:
        with keep_awake(display=True):
            for name,model in models['models'].items():
                native_ids=json.loads((ROOT/'build/numerics-20261017/decode-silu'/(name+'-work/tokens.json')).read_text())['benchmark_ids']
                passes=record['models'][name]=dict(passes=[])
                stages=['before','control','after','after','control','before'] if screen else ['before','after','after','before']*3
                for stage in stages:
                    exe=old if stage=='before' else binary
                    artifact=Path(model['after' if stage=='after' else 'before'])
                    started=utc_now()
                    with power_window() as power:
                        _,metrics=run_native(exe,artifact,[native_ids],mode='bench',runs=runs,warmup=warmup,
                            activation_bits=32,windows_above_normal=True)
                    if stage!='before' and not metrics.get('benchmark_ac_checked'): raise ValueError('Native AC guard missing')
                    if stage=='after' and not metrics['source_bf16_linear_calls']: raise ValueError('BF16 path inactive')
                    passes['passes'].append(dict(stage=stage,started_at_utc=started,
                        ended_at_utc=utc_now(),metrics=metrics,power=power))
                    write_record(report,record)
                    print(name,stage,metrics['prefill_p50_ms'],metrics['decode_p50_ms'],flush=True)
                summary={}
                for phase,limit in [('prefill',1.02),('decode',0.98)]:
                    key=phase+'_samples_ms'
                    pooled={s:[v for p in passes['passes'] if p['stage']==s for v in p['metrics'][key]] for s in ('before','after')}
                    values={s:statistics.median(v) for s,v in pooled.items()}
                    ordered={s:[statistics.median(p['metrics'][key]) for p in passes['passes'] if p['stage']==s] for s in pooled}
                    paired=statistics.median(a/b for a,b in zip(ordered['after'],ordered['before']))
                    stable=all(latency_stability(p['metrics'][key])['passed'] for p in passes['passes'])
                    stable=stable and all(latency_stability(v)['passed'] for v in pooled.values())
                    summary[phase]=dict(medians_ms=values,pooled_ratio=values['after']/values['before'],
                        median_paired_ratio=paired,stable=stable,passed=stable and paired<=limit)
                passes['summary']=summary
                write_record(report,record); print(name,json.dumps(summary),flush=True)
        if any(digest(Path(p))!=h for p,h in hashes.items()): raise ValueError('Timing inputs changed')
        record['sleep_audit']=dict(display_guard=True,events=sleep_events(record['measured_at_utc']))
        record['complete']=True
        record['release_qualified']=not screen and not record['sleep_audit']['events'] and all(v['passed'] for m in record['models'].values() for v in m['summary'].values())
    finally:
        write_record(report,record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','check','screen','acceptance'))
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--binary',type=Path)
    parser.add_argument('--attempt',help='fresh timing record suffix; previous attempts are never overwritten')
    args=parser.parse_args(); directory=args.directory.resolve(); directory.mkdir(parents=True,exist_ok=True)
    binary=args.binary.resolve() if args.binary else directory/'accurate-final/leaf_decoder.exe'
    if args.stage=='prepare': prepare(directory)
    elif args.stage=='check': check(directory,binary)
    else: measure(directory,binary,args.stage=='screen',args.attempt)


if __name__=='__main__': main()
