"""Exactness first; serial packing-inclusive scheduling microbenchmark."""
import argparse
import importlib.util
import json
from pathlib import Path
import os
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tools.decoder_validation import latency_stability
from tools.validate_decoder import digest,utc_now,write_record
from leaf.affinity import pin_cpu
from leaf.power import keep_awake,require_benchmark_ac


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('check','timing'))
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--attempt',help='fresh timing suffix; never replace previous attempts')
    parser.add_argument('--cores',type=int,nargs='+',default=[2])
    args=parser.parse_args(); directory=args.directory.resolve()
    if args.attempt and (not args.attempt.isascii() or not all(c.isalnum() or c=='-' for c in args.attempt)):
        raise ValueError('Attempt must use ASCII letters, digits or hyphens')
    report=directory/(args.stage+('-'+args.attempt if args.attempt else '')+'.json')
    if report.exists(): raise ValueError('Choose fresh outputs; retain failed attempts')
    build=json.loads((directory/'build.json').read_text())
    binary=directory/'probe.exe'
    if not build['complete'] or digest(binary)!=build['binary_sha256']:
        raise ValueError('Require a complete unchanged build')
    if any(digest(Path(p))!=h for p,h in build['source_sha256'].items()): raise ValueError('Build inputs changed')
    record=dict(complete=False,passed=False,whole_model_speed_claim=False,utc=utc_now(),
                hashes={str(p):digest(p) for p in (binary,directory/'build.json',Path(__file__))})
    write_record(report,record)
    try:
        if args.stage=='timing':
            quality=json.loads((directory/'check.json').read_text())
            if not quality['complete'] or not quality['passed'] or quality['hashes'][str(binary)]!=digest(binary):
                raise ValueError('Require bound passing exactness first')
            record['hashes'][str(directory/'check.json')]=digest(directory/'check.json')
            import psutil
            from tools.benchmark_core_scaling import validate_cores
            from tools.diagnose_windows_benchmark import cpu_sets
            record['topology']=validate_cores(args.cores,cpu_sets())
            process=psutil.Process(); process.cpu_affinity(args.cores)
            if process.cpu_affinity()!=sorted(args.cores): raise ValueError('Affinity not applied')
            process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            record['cores']=args.cores
            require_benchmark_ac()
        with keep_awake(display=args.stage=='timing'):
            result=subprocess.run([str(binary),args.stage],capture_output=True,text=True,check=True)
        data=json.loads(result.stdout); record['probe']=data
        if args.stage=='timing':
            spec=importlib.util.spec_from_file_location('storage_guard',ROOT/'benchmark/experiments/source-bf16/integration.py')
            helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
            record['sleep_events']=helper.sleep_events(record['utc'])
            summaries=[]
            for shape in data['shapes']:
                passes=shape['passes']
                pooled={s:[v for p in passes if p['stage']==s for v in p['samples_ms']] for s in ('before','after')}
                ordered={s:[statistics.median(p['samples_ms']) for p in passes if p['stage']==s] for s in pooled}
                ratio=statistics.median(a/b for a,b in zip(ordered['after'],ordered['before']))
                stable=all(latency_stability(p['samples_ms'])['passed'] for p in passes)
                stable=stable and all(latency_stability(v)['passed'] for v in pooled.values())
                summaries.append(dict(rows=shape['rows'],columns=shape['columns'],median_paired_ratio=ratio,
                    medians_ms={s:statistics.median(v) for s,v in pooled.items()},stable=stable,improved=stable and ratio<=0.98))
            record['summary']=summaries
            record['passed']=not record['sleep_events'] and all(s['improved'] for s in summaries)
            print(json.dumps(summaries,indent=2),flush=True)
        else:
            record['passed']=data['passed']
            print('Exact scalar/AVX outputs, guards and immutable inputs:',len(data['shapes']),'shapes pass',flush=True)
        if any(digest(Path(p))!=h for p,h in record['hashes'].items()): raise ValueError('Inputs changed during probe')
        record['complete']=True
    except BaseException as error:
        record['error']=str(error); raise
    finally: write_record(report,record)


if __name__=='__main__': main()
