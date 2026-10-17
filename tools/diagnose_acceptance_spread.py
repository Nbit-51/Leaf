"""Replay the acceptance launch path with dispatch counters; never a speed gate.

No PDH monitor, profiling trace, build or file scan should run concurrently.
All samples are retained. Slow samples are described, never removed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.affinity import pin_cpu
from leaf.power import keep_awake
from tools.decoder_validation import benchmark_stability, run_native
from tools.diagnose_windows_benchmark import validate_diagnostic_request
from tools.validate_decoder import digest, utc_now, write_record


def dispatch_summary(metrics: dict, phase: str) -> dict:
    wall = metrics[phase + '_samples_ms']
    cycles = metrics['thread_cycles_' + phase]
    switches = metrics['thread_context_switches_' + phase]
    faults = metrics['process_page_faults_' + phase]
    reasons = metrics['thread_wait_reason_mask_' + phase]
    median_wall, median_cycles = statistics.median(wall), statistics.median(cycles)
    slow = [i for i, value in enumerate(wall) if value > 1.2 * median_wall]
    return dict(median_wall_ms=median_wall, median_cycles=median_cycles,
        median_context_switches=statistics.median(switches), total_page_faults=sum(faults),
        slow_definition='wall time > 1.2 times this pass median; descriptive only, no filtering',
        slow_samples=[dict(index=i, wall_ms=wall[i], cycle_ratio=cycles[i]/median_cycles,
            context_switches=switches[i], page_faults=faults[i], wait_reason_mask=reasons[i]) for i in slow])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'after', 'artifact', 'tokens', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--comparison-cpu', type=int,
                        help='diagnose core placement: AFTER uses this CPU; use the same binary for both stages')
    parser.add_argument('--pairs', type=int, default=6)
    parser.add_argument('--runs', type=int, default=31)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()
    if os.name != 'nt' or args.output.exists():
        parser.error('Require Windows and a fresh output path')
    if args.pairs < 6 or args.pairs % 2 or args.runs < 31 or args.warmup < 10:
        parser.error('Match acceptance: even >=6 pairs, >=31 samples and >=10 warmups')
    if any(value for name,value in os.environ.items() if name.startswith('LEAF_EXPERIMENTAL_') or
           name in ('LEAF_DISABLE_VNNI', 'LEAF_DECODER_PROFILE', 'LEAF_DIAGNOSTIC_START_GATE')):
        parser.error('Clear inherited experiment, profiling and diagnostic gate policies')
    for binary in (args.before, args.after):
        info = json.loads(subprocess.check_output([str(binary.resolve()), '--runtime-info'], text=True))
        if info.get('diagnostic_counter_version') != 4:
            parser.error('Require version-4 LEAF_DIAGNOSTIC_TIMING builds')
    hashes = {str(p): digest(p) for p in (args.before,args.after,args.artifact,args.tokens,Path(__file__),
        ROOT/'tools/decoder_validation.py',ROOT/'tests/native/decoder_diagnostic.h')}
    ids = json.loads(args.tokens.read_text())['benchmark_ids']
    cpu_by_stage = dict(before=args.cpu, after=args.cpu if args.comparison_cpu is None else args.comparison_cpu)
    pin_cpu(args.cpu)
    record = dict(format='leaf-acceptance-dispatch-diagnostic-v1', measured_at_utc=utc_now(),
        cpu=args.cpu, threads=1, runs=args.runs, warmup=args.warmup, hashes=hashes,
        cpu_by_stage=cpu_by_stage,
        launch_path='run_native, inherited affinity, Above Normal at creation; same as benchmark_default_abba',
        scope=__doc__, acceptance_timing=False, automatic_selection_authorized=False,
        complete=False, passes=[])
    write_record(args.output,record)
    try:
        with keep_awake():
            for pair in range(args.pairs):
                for stage in (('before','after') if pair%2==0 else ('after','before')):
                    pin_cpu(cpu_by_stage[stage])
                    started = utc_now()
                    _,metrics = run_native(args.before if stage=='before' else args.after,args.artifact,[ids],
                        mode='bench',threads=1,runs=args.runs,warmup=args.warmup,
                        windows_above_normal=True,activation_bits=32)
                    item = dict(pair=pair,stage=stage,measured_at_utc=started,finished_at_utc=utc_now(),
                                cpu=cpu_by_stage[stage],
                                latency=metrics,request_validated=False)
                    record['passes'].append(item)
                    validate_diagnostic_request(metrics,args.runs,expected_cpu=cpu_by_stage[stage])
                    item.update(request_validated=True,stability=benchmark_stability(metrics),
                        dispatch={phase: dispatch_summary(metrics,phase) for phase in ('prefill','decode')})
                    write_record(args.output,record)
                    print(pair+1,stage,metrics['prefill_p50_ms'],metrics['decode_p50_ms'],
                          item['stability']['passed'],flush=True)
    finally:
        record['finished_at_utc']=utc_now()
        record['inputs_unchanged']=all(digest(Path(p))==h for p,h in hashes.items())
        record['complete']=len(record['passes'])==2*args.pairs and all(p['request_validated'] for p in record['passes'])
        write_record(args.output,record)
    if not record['inputs_unchanged']:
        raise RuntimeError('Diagnostic inputs changed')


if __name__=='__main__':
    main()
