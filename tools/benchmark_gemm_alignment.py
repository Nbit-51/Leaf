"""Audit packed-input alignment on actual MLP shapes; diagnostic, not model acceptance."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.evaluation import write_record
from leaf.power import benchmark_power_mode_guard, keep_awake, require_benchmark_ac
from tools.decoder_validation import latency_stability
from tools.validate_decoder import digest, utc_now


def summarize_records(rows: list[dict], rounds: int, warmup: int) -> list[dict]:
    """Fail closed on partial, reordered or invalid native diagnostic records."""
    shapes = {(m, n, k, copies) for m in (63, 31)
              for n, k in ((2560, 960), (960, 2560), (4864, 896), (896, 4864))
              for copies in (1, 4)}
    if (not isinstance(rows, list) or len(rows) != 17 or
            rows[-1].get('complete') is not True or rows[-1].get('cases') != 16 or
            rows[-1].get('timing_qualified') is not False or
            rows[-1].get('acceptance_timing') is not False or rows[-1] != dict(type='summary', complete=True,
            cases=16, timing_qualified=False, acceptance_timing=False)):
        raise ValueError('Require all sixteen native cases and the final diagnostic summary')
    expected_order = [16 * (3 - phase if r % 2 else phase)
                      for r in range(rounds) for phase in (0, 1, 2, 3, 3, 2, 1, 0)]
    seen, summaries = set(), []
    for case in rows[:-1]:
        key = tuple(case.get(x) for x in ('tokens', 'rows', 'columns', 'weight_copies'))
        if any(type(x) is not int for x in key) or key not in shapes or key in seen:
            raise ValueError('Unexpected or duplicated MLP shape')
        seen.add(key)
        m, n, k, copies = key
        if (case.get('type') != 'case' or case.get('complete') is not True or
                case.get('rounds') != rounds or case.get('warmup_sweeps') != warmup or
                case.get('packing_included') is not False or case.get('fp32_math') != 'fast' or
                case.get('whole_model_speed_claim') is not False or
                case.get('raw_entry') != 'single_noinline_runtime_shape' or
                case.get('bit_exact_values') != 4*m*n*copies or
                case.get('scalar_oracle_positions') != 36*copies or
                case.get('oracle_kind') != 'hardware_scalar_fma_direct_input' or
                case.get('issued_fma_instructions_per_matrix') != ((m+15)//16)*2*n*k or
                case.get('sample_order') != expected_order or
                any(type(x) is not int for x in case.get('sample_order', []))):
            raise ValueError('Native request, exactness, instruction count or sample order differs')
        phases = case.get('phases', [])
        if (len(phases) != 4 or any(type(p.get('packed_mod64')) is not int for p in phases) or
                [p.get('packed_mod64') for p in phases] != [0, 16, 32, 48]):
            raise ValueError('Missing or duplicated controlled alignment phase')
        stats = {}
        for phase in phases:
            samples = phase.get('ms_per_matrix')
            if not isinstance(samples, list) or len(samples) != 2*rounds:
                raise ValueError('Incomplete phase samples')
            stability = latency_stability(samples)
            if stability['p10_ms'] is None:
                raise ValueError('Invalid phase samples')
            stats[phase['packed_mod64']] = dict(median_ms=statistics.median(samples), stability=stability)
        fmas = [case.get('fma_ns_'+side) for side in ('before', 'after')]
        if any(not isinstance(values, list) or len(values) != 3 or
               any(type(x) not in (float, int) or not math.isfinite(x) or x <= 0 for x in values)
               for values in fmas):
            raise ValueError('Invalid adjacent FMA control')
        fma = [statistics.median(v) for v in fmas]
        pair_ratios = [[a/b for a, b in zip(phases[ai]['ms_per_matrix'], phases[bi]['ms_per_matrix'])]
                       for ai, bi in ((0, 1), (2, 3))]
        summaries.append(dict(tokens=m, rows=n, columns=k, weight_copies=copies,
            phases=stats, all_phases_stable=all(p['stability']['passed'] for p in stats.values()),
            original_packed_mod64=case.get('original_packed_mod64'),
            aligned_over_split_paired_ratios=[statistics.median(v) for v in pair_ratios],
            aligned_over_split_pooled_ratio=statistics.median(phases[0]['ms_per_matrix']+phases[2]['ms_per_matrix']) /
                                           statistics.median(phases[1]['ms_per_matrix']+phases[3]['ms_per_matrix']),
            fma_control_before_after_ns=fma, fma_control_larger_over_smaller=max(fma)/min(fma),
            fraction_of_adjacent_fma_control={phase: case['issued_fma_instructions_per_matrix'] *
                statistics.median(fma)*1e-6 / stat['median_ms'] for phase, stat in stats.items()}))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args()
    if sys.platform != 'win32' or args.rounds < 3 or args.warmup < 1:
        parser.error('Require Windows, >=3 balanced rounds and >=1 warmup')
    directory, binary = args.directory.resolve(), args.binary.resolve()
    if not directory.is_relative_to(ROOT/'build') or directory == ROOT/'build':
        parser.error('Use a fresh directory inside build/')
    if not binary.is_file():
        parser.error('Benchmark executable does not exist')
    directory.mkdir(parents=True, exist_ok=False)
    record = dict(format='leaf-gemm-alignment-diagnostic-v1', utc=utc_now(),
        complete=False, timing_qualified=False, acceptance_timing=False,
        cpu=args.cpu, rounds=args.rounds, warmup=args.warmup,
        scope=__doc__, power_mode={}, system_power_settings_changed=False,
        packing_timed=False, ceiling_is_estimate=True,
        ceiling_note='Adjacent register-only FMA throughput is a local control, not a fixed hardware ceiling or GHz measurement.',
        hashes={str(p): digest(p) for p in (binary, Path(__file__),
            ROOT/'benchmark/gemm_alignment_benchmark.cpp', ROOT/'engine/include/leaf/kernels/token_panel.h',
            ROOT/'engine/include/leaf/kernels/row_reuse.h', ROOT/'leaf/power.py')})
    report = directory/'alignment.json'
    write_record(report, record)
    parent = psutil.Process()
    old_affinity, old_priority = parent.cpu_affinity(), parent.nice()
    process = None
    try:
        from tools.benchmark_core_scaling import validate_cores
        from tools.diagnose_windows_benchmark import cpu_sets
        record['topology'] = validate_cores([args.cpu], cpu_sets())
        parent.cpu_affinity([args.cpu])
        parent.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        record['priority'] = 'above-normal'
        record['command'] = [str(binary), str(directory/'raw.jsonl'), str(args.rounds), str(args.warmup)]
        record['ac_before'] = require_benchmark_ac()
        with keep_awake(display=True), benchmark_power_mode_guard(record['power_mode'], required_mode='better-battery'):
            time.sleep(3)
            with (directory/'native.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen(record['command'], stdout=log, stderr=subprocess.STDOUT)
                record['child_pid'] = process.pid
                record['child_affinity'] = psutil.Process(process.pid).cpu_affinity()
                if record['child_affinity'] != [args.cpu]:
                    raise RuntimeError('Child did not inherit the pinned core')
                ac_checks = 0
                while process.poll() is None:
                    require_benchmark_ac()
                    ac_checks += 1
                    time.sleep(1)
                record['ac_checks_during_run'] = ac_checks
                record['native_exit_code'] = process.returncode
                if process.returncode:
                    raise RuntimeError('Native benchmark failed; inspect native.log and retained raw.jsonl')
            record['ac_after'] = require_benchmark_ac()
        if any(digest(Path(p)) != h for p, h in record['hashes'].items()):
            raise RuntimeError('Benchmark executable or source changed during measurement')
        rows = [json.loads(line) for line in (directory/'raw.jsonl').read_text().splitlines()]
        record['raw_records'] = rows
        record['cases'] = summarize_records(rows, args.rounds, args.warmup)
        record['complete'] = True
    except BaseException as error:
        record['error'] = str(error)
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        parent.cpu_affinity(old_affinity)
        parent.nice(old_priority)
        write_record(report, record)
    for case in record['cases']:
        print(case['tokens'], case['rows'], case['columns'], case['weight_copies'],
              round(case['aligned_over_split_pooled_ratio'], 4), case['all_phases_stable'], flush=True)
    print('Completed diagnostic:', report, flush=True)


if __name__ == '__main__':
    main()
