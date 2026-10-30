"""Paired FP32 policy cost diagnostic with optional native CPU-work counters.

This is not release acceptance or a competitor benchmark. Each request performs
prefill on ids[:-1] and decodes the fixed last token, resetting between samples.
Every sample is retained. Raw Windows cycles are never converted to CPU Hz.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaf.evaluation import write_record
from leaf.power import (WINDOWS_POWER_MODE_NAMES, benchmark_power_mode_guard,
                        keep_awake, require_benchmark_ac)
from tools.benchmark_counters import summarize_benchmark_counters
from tools.decoder_validation import benchmark_stability, latency_stability, run_native
from tools.validate_decoder import digest, utc_now


def comparison_summary(passes: list[dict]) -> dict:
    """Describe both raw distributions without accepting a latency claim."""
    if not passes or len(passes) % 4:
        raise ValueError('Require complete balanced AB/BA pairs')
    by_stage = {stage: [p for p in passes if p['stage'] == stage]
                for stage in ('before', 'after')}
    if len(by_stage['before']) != len(by_stage['after']):
        raise ValueError('Unbalanced policy passes')
    pairs = sorted({p['pair'] for p in passes})
    expected = [stage for pair in range(len(pairs))
                for stage in (('before', 'after') if pair % 2 == 0 else ('after', 'before'))]
    if pairs != list(range(len(pairs))) or [p['stage'] for p in passes] != expected:
        raise ValueError('Require the declared balanced AB/BA process order')
    if any(sorted(p['stage'] for p in passes if p['pair'] == pair) != ['after', 'before']
           for pair in pairs):
        raise ValueError('Each pair must contain each policy exactly once')
    result = {}
    for phase in ('prefill', 'decode'):
        key = phase + '_samples_ms'
        pooled = {stage: [v for p in rows for v in p['metrics'][key]]
                  for stage, rows in by_stage.items()}
        if any(latency_stability(v)['p10_ms'] is None for v in pooled.values()):
            raise ValueError('Cannot summarize missing or invalid latency samples')
        stability = {stage: dict(pooled=latency_stability(values),
            individual=[benchmark_stability(p['metrics'])[phase] for p in by_stage[stage]])
            for stage, values in pooled.items()}
        medians = {stage: statistics.median(values) for stage, values in pooled.items()}
        ratios = []
        for pair in pairs:
            one = {p['stage']: p for p in passes if p['pair'] == pair}
            ratios.append(statistics.median(one['after']['metrics'][key]) /
                          statistics.median(one['before']['metrics'][key]))
        result[phase] = dict(medians_ms=medians,
            pooled_ratio=medians['after'] / medians['before'],
            median_paired_ratio=statistics.median(ratios), stability=stability,
            stable=all(v['pooled']['passed'] and all(x['passed'] for x in v['individual'])
                       for v in stability.values()))
    return result


def validate_quality_binding(proof: dict, binary: Path, artifact: Path) -> None:
    """Require a completed full trained audit bound to the exact runtime/input."""
    if proof.get('complete') is not True or proof.get('passed') is not True:
        raise ValueError('Require a complete passing trained quality report')
    hashes = proof.get('hashes', {})
    for path in (binary, artifact):
        if hashes.get(str(path.resolve())) != digest(path):
            raise ValueError('Quality report does not bind the requested binary and artifact')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'artifact', 'tokens', 'quality-report', 'directory'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--before-binary', type=Path, help='Optional prior runtime for an isolated implementation comparison')
    parser.add_argument('--before-quality-report', type=Path)
    parser.add_argument('--before-artifact', type=Path,
                        help='Optional bound prior storage artifact for an exact source-value comparison')
    parser.add_argument('--before-math', choices=('fast', 'accurate'), default='fast')
    parser.add_argument('--after-math', choices=('fast', 'accurate'), default='accurate')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--worker-cpus', type=int, nargs='+')
    parser.add_argument('--pairs', type=int, default=2)
    parser.add_argument('--runs', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--settle-seconds', type=float, default=5)
    parser.add_argument('--without-counters', action='store_true',
                        help='Matched uninstrumented diagnostic; does not change acceptance criteria')
    parser.add_argument('--required-power-mode', choices=WINDOWS_POWER_MODE_NAMES)
    args = parser.parse_args()
    if (args.before_binary is None) != (args.before_quality_report is None):
        parser.error('Provide prior binary and its bound trained quality together')
    if args.before_artifact is not None and args.before_binary is None:
        parser.error('Provide prior binary and its bound quality for a different storage artifact')
    if sys.platform != 'win32':
        parser.error('This power-audited diagnostic currently requires Windows')
    if (args.pairs < 2 or args.pairs % 2 or args.runs < 5 or args.warmup < 1 or
            args.threads < 1 or not math.isfinite(args.settle_seconds) or args.settle_seconds < 0):
        parser.error('Require even >=2 pairs, >=5 samples, >=1 warmup and valid thread/settle settings')
    args.directory = args.directory.resolve()
    if not args.directory.is_relative_to(ROOT / 'build') or args.directory == ROOT / 'build':
        parser.error('Use a fresh directory inside build/')
    return args


def main():
    args = parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    report = args.directory / 'comparison.json'
    record = dict(format='leaf-fp32-policy-cost-v1', utc=utc_now(), complete=False,
        acceptance_timing=False, timing_qualified=False, scope=__doc__, passes=[],
        before_math=args.before_math, after_math=args.after_math, threads=args.threads,
        worker_cpus=args.worker_cpus, counters_enabled=not args.without_counters,
        runs=args.runs, warmup=args.warmup, pairs=args.pairs,
        power_mode={}, system_power_settings_changed=False)
    write_record(report, record)
    process = None
    previous_affinity = None
    prior_ac = os.environ.get('LEAF_BENCHMARK_REQUIRE_AC')
    try:
        scratch = args.directory / 'tmp'
        scratch.mkdir()
        os.environ['TEMP'] = os.environ['TMP'] = str(scratch)
        tempfile.tempdir = str(scratch)
        if any(value for key, value in os.environ.items() if key.startswith('LEAF_EXPERIMENTAL_') or
               key in ('LEAF_DECODER_PROFILE', 'LEAF_DISABLE_VNNI', 'LEAF_DIAGNOSTIC_START_GATE')):
            raise ValueError('Clear inherited experiment, profiling and diagnostic policies')
        binary, artifact = args.binary.resolve(), args.artifact.resolve()
        validate_quality_binding(json.loads(args.quality_report.read_text()), binary, artifact)
        before_artifact = getattr(args, 'before_artifact', None)
        before_artifact = before_artifact.resolve() if before_artifact else artifact
        before_binary = getattr(args, 'before_binary', None)
        before_proof = getattr(args, 'before_quality_report', None)
        if before_binary is not None:
            before_binary = before_binary.resolve()
            validate_quality_binding(json.loads(before_proof.read_text()), before_binary, before_artifact)
        else:
            before_binary = binary
        record['binaries'] = dict(before=str(before_binary), after=str(binary))
        record['artifacts'] = dict(before=str(before_artifact), after=str(artifact))
        for executable in {binary, before_binary}:
            info = json.loads(subprocess.check_output([str(executable), '--runtime-info'], text=True))
            if not {'fast', 'accurate'}.issubset(info.get('fp32_math_modes', [])):
                raise ValueError('Runtime does not support both requested FP32 policies')
            if not args.without_counters and (info.get('benchmark_counters_supported') is not True or
                                             info.get('benchmark_counter_version') != 1):
                raise ValueError('Runtime lacks version-1 optional benchmark counters')
        ids = json.loads(args.tokens.read_text())['benchmark_ids']
        if (not isinstance(ids, list) or len(ids) < 2 or
                any(type(x) is not int or not 0 <= x <= 0xffffffff for x in ids)):
            raise ValueError('Require integer benchmark_ids with prefix and final token')
        if args.worker_cpus is not None:
            from leaf.affinity import validate_worker_cpus
            from tools.benchmark_core_scaling import validate_cores
            from tools.diagnose_windows_benchmark import cpu_sets
            import psutil
            validate_worker_cpus(args.worker_cpus, args.threads)
            record['topology'] = validate_cores(args.worker_cpus, cpu_sets())
            process = psutil.Process()
            previous_affinity = process.cpu_affinity()
            process.cpu_affinity(args.worker_cpus)
            if sorted(process.cpu_affinity()) != sorted(args.worker_cpus):
                raise RuntimeError('Requested diagnostic affinity was not applied')
        paths = [binary, artifact, args.tokens.resolve(), args.quality_report.resolve(),
                 Path(__file__), ROOT/'tools/decoder_validation.py',
                 ROOT/'tools/benchmark_counters.py', ROOT/'leaf/power.py']
        if before_proof is not None:
            paths.extend([before_binary, before_proof.resolve(), before_artifact])
        record['hashes'] = {str(path): digest(path) for path in paths}
        record['request'] = dict(token_count=len(ids), prefill_tokens=len(ids)-1,
                                 decode_tokens=1, reset_each_sample=True, all_token_logits=False)
        os.environ['LEAF_BENCHMARK_REQUIRE_AC'] = '1'
        with keep_awake(display=True), benchmark_power_mode_guard(
                record['power_mode'], required_mode=args.required_power_mode):
            import time
            time.sleep(args.settle_seconds)
            for pair in range(args.pairs):
                for stage in (('before', 'after') if pair % 2 == 0 else ('after', 'before')):
                    policy = args.before_math if stage == 'before' else args.after_math
                    item = dict(pair=pair, stage=stage, fp32_math=policy, started_at_utc=utc_now())
                    record['current_pass'] = item
                    write_record(report, record)
                    item['ac_before'] = require_benchmark_ac()
                    executable = before_binary if stage == 'before' else binary
                    selected_artifact = before_artifact if stage == 'before' else artifact
                    _, metrics = run_native(executable, selected_artifact, [ids], mode='bench',
                        threads=args.threads, worker_cpus=args.worker_cpus,
                        runs=args.runs, warmup=args.warmup, activation_bits=32,
                        windows_above_normal=True, fp32_math=policy,
                        benchmark_counters=not args.without_counters)
                    item.update(metrics=metrics, ended_at_utc=utc_now())
                    record['passes'].append(item)
                    record.pop('current_pass', None)
                    write_record(report, record)
                    item['ac_after'] = require_benchmark_ac()
                    if (metrics.get('fp32_math') != policy or metrics.get('threads') != args.threads or
                            metrics.get('activation_bits') != 32 or not metrics.get('benchmark_ac_checked') or
                            any(len(metrics.get(p+'_samples_ms', [])) != args.runs for p in ('prefill', 'decode'))):
                        raise ValueError('Native diagnostic request/metadata differs')
                    if not args.without_counters:
                        item['cpu_work'] = summarize_benchmark_counters(metrics)
                    item['stability'] = benchmark_stability(metrics)
                    item['validated'] = True
                    write_record(report, record)
                    print(pair+1, stage, policy, metrics['prefill_p50_ms'],
                          metrics['decode_p50_ms'], item['stability']['passed'], flush=True)
        record['summary'] = comparison_summary(record['passes'])
        record['inputs_unchanged'] = all(digest(Path(p)) == h for p, h in record['hashes'].items())
        if not record['inputs_unchanged']:
            raise ValueError('Diagnostic inputs changed')
        record['complete'] = True
        print(json.dumps(record['summary'], indent=2), flush=True)
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        try:
            if process is not None and previous_affinity is not None:
                process.cpu_affinity(previous_affinity)
        except BaseException as error:
            record['complete'] = False
            record['cleanup_error'] = repr(error)
            raise
        finally:
            if prior_ac is None:
                os.environ.pop('LEAF_BENCHMARK_REQUIRE_AC', None)
            else:
                os.environ['LEAF_BENCHMARK_REQUIRE_AC'] = prior_ac
            record['finished_at_utc'] = utc_now()
            write_record(report, record)


if __name__ == '__main__':
    main()
