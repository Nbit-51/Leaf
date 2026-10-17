"""Paired exact-erf GELU operator experiment; not a model-speed claim."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.evaluation import write_record
from leaf.power import keep_awake
from tools.decoder_validation import latency_stability
from tools.validate_decoder import digest, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=2)
    args = parser.parse_args()
    raw, report = args.directory / 'micro.jsonl', args.directory / 'micro.json'
    if raw.exists() or report.exists():
        parser.error('Use a fresh output directory; completed samples are never overwritten')
    executable = args.directory / 'gelu_erf_benchmark.exe'
    paths = [executable, Path(__file__), ROOT/'benchmark/gelu_erf_benchmark.cpp',
             ROOT/'engine/include/leaf/kernels/gelu_erf.h']
    hashes = {str(p): digest(p) for p in paths}
    record = dict(format='leaf-gelu-erf-micro-v1', measured_at_utc=utc_now(),
                  cpu=args.cpu, threads=1, windows_above_normal=True, hashes=hashes,
                  criteria='Each pass and pooled samples p90/p10 <=1.25; pooled and median paired ratio <=0.95',
                  complete=False, model_speed_qualified=False, cases=[])
    write_record(report, record)
    parent = psutil.Process()
    affinity = parent.cpu_affinity()
    with keep_awake():
        try:
            parent.cpu_affinity([args.cpu])
            process = subprocess.Popen([str(executable.resolve()), str(raw.resolve())],
                                       creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS)
        finally:
            parent.cpu_affinity(affinity)
        if process.wait():
            raise RuntimeError('Native microbenchmark failed; raw samples retained')
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError('Measured inputs changed')
    for line in raw.read_text().splitlines():
        case = json.loads(line)
        pooled = {s: [v for p in case['pairs'] for v in p[s+'_ms']] for s in ('before','after')}
        medians = {s: statistics.median(v) for s, v in pooled.items()}
        ratio = medians['after']/medians['before']
        paired = statistics.median(statistics.median(p['after_ms'])/statistics.median(p['before_ms'])
                                   for p in case['pairs'])
        stability = {s: latency_stability(v) for s,v in pooled.items()}
        every_pass = all(latency_stability(p[s+'_ms'])['passed'] for p in case['pairs'] for s in pooled)
        case['gate'] = dict(medians_ms=medians, pooled_ratio=ratio, median_paired_ratio=paired,
                            pooled_stability=stability, every_pass_stable=every_pass,
                            passed=every_pass and all(v['passed'] for v in stability.values())
                            and ratio<=0.95 and paired<=0.95)
        record['cases'].append(case)
        print(case['elements'],case['distribution'],case['scale'],case['gate'],flush=True)
    record['complete'] = len(record['cases'])==5
    record['passed'] = record['complete'] and all(c['gate']['passed'] for c in record['cases'])
    write_record(report, record)


if __name__ == '__main__':
    main()
