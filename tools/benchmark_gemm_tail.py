"""Run/inspect the paired tail-tile microbenchmark; never a whole-model speed claim."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.power import keep_awake
from tools.benchmark_embedding_competitor import distribution
from tools.validate_decoder import digest, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('build/gemm-tail-20261011'))
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    raw = args.directory / 'micro.jsonl'
    report = args.directory / 'micro.json'
    if not args.inspect:
        if raw.exists() or report.exists():
            parser.error('Existing samples: use --inspect or a fresh directory; never overwrite')
        paths = [args.directory / 'gemm_tail_benchmark.exe', Path(__file__),
                 ROOT / 'benchmark/gemm_tail_benchmark.cpp', ROOT / 'engine/include/leaf/kernels/row_reuse.h']
        record = dict(format='leaf-gemm-tail-micro-v1', measured_at_utc=utc_now(), cpu=args.cpu,
                      hashes={str(p): digest(p) for p in paths}, complete=False,
                      model_speed_qualified=False, cases=[])
        report.write_text(json.dumps(record, indent=2) + '\n')
        parent = psutil.Process()
        affinity = parent.cpu_affinity()
        with keep_awake():
            try:
                parent.cpu_affinity([args.cpu])
                process = subprocess.Popen([str(paths[0].resolve()), str(raw.resolve())])
            finally:
                parent.cpu_affinity(affinity)
            if process.wait():
                raise RuntimeError('Microbenchmark failed; partial raw records retained')
        if any(digest(Path(p)) != h for p, h in record['hashes'].items()):
            raise ValueError('Experiment inputs changed')
    else:
        record = json.loads(report.read_text())
    cases = []
    for line in raw.read_text().splitlines():
        case = json.loads(line)
        case['before'] = distribution(case['before_ms'])
        case['after'] = distribution(case['after_ms'])
        case['median_paired_ratio'] = float(np.median(np.asarray(case['after_ms']) / case['before_ms']))
        cases.append(case)
        print(case['tokens'], case['rows'], case['columns'], round(case['median_paired_ratio'], 4),
              case['before']['stable'] and case['after']['stable'], flush=True)
    record['cases'] = cases
    record['complete'] = len(cases) == 36
    report.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
