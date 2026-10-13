"""Run and retain the paired attention-span microbenchmark; not model acceptance."""
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
    directory = Path(sys.argv[1])
    raw = directory / 'micro.jsonl'
    report = directory / 'micro.json'
    if raw.exists() or report.exists():
        raise ValueError('Use a fresh directory; preserve previous samples')
    exe = directory / 'micro.exe'
    hashes = {str(p): digest(p) for p in [exe, Path(__file__), ROOT / 'benchmark/attention_span_benchmark.cpp',
                                         ROOT / 'engine/include/leaf/kernels/attention_vector.h']}
    record = dict(format='leaf-attention-span-micro-v1', measured_at_utc=utc_now(), cpu=2, hashes=hashes,
                  complete=False, model_speed_qualified=False)
    report.write_text(json.dumps(record, indent=2) + '\n')
    parent = psutil.Process()
    affinity = parent.cpu_affinity()
    with keep_awake():
        try:
            parent.cpu_affinity([2])
            process = subprocess.Popen([str(exe.resolve()), str(raw.resolve())])
        finally:
            parent.cpu_affinity(affinity)
        if process.wait():
            raise RuntimeError('Attention microbenchmark failed')
    cases = [json.loads(line) for line in raw.read_text().splitlines()]
    for case in cases:
        case['before'] = distribution(case['before_ms'])
        case['after'] = distribution(case['after_ms'])
        case['paired_ratio'] = float(np.median(np.asarray(case['after_ms']) / case['before_ms']))
        print(case['tokens'], case['mask'], round(case['paired_ratio'], 4),
              case['before']['stable'] and case['after']['stable'], flush=True)
    if len(cases) != 12 or any(digest(Path(p)) != sha for p, sha in hashes.items()):
        raise ValueError('Incomplete or changed experiment')
    record.update(complete=True, cases=cases)
    report.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
