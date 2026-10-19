"""Test source-preserving BF16 storage on real tied vocabulary weights.

Isolated GEMV probe, not model speed or artifact-format support. No source
weight rounding: only an already BF16 checkpoint is accepted.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from leaf.evaluation import write_record
from leaf.power import keep_awake, require_benchmark_ac
from tools.decoder_validation import latency_stability
from tools.export_decoder import tensor_index
from tools.validate_decoder import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true', help='exact output comparison, without timing or AC requirement')
    args = parser.parse_args()
    if not args.check_only:
        require_benchmark_ac()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('probe.cpp')
    executable = output/'probe.exe'
    command = ['C:/msys64/mingw64/bin/g++.exe', '-std=c++17', '-O3', '-DNDEBUG', '-static',
               '-I', str(ROOT/'engine/include'), str(source), '-o', str(executable)]
    record = dict(complete=False, acceptance_timing=False, model_speed_claim=False,
        scope=__doc__, command=command, hashes={str(p):digest(p) for p in
        (Path(__file__), source, ROOT/'engine/include/leaf/kernels/gemv_pair.h')}, models={})
    report = output/'results.json'
    write_record(report, record)
    subprocess.run(command, check=True)
    record['hashes'][str(executable)] = digest(executable)
    paths = json.loads((ROOT/'build/numerics-20261017/model-paths.json').read_text())
    import numpy as np
    for name in ('smol', 'qwen'):
        weight_file = Path(paths[name]['snapshot'])/'model.safetensors'
        record['hashes'][str(weight_file)] = digest(weight_file)
        base, index = tensor_index(weight_file)
        entry = index['model.embed_tokens.weight']
        if entry['dtype'] != 'BF16' or len(entry['shape']) != 2:
            raise ValueError('Probe requires original BF16 vocabulary weights; no lossy conversion')
        raw = output/(name+'-head.bf16')
        values = np.memmap(weight_file, mode='r', dtype='<u2', shape=tuple(entry['shape']),
                           offset=base+entry['data_offsets'][0])
        if np.any((values & 0x7f80) == 0x7f80):
            raise ValueError('Source tensor contains nonfinite weights')
        values.tofile(raw)
        record['hashes'][str(raw)] = digest(raw)
        del values
        with keep_awake():
            if not args.check_only:
                require_benchmark_ac()
            result = subprocess.run([str(executable),str(raw),*map(str,entry['shape']),
                                    *(['--check-only'] if args.check_only else [])],
                                    capture_output=True,text=True)
            if not args.check_only:
                require_benchmark_ac()
        (output/(name+'-stdout.json')).write_text(result.stdout)
        (output/(name+'-stderr.txt')).write_text(result.stderr)
        result.check_returncode()
        native = json.loads(result.stdout)
        if args.check_only:
            record['models'][name] = native
            write_record(report, record)
            print(name, 'exact_fp32_output', native['exact_fp32_output'], 'no timing', flush=True)
            continue
        samples = {s:[v for p in native['passes'] if p['stage']==s for v in p['ms']]
                   for s in ('fp32','bf16')}
        medians = {s:statistics.median(v) for s,v in samples.items()}
        native['medians_ms'] = medians
        native['ratio'] = medians['bf16']/medians['fp32']
        native['pooled_stability'] = {s:latency_stability(v) for s,v in samples.items()}
        native['every_pass_stable'] = all(latency_stability(p['ms'])['passed'] for p in native['passes'])
        record['models'][name] = native
        write_record(report, record)
        print(name, medians, 'ratio',native['ratio'], 'every_pass_stable',native['every_pass_stable'],flush=True)
    if any(digest(Path(p)) != h for p,h in record['hashes'].items()):
        raise ValueError('Probe inputs changed')
    record['complete'] = True
    write_record(report, record)


if __name__ == '__main__':
    main()
