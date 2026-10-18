"""Build and run a matched external BLAS projection screen, never a backend."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from leaf.power import keep_awake, require_benchmark_ac
from tools.validate_decoder import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dll', type=Path)
    args = parser.parse_args()
    if os.name != 'nt':
        raise ValueError('This external DLL probe requires Windows')
    if args.dll is None:
        matches = list((Path(np.__file__).parent.parent / 'numpy.libs').glob('libscipy_openblas64_*.dll'))
        if len(matches) != 1:
            raise ValueError('Specify an exact existing ILP64 OpenBLAS DLL with --dll')
        args.dll = matches[0]
    args.dll = args.dll.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    executable = output / 'blas_probe.exe'
    source = HERE / 'blas_probe.cpp'
    inputs = [source, HERE / 'run_blas_probe.py', HERE / 'tree_sum.h', HERE / 'blocked_sum.h',
              ROOT / 'engine/include/leaf/kernels/row_reuse.h',
              ROOT / 'engine/include/leaf/kernels/token_panel.h', args.dll]
    hashes = {str(p): digest(p) for p in inputs}
    command = ['C:/msys64/mingw64/bin/g++.exe', '-std=c++17', '-O3', '-DNDEBUG', '-static',
               '-static-libgcc', '-static-libstdc++', '-municode', '-DLEAF_ACCUMULATION_INLINE_PARTIAL=1',
               '-I', str(ROOT / 'engine/include'), str(source), '-o', str(executable)]
    record = dict(complete=False, acceptance_timing=False, full_model_speed_qualified=False,
                  hashes=hashes, build_command=command)
    report = output / 'results.json'
    try:
        subprocess.run(command, check=True, cwd=ROOT)
        record['executable_sha256'] = digest(executable)
        with keep_awake():
            record['power_before'] = require_benchmark_ac()
            process = subprocess.run([str(executable), str(args.dll)], capture_output=True, text=True)
            record['power_after'] = require_benchmark_ac()
        (output / 'stdout.txt').write_text(process.stdout)
        (output / 'stderr.txt').write_text(process.stderr)
        process.check_returncode()
        native = json.loads(process.stdout)
        for shape in native['shapes']:
            medians = {name: statistics.median(v for p in shape['passes'] if p['algorithm'] == name
                                               for v in p['ms'])
                       for name in ('leaf_release', 'leaf_accurate_tree', 'openblas')}
            shape['median_ms'] = medians
            shape['openblas_over_leaf_release'] = medians['openblas'] / medians['leaf_release']
            print(shape['m'], shape['k'], shape['n'], medians, flush=True)
        if any(digest(Path(p)) != checksum for p, checksum in hashes.items()):
            raise ValueError('Input changed during matched screen')
        record.update(complete=True, native=native)
    except BaseException as error:
        record['error'] = str(error) or type(error).__name__
        raise
    finally:
        report.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
