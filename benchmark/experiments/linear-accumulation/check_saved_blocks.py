"""Check only the two retained failing 128-token blocks; no new downloads or timing."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.decoder_validation import run_native
from tools.validate_decoder import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=ROOT / 'build/numerics-20261017')
    parser.add_argument('--quality-root', type=Path, default=ROOT / 'build/family-models-20261017')
    args = parser.parse_args()
    output = args.build / 'saved-blocks.json'
    if output.exists():
        raise ValueError('Refuse to overwrite retained results')
    build = json.loads((args.build / 'build.json').read_text())
    runtime = args.build / 'leaf_decoder.exe'
    if not build['complete'] or digest(runtime) != build['binary_sha256'][runtime.name]:
        raise ValueError('Require a completed, matching candidate build')
    if any(digest(Path(p)) != h for p, h in build['source_sha256'].items()):
        raise ValueError('Candidate sources changed since build')
    temp = args.build / 'tmp'
    temp.mkdir(exist_ok=True)
    tempfile.tempdir = str(temp.resolve())
    paths = json.loads((args.fixtures / 'model-paths.json').read_text())
    inputs = {runtime: digest(runtime), Path(__file__): digest(Path(__file__)),
              args.build / 'build.json': digest(args.build / 'build.json')}
    jobs = []
    for name, folder in [('smol', 'smol-block1-trace-v2'), ('qwen', 'qwen-block1-trace')]:
        quality = args.quality_root / (name + '-quality-8/quality.json')
        tokens = quality.with_name('tokens.json')
        record = json.loads(quality.read_text())
        artifact = Path(paths[name]['artifact'])
        expected = args.fixtures / folder / 'expected/logits.npy'
        trace_file = args.fixtures / folder / 'trace.json'
        for path in (quality, tokens, artifact, expected, trace_file):
            inputs[path] = digest(path)
        if inputs[tokens] != record['tokens_sha256'] or inputs[artifact] != record['artifact_sha256']:
            raise ValueError('Frozen token/artifact identity mismatch')
        trace = json.loads(trace_file.read_text())
        if (not trace['complete'] or not trace['instrumentation_bit_exact'] or trace['block'] != 1
                or trace['quality_record_sha256'] != inputs[quality]):
            raise ValueError('Expected arrays lack a complete bound reference trace')
        jobs.append((name, artifact, tokens, expected))
    report = dict(complete=False, atol=2e-4, rtol=2e-4, full_quality_qualified=False,
        release_default_changed=False, acceptance_timing=False,
        scope='Frozen failing block 1 per model only; no full trained-model gate or latency claim',
        input_sha256={str(p): h for p, h in inputs.items()}, results={})
    output.write_text(json.dumps(report, indent=2))
    try:
        for name, artifact, tokens, expected_file in jobs:
            ids = json.loads(tokens.read_text())['sequences'][1]
            expected = np.load(expected_file, mmap_mode='r').reshape(len(ids), -1)
            actual, metrics = run_native(runtime, artifact, [ids])
            assert metrics.get('experimental_accumulation_build') is True
            actual = actual.reshape(expected.shape)
            np.save(args.build / (name + '-block1-logits.npy'), actual)
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ValueError('Nonfinite trained output')
            result = dict(block=1, tokens=len(ids), violations=0, max_abs_error=0.0,
                          max_tolerance_ratio=0.0, row0_violations=0,
                          next_token_agreement=float(np.mean(actual[:-1].argmax(-1) == expected[:-1].argmax(-1))))
            for first in range(0, len(ids), 8):
                a = actual[first:first+8].astype(np.float64)
                e = expected[first:first+8].astype(np.float64)
                delta = np.abs(a - e)
                ratios = delta / (2e-4 + 2e-4 * np.abs(e))
                result['violations'] += int((ratios > 1).sum())
                result['max_abs_error'] = max(result['max_abs_error'], float(delta.max()))
                if first == 0:
                    result['row0_violations'] = int((ratios[0] > 1).sum())
                index = np.unravel_index(np.argmax(ratios), ratios.shape)
                if float(ratios[index]) > result['max_tolerance_ratio']:
                    result['max_tolerance_ratio'] = float(ratios[index])
                    result['worst'] = dict(row=first + int(index[0]), token=int(index[1]),
                                          native=float(a[index]), reference=float(e[index]))
            result['allclose'] = result['violations'] == 0
            report['results'][name] = result
            output.write_text(json.dumps(report, indent=2))
            print(name, json.dumps(result), flush=True)
            del actual, expected
        if any(digest(p) != h for p, h in inputs.items()):
            raise ValueError('Frozen input changed during execution')
        report['complete'] = True
    except BaseException as error:
        report['error'] = str(error) or type(error).__name__
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
