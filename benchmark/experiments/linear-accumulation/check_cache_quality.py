"""Strict retained-reference cache, scalar, thread and unchanged-precision checks."""
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


def difference(actual, expected):
    actual, expected = actual.reshape(expected.shape), np.asarray(expected)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Nonfinite logits')
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    ratio = delta / (2e-4 + 2e-4 * np.abs(expected.astype(np.float64)))
    return dict(passed=bool((ratio <= 1).all()), violations=int((ratio > 1).sum()),
                max_abs_error=float(delta.max()), max_tolerance_ratio=float(ratio.max()),
                next_token_agreement=float(np.mean(actual.argmax(-1) == expected.argmax(-1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'build/numerics-20261017/release/leaf_decoder.exe')
    parser.add_argument('--output', type=Path, help='fresh report path; defaults inside the build directory')
    args = parser.parse_args()
    output = args.output or args.build / 'cache-quality.json'
    if output.exists():
        raise ValueError('Refuse to overwrite retained results')
    runtime = args.build / 'leaf_decoder.exe'
    manifest = json.loads((args.build / 'build.json').read_text())
    if not manifest['complete'] or digest(runtime) != manifest['binary_sha256'][runtime.name]:
        raise ValueError('Candidate does not match completed build')
    scratch = args.build / 'tmp'
    scratch.mkdir(exist_ok=True)
    tempfile.tempdir = str(scratch.resolve())
    inputs = {str(p): digest(p) for p in (runtime, args.baseline, Path(__file__))}
    record = dict(complete=False, passed=False, acceptance_timing=False, atol=2e-4, rtol=2e-4,
                  scope='Retained trained-reference prefix/cache/thread/scalar checks and exact W8A8 exclusions',
                  hashes=inputs, cases=[])
    paths = json.loads((ROOT / 'build/numerics-20261017/model-paths.json').read_text())
    try:
        for name, folder in [('smol', 'smol-block1-trace-v2'), ('qwen', 'qwen-block1-trace')]:
            trace_dir = ROOT / 'build/numerics-20261017' / folder
            trace = json.loads((trace_dir / 'trace.json').read_text())
            artifact = Path(paths[name]['artifact'])
            tokens = ROOT / 'build/family-models-20261017' / (name + '-quality-8/tokens.json')
            expected_path = trace_dir / 'expected/logits.npy'
            for p in (artifact, tokens, expected_path, trace_dir / 'trace.json'):
                inputs[str(p)] = digest(p)
            if not trace['complete'] or digest(artifact) != trace['artifact_sha256']:
                raise ValueError('Incomplete trace or changed artifact')
            if digest(tokens) != trace['input_sha256'][str(tokens)]:
                raise ValueError('Changed retained tokens')
            ids = json.loads(tokens.read_text())['sequences'][1][:17]
            reference = np.load(expected_path, mmap_mode='r')
            reference = reference.reshape(-1, reference.shape[-1])[:len(ids)]
            for label, options, length in [('full', {}, 17),
                    ('chunk7', dict(mode='chunked', chunk=7), 17),
                    ('tokenwise-threads2', dict(mode='chunked', chunk=1, threads=2), 17),
                    ('scalar-tokenwise', dict(mode='chunked', chunk=1, scalar=True), 9)]:
                actual, metrics = run_native(runtime, artifact, [ids[:length]], **options)
                assert metrics.get('experimental_accumulation_build') is True
                result = dict(model=name, workload=label, **difference(actual, reference[:length]))
                record['cases'].append(result)
                output.write_text(json.dumps(record, indent=2))
                print(name, label, result['passed'], 'violations', result['violations'], flush=True)
                del actual
            del reference
        for name, directory in [('gpt2', 'build/gpt2_windows_fair'),
                                ('tinyllama', 'build/tinyllama_windows_fair')]:
            task = ROOT / directory
            artifact, tokens, expected_path = task / 'decoder-32.leaf', task / 'tokens.json', task / 'reference.npy'
            for p in (artifact, tokens, expected_path):
                inputs[str(p)] = digest(p)
            ids = json.loads(tokens.read_text())['sequences'][0][:17]
            reference = np.load(expected_path, mmap_mode='r')[:len(ids)]
            actual, _ = run_native(runtime, artifact, [ids])
            result = dict(model=name, workload='retained-trained-prefix', **difference(actual, reference))
            record['cases'].append(result)
            output.write_text(json.dumps(record, indent=2))
            print(name, result['workload'], result['passed'], 'violations', result['violations'], flush=True)
            del actual, reference
        tiny = ROOT / 'build/tinyllama_windows_fair'
        artifact, tokens = tiny / 'decoder-8-smooth.leaf', tiny / 'tokens.json'
        inputs[str(artifact)] = digest(artifact)
        ids = json.loads(tokens.read_text())['sequences'][0][:9]
        for label, options in [('w8a8-tokenwise', dict(mode='chunked', chunk=1)),
                               ('w8a8-generation', dict(mode='generate', generate=4))]:
            left, lm = run_native(args.baseline, artifact, [ids], activation_bits=8, **options)
            right, rm = run_native(runtime, artifact, [ids], activation_bits=8, **options)
            exact = left.shape == right.shape and left.tobytes() == right.tobytes()
            same = lm['generated_tokens'] == rm['generated_tokens']
            record['cases'].append(dict(model='tinyllama', workload=label, passed=exact and same,
                                        logits_exact=exact, generation_exact=same))
            print(label, exact and same, flush=True)
            output.write_text(json.dumps(record, indent=2))
        if any(digest(Path(p)) != h for p, h in inputs.items()):
            raise ValueError('Frozen inputs changed during checks')
        record.update(complete=True, passed=all(row['passed'] for row in record['cases']))
        if not record['passed']:
            raise ValueError('One or more retained-reference checks failed')
    except BaseException as error:
        record['error'] = str(error) or type(error).__name__
        raise
    finally:
        output.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
