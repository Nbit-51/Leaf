"""Bit-exact ordinary-runtime regression, separate from PyTorch qualification."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from leaf.power import keep_awake
from tools.decoder_validation import run_native
from tools.validate_decoder import digest, write_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binary, output = args.binary.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError('Retain existing records; use fresh output')
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = output.parent / 'ordinary-tmp'
    scratch.mkdir(exist_ok=True)
    os.environ['TEMP'] = os.environ['TMP'] = str(scratch)
    prepared = ROOT / 'build/source-bf16-20261020/prepared.json'
    models = json.loads(prepared.read_text())
    baseline = ROOT / 'build/source-bf16-20261020/default/leaf_decoder.exe'
    cases = []
    for name, row in models['models'].items():
        artifact = Path(row['before'])
        if digest(artifact) != row['before_sha256']:
            raise ValueError('Prepared FP32 artifact changed')
        cases.append((name, artifact, ROOT / ('build/numerics-20261017/decode-silu/' + name + '-work/tokens.json'), 32, 33))
    cases.extend([
        ('gpt2', ROOT / 'build/gpt2_decoder/decoder-32.leaf', ROOT / 'build/gpt2_decoder/tokens.json', 32, 33),
        ('tinyllama-w8a8', ROOT / 'build/tinyllama_windows_fair/decoder-8-smooth.leaf', ROOT / 'build/tinyllama_windows_fair/tokens.json', 8, 17),
    ])
    paths = [prepared, binary, baseline, Path(__file__), ROOT / 'tools/decoder_validation.py']
    paths.extend(p for _, artifact, tokens, _, _ in cases for p in (artifact, tokens))
    record = dict(complete=False, passed=False,
        scope='Ordinary runtime regression only; not strict PyTorch qualification',
        binary_sha256=dict(before=digest(baseline), after=digest(binary)),
        hashes={str(p): digest(p) for p in paths}, cases=[])
    write_record(output, record)
    try:
        with keep_awake(display=True):
            for name, artifact, tokens, activation_bits, length in cases:
                ids = json.loads(tokens.read_text())['benchmark_ids'][:length]
                for mode in ('verify', 'chunked', 'generate'):
                    options = dict(mode=mode, chunk=9, generate=4, activation_bits=activation_bits)
                    a, am = run_native(baseline, artifact, [ids], threads=1, **options)
                    b, bm = run_native(binary, artifact, [ids], threads=4, worker_cpus=[2, 4, 6, 8], **options)
                    exact = bool(np.array_equal(a.view('<u4'), b.view('<u4')))
                    tokens_exact = am.get('generated_tokens') == bm.get('generated_tokens')
                    record['cases'].append(dict(model=name, mode=mode, exact=exact, tokens_exact=tokens_exact, logits=int(a.size)))
                    write_record(output, record)
                    print(name, mode, exact, tokens_exact, flush=True)
                    if not (exact and tokens_exact):
                        raise ValueError('Ordinary runtime output changed')
        if any(digest(Path(p)) != h for p, h in record['hashes'].items()):
            raise ValueError('Regression inputs changed')
        record.update(complete=True, passed=True)
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        write_record(output, record)


if __name__ == '__main__':
    main()
