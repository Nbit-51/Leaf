"""Check an implementation-only revision against the retained candidate outputs."""
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
    parser.add_argument('--prior', type=Path, required=True)
    args = parser.parse_args()
    output = args.build / 'prior-parity.json'
    if output.exists():
        raise ValueError('Refuse to overwrite retained results')
    previous = json.loads((args.prior / 'saved-blocks.json').read_text())
    manifest = json.loads((args.build / 'build.json').read_text())
    runtime = args.build / 'leaf_decoder.exe'
    if (not previous['complete'] or not manifest['complete']
            or digest(runtime) != manifest['binary_sha256'][runtime.name]):
        raise ValueError('Incomplete or changed candidate evidence')
    for path, checksum in previous['input_sha256'].items():
        # The prior tool and generated-build manifest are preserved on disk.
        if digest(Path(path)) != checksum:
            raise ValueError('Prior input changed: ' + path)
    for path, checksum in manifest['source_sha256'].items():
        if digest(Path(path)) != checksum:
            raise ValueError('Candidate source changed: ' + path)
    scratch = args.build / 'tmp'
    scratch.mkdir(exist_ok=True)
    tempfile.tempdir = str(scratch.resolve())
    paths = json.loads((ROOT / 'build/numerics-20261017/model-paths.json').read_text())
    hashes = {str(runtime): digest(runtime), str(Path(__file__)): digest(Path(__file__))}
    report = dict(complete=False, passed=False, acceptance_timing=False,
                  full_quality_qualified=False, scope='Exact old-candidate logits on two retained 128-token blocks only',
                  input_sha256=hashes, cases=[])
    try:
        for name in ('smol', 'qwen'):
            artifact = Path(paths[name]['artifact'])
            tokens = ROOT / 'build/family-models-20261017' / (name + '-quality-8/tokens.json')
            old_logits = args.prior / (name + '-block1-logits.npy')
            for path in (artifact, tokens, old_logits, args.prior / 'saved-blocks.json'):
                hashes[str(path)] = digest(path)
            ids = json.loads(tokens.read_text())['sequences'][1]
            expected = np.load(old_logits, mmap_mode='r')
            actual, metrics = run_native(runtime, artifact, [ids])
            assert metrics.get('experimental_accumulation_build') is True
            actual = actual.reshape(expected.shape)
            exact = actual.tobytes() == expected.tobytes()
            report['cases'].append(dict(model=name, exact=exact, values=int(actual.size),
                                        prior_quality=previous['results'][name]))
            output.write_text(json.dumps(report, indent=2))
            print(name, 'exact prior-candidate logits:', exact, flush=True)
            if not exact:
                raise ValueError('Implementation revision changed candidate arithmetic')
            del actual, expected
        if any(digest(Path(path)) != checksum for path, checksum in hashes.items()):
            raise ValueError('Inputs changed during parity check')
        report.update(complete=True, passed=True)
    except BaseException as error:
        report['error'] = str(error) or type(error).__name__
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
