"""Rebind retained trained quality only after exact logits on every old block and continuation."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.decoder_validation import run_native
from tools.validate_decoder import digest, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    args = parser.parse_args()
    runtime, before = args.build / 'leaf_decoder.exe', args.prior / 'leaf_decoder.exe'
    manifest = json.loads((args.build / 'build.json').read_text())
    if not manifest['complete'] or digest(runtime) != manifest['binary_sha256'][runtime.name]:
        raise ValueError('Require a completed matching new build')
    scratch = args.build / 'tmp'
    scratch.mkdir(exist_ok=True)
    tempfile.tempdir = str(scratch.resolve())
    output = args.build / 'quality-binding.json'
    if output.exists():
        raise ValueError('Refuse to overwrite retained binding evidence')
    hashes = {str(p): digest(p) for p in (runtime, before, Path(__file__))}
    record = dict(complete=False, passed=False, acceptance_timing=False,
                  scope='Exact new/default-AVX logits on all eight old quality blocks plus all six continuations per model',
                  hashes=hashes, cases=[])
    try:
        for name in ('smol', 'qwen'):
            source = args.prior / ('full-quality-' + name) / 'quality.json'
            old = json.loads(source.read_text())
            if (old.get('complete') is not True or old.get('passed') is not True
                    or old['executable_sha256'] != digest(before) or len(old['blocks']) != 8
                    or len(old['generations']) != 6):
                raise ValueError('Prior trained quality is incomplete or unbound')
            artifact = Path(old['snapshot'])
            for filename, checksum in old['source_sha256'].items():
                path = artifact / filename
                if digest(path) != checksum:
                    raise ValueError('Prior model source changed')
                hashes[str(path)] = checksum
            cache = json.loads((ROOT / 'build/numerics-20261017/model-paths.json').read_text())
            artifact = Path(cache[name]['artifact'])
            tokens = source.with_name('tokens.json')
            if digest(artifact) != old['artifact_sha256'] or digest(tokens) != old['tokens_sha256']:
                raise ValueError('Prior quality inputs changed')
            for p in (source, artifact, tokens):
                hashes[str(p)] = digest(p)
            rows = json.loads(tokens.read_text())['sequences']
            for index, ids in enumerate(rows):
                left, _ = run_native(before, artifact, [ids])
                right, metrics = run_native(runtime, artifact, [ids])
                exact = left.shape == right.shape and left.tobytes() == right.tobytes()
                if not metrics['avx2'] or not metrics.get('experimental_accumulation_build') or not exact:
                    raise ValueError('Default-AVX implementation changed trained logits')
                record['cases'].append(dict(model=name, block=index, exact=True))
                output.write_text(json.dumps(record, indent=2))
                print(name, 'bound quality block', index + 1, '/ 8', flush=True)
                del left, right
            for index, generation in enumerate(old['generations']):
                _, metrics = run_native(runtime, artifact, [generation['input_ids']], mode='generate',
                                        generate=old['generation_policy']['max_new_tokens'])
                if metrics['generated_tokens'] != generation['reference_tokens']:
                    raise ValueError('New generated tokens differ from the retained reference')
                record['cases'].append(dict(model=name, generation=index, exact=True))
                output.write_text(json.dumps(record, indent=2))
                print(name, 'bound generation', index + 1, '/ 6', flush=True)
            destination = args.build / ('full-quality-' + name)
            destination.mkdir(exist_ok=False)
            shutil.copyfile(tokens, destination / 'tokens.json')
            derived = copy.deepcopy(old)
            derived.update(executable_sha256=digest(runtime), measured_at_utc=utc_now(),
                evaluator_sha256=digest(Path(__file__)),
                exact_rebinding=dict(prior_quality=str(source), prior_quality_sha256=digest(source),
                    prior_evaluator_sha256=old['evaluator_sha256'],
                    method='Every full block matches the old native logits byte-for-byte; all six continuations match saved PyTorch tokens',
                    binding_report=str(output)))
            (destination / 'quality.json').write_text(json.dumps(derived, indent=2) + '\n')
        if any(digest(Path(p)) != h for p, h in hashes.items()):
            raise ValueError('Bound inputs changed during execution')
        record.update(complete=True, passed=True)
    except BaseException as error:
        record['error'] = str(error) or type(error).__name__
        raise
    finally:
        output.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
