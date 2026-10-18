"""Preserve completed quality and acceptance evidence without model weights/logit arrays."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.validate_decoder import digest, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build = json.loads((args.build / 'build.json').read_text())
    runtime = args.build / 'leaf_decoder.exe'
    if not build['complete'] or digest(runtime) != build['binary_sha256'][runtime.name]:
        raise ValueError('Require the completed, unchanged final build')
    if any(digest(Path(p)) != h for p, h in build['source_sha256'].items()):
        raise ValueError('Build sources changed; freeze the compiled sources before publishing')
    unit = ET.parse(args.build / 'unit-tests.xml').getroot()
    counts = {key: sum(int(s.get(key, '0')) for s in unit.iter('testsuite'))
              for key in ('tests', 'failures', 'errors', 'skipped')}
    if counts['failures'] or counts['errors'] or counts['tests'] != 1048:
        raise ValueError('Expected completed passing unit validation')
    files = {'build.json': args.build / 'build.json',
             'cache-quality.json': args.build / 'cache-quality.json',
             'quality-binding.json': args.build / 'quality-binding.json',
             'architecture-quality.json': args.build / 'architecture-quality.json'}
    summaries = {}
    for name in ('smol', 'qwen'):
        quality_path = args.build / ('full-quality-' + name) / 'quality.json'
        quality = json.loads(quality_path.read_text())
        if not quality.get('complete') or not quality.get('passed') or quality['executable_sha256'] != digest(runtime):
            raise ValueError('Final strict trained quality did not pass')
        acceptance = args.build / ('acceptance-smol-v2.json' if name == 'smol' else 'acceptance-qwen.json')
        record = json.loads(acceptance.read_text())
        if not record['complete'] or len(record['pairs']) != 6:
            raise ValueError('Acceptance is incomplete; never publish a partial gate as final')
        if any(digest(Path(p)) != h for p, h in record['hashes'].items()):
            raise ValueError('Bound acceptance inputs changed')
        summaries[name] = dict(quality_passed=True, quality_summary=quality['summary'],
                              generations_exact=6, gate=record['gate'])
        files[name + '-quality.json'] = quality_path
        files[name + '-tokens.json'] = quality_path.with_name('tokens.json')
        files[name + '-acceptance.json'] = acceptance
        prior = Path(quality['exact_rebinding']['prior_quality'])
        if digest(prior) != quality['exact_rebinding']['prior_quality_sha256']:
            raise ValueError('Prior fully measured quality changed')
        files[name + '-reference-quality.json'] = prior
    for filename in ('cache-quality.json', 'quality-binding.json'):
        record = json.loads(files[filename].read_text())
        if not record.get('complete') or not record.get('passed'):
            raise ValueError('Final cache/binding checks did not pass')
    architectures = json.loads(files['architecture-quality.json'].read_text())
    if len(architectures['results']) != 10:
        raise ValueError('Incomplete reduced architecture validation')
    prior = ROOT / 'build/linear-accumulation-20261017-v2'
    files.update({'scalar-baseline-control.json': prior / 'scalar-baseline-control.json',
                  'earlier-scalar-failures.json': prior / 'cache-quality-v2.json',
                  'interrupted-smol-acceptance.json': args.build / 'acceptance-smol.json'})
    for name in ('inline', 'cached'):
        directory = ROOT / ('build/linear-accumulation-20261018-' + name)
        for filename in ('build.json', 'probe-correctness.json', 'probe-timing.json',
                         'prior-parity.json', 'assembly-audit.json'):
            files[name + '-' + filename] = directory / filename
    files['cached-edges.json'] = ROOT / 'build/linear-accumulation-20261018-cached/cached-edges.json'
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(format='leaf-shared-accumulation-validation-v1', complete=False,
                  recorded_at_utc=utc_now(), release_default_changed=False,
                  precision='FP32', unit_tests=counts, models=summaries,
                  script_sha256=digest(Path(__file__)), reports={})
    for name, source in files.items():
        shutil.copyfile(source, args.output / name)
        record['reports'][name] = dict(source=str(source), sha256=digest(source))
    record['complete'] = True
    (args.output / 'validation.json').write_text(json.dumps(record, indent=2) + '\n', newline='\n')
    print('Preserved complete accuracy and timing evidence:', args.output, flush=True)


if __name__ == '__main__':
    main()
