"""Bind trained accumulation quality to the unchanged serial ABBA gate."""
import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.validate_decoder import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refuse to overwrite retained timing evidence')
    quality = json.loads(args.quality.read_text())
    expected_gates = dict(min_next_token_agreement=0.999, max_perplexity_ratio=1.001,
                          logit_atol=2e-4, logit_rtol=2e-4, exact_generation_required=True)
    if (quality.get('complete') is not True or quality.get('passed') is not True
            or quality.get('precision') != 'FP32' or quality.get('requested_blocks') != 8
            or quality.get('sequence_length') != 128 or quality.get('gates') != expected_gates
            or quality.get('executable_sha256') != digest(args.after)
            or quality.get('artifact_sha256') != digest(args.workdir / 'decoder-32.leaf')
            or len(quality.get('generations', [])) != 6
            or not all(row['exact_match'] for row in quality['generations'])):
        raise ValueError('Require the complete strict eight-block trained-quality pass for this candidate/artifact')
    tokens = args.quality.with_name('tokens.json')
    if digest(tokens) != quality['tokens_sha256']:
        raise ValueError('Quality token fixture changed')
    heldout = json.loads(tokens.read_text())['sequences']
    ids = json.loads((args.workdir / 'tokens.json').read_text())['benchmark_ids']
    if ids != heldout[0][:64]:
        raise ValueError('Acceptance prefix differs from the held-out fixture')
    additional = {str(p): digest(p) for p in (args.quality, tokens, Path(__file__))}
    if quality.get('exact_rebinding'):
        binding = Path(quality['exact_rebinding']['binding_report'])
        rebound = json.loads(binding.read_text())
        if not rebound.get('complete') or not rebound.get('passed'):
            raise ValueError('Quality rebinding did not complete successfully')
        additional[str(binding)] = digest(binding)
    sys.argv = ['benchmark_default_abba', '--before', str(args.before), '--after', str(args.after),
                '--workdir', str(args.workdir), '--output', str(args.output), '--target-phase', 'prefill',
                '--cpu', '2', '--pairs', '6', '--runs', '31', '--warmup', '10']
    runpy.run_path(str(ROOT / 'tools/benchmark_default_abba.py'), run_name='__main__')
    result = json.loads(args.output.read_text())
    if any(digest(Path(p)) != h for p, h in additional.items()):
        raise ValueError('Bound quality inputs changed during acceptance')
    result['bound_accumulation_quality'] = dict(quality=str(args.quality),
        quality_sha256=additional[str(args.quality)], complete=True, passed=True,
        scope='Eight length-128 blocks and six greedy reference continuations; cache/thread checked separately')
    result['hashes'].update(additional)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
