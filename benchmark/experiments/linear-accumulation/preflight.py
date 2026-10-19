"""Short serial AB/BA screen. This never qualifies a release or changes a gate."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from leaf.affinity import pin_cpu
from leaf.power import keep_awake, require_benchmark_ac
from tools.decoder_validation import run_native
from tools.validate_decoder import digest, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'after', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model', choices=('smol', 'qwen', 'both'), default='both')
    args = parser.parse_args()
    if os.name != 'nt' or args.output.exists():
        raise ValueError('Require Windows and a fresh output')
    if any(value for key, value in os.environ.items() if key.startswith('LEAF_EXPERIMENTAL_')
           or key in ('LEAF_DISABLE_VNNI', 'LEAF_DECODER_PROFILE')):
        raise ValueError('Clear inherited experimental/profiling policies')
    scratch = args.output.parent / 'tmp'
    scratch.mkdir(exist_ok=True)
    tempfile.tempdir = str(scratch.resolve())
    pin_cpu(2)
    require_benchmark_ac()
    models = ('smol', 'qwen') if args.model == 'both' else (args.model,)
    inputs = [args.before, args.after, Path(__file__), ROOT / 'tools/decoder_validation.py']
    jobs = []
    for name in models:
        work = ROOT / 'build/numerics-20261017/decode-silu' / (name + '-work')
        artifact, tokens = work / 'decoder-32.leaf', work / 'tokens.json'
        inputs.extend((artifact, tokens))
        jobs.append((name, artifact, json.loads(tokens.read_text())['benchmark_ids']))
    hashes = {str(p): digest(p) for p in inputs}
    report = dict(complete=False, acceptance_timing=False, release_qualified=False,
                  scope='Diagnostic only: two balanced pairs, seven samples and three warmups per pass',
                  measured_at_utc=utc_now(), cpu=2, threads=1, hashes=hashes, models={})

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    save()
    try:
        with keep_awake():
            for name, artifact, ids in jobs:
                record = dict(pairs=[])
                report['models'][name] = record
                for index, order in enumerate((('before', 'after'), ('after', 'before'))):
                    pair = dict(order=order)
                    record['pairs'].append(pair)
                    for stage in order:
                        power = require_benchmark_ac()
                        _, metrics = run_native(getattr(args, stage), artifact, [ids], mode='bench',
                                                threads=1, runs=7, warmup=3, activation_bits=32,
                                                windows_above_normal=True)
                        if not metrics.get('optimized_fp32_active'):
                            raise ValueError('Require the normal FP32 dispatch')
                        metrics['power_at_pass_boundaries'] = dict(before=power, after=require_benchmark_ac())
                        pair[stage] = metrics
                        save()
                        print(name, index + 1, stage, metrics['prefill_p50_ms'], metrics['decode_p50_ms'], flush=True)
                record['summary'] = {}
                for phase in ('prefill', 'decode'):
                    medians = {stage: statistics.median(v for pair in record['pairs']
                               for v in pair[stage][phase + '_samples_ms']) for stage in ('before', 'after')}
                    record['summary'][phase] = dict(pooled_median_ms=medians,
                        pooled_ratio=medians['after'] / medians['before'],
                        paired_ratios=[p['after'][phase + '_p50_ms'] / p['before'][phase + '_p50_ms']
                                       for p in record['pairs']])
                print(name, json.dumps(record['summary']), flush=True)
                save()
        if any(digest(Path(p)) != h for p, h in hashes.items()):
            raise ValueError('Input changed during screen')
        report['complete'] = True
    except BaseException as error:
        report['error'] = str(error) or type(error).__name__
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
