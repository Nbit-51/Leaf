"""Matched FP32 Leaf/llama.cpp decoder baseline with retained numerical failures."""
import argparse
import gc
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leaf.affinity import pin_cpu
from leaf.evaluation import logit_diagnostics, write_record
from leaf.power import keep_awake
from tools.decoder_validation import benchmark_stability, latency_stability, quality, run_native
from tools.validate_decoder import digest, native_policy, utc_now


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('quality-record', 'tokens', 'leaf-executable', 'leaf-artifact', 'llama-executable', 'gguf', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--cpu', type=int, default=2)
    p.add_argument('--rounds', type=int, default=6)
    p.add_argument('--runs', type=int, default=11)
    p.add_argument('--warmups', type=int, default=3)
    p.add_argument('--diagnostic-timing', action='store_true',
                   help='Retain timings despite numerical mismatch; never qualifies an equal-quality speed claim')
    args = p.parse_args()
    if args.output.exists() or args.rounds < 1 or args.runs < 1 or args.warmups < 0:
        p.error('Use positive counts and a fresh output directory')
    import os
    if any(value and (name.startswith('LEAF_EXPERIMENTAL_') or name in
                     ('LEAF_DECODER_PROFILE', 'LEAF_DISABLE_VNNI')) for name, value in os.environ.items()):
        raise ValueError('Unset profiling/experimental kernel overrides before comparison')
    args.output.mkdir(parents=True)
    saved = json.loads(args.quality_record.read_text())
    if digest(args.tokens) != saved['tokens_sha256'] or digest(args.leaf_artifact) != saved['artifact_sha256']:
        raise ValueError('Saved fixture/artifact identity mismatch')
    snapshot = Path(saved['snapshot'])
    for name, checksum in saved['source_sha256'].items():
        if digest(snapshot / name) != checksum:
            raise ValueError('Source snapshot changed')
    row = json.loads(args.tokens.read_text())['sequences'][0]
    if len(row) < 128:
        raise ValueError('Need a saved 128-token fixture')
    sequences = [row[:64], row[:128]]
    paths = [args.quality_record, args.tokens, args.leaf_executable, args.leaf_artifact,
             args.llama_executable, args.gguf, Path(__file__), Path('benchmark/llama_decoder_benchmark.cpp')]
    paths.extend(sorted(args.llama_executable.parent.glob('*.dll')))
    hashes = {str(path): digest(path) for path in paths}
    record = dict(format='leaf-llama-decoder-baseline-v1', complete=False, model=saved['model'],
                  measured_at_utc=utc_now(), platform=platform.platform(), cpu=platform.processor(),
                  cpu_affinity=[args.cpu], threads=1, weight_precision='FP32', kv_precision='FP32', gpu_layers=0,
                  lengths=[63, 127], native_policy=native_policy(), hashes=hashes,
                  scope='Warm prefix to last-token logits followed by one teacher-forced cached decode. '
                        'Reset/load/tokenization/IO/HTTP excluded. No sustained generation measurement.',
                  warmups=args.warmups, runs=args.runs, rounds=args.rounds,
                  stability_p90_p10_limit=1.25, required_leaf_over_llama=0.98,
                  preflight=[], passes=[], overall_speed_claim=False)
    record['diagnostic_timing_despite_mismatch'] = args.diagnostic_timing
    report = args.output / 'comparison.json'
    def save():
        write_record(report, record)
    save()
    pin_cpu(args.cpu) # Both engines inherit the identical affinity, before model load.

    def execute(engine, index, mode, label, runs, warmups):
        ids = sequences[index]
        if engine == 'leaf':
            values, measured = run_native(args.leaf_executable, args.leaf_artifact, [ids],
                                         mode=mode, runs=runs, warmup=warmups)
            write_record(args.output / (label + '.json'), measured)
            return values, measured
        request = args.output / (label + '.tokens')
        request.write_bytes(struct.pack('<II', 1, len(ids)) + np.asarray(ids, dtype='<u4').tobytes())
        output, metrics = args.output / (label + '.f32'), args.output / (label + '.json')
        command = [str(args.llama_executable.resolve()), str(args.gguf.resolve()), str(request.resolve()),
                   str(output.resolve()), str(metrics.resolve()), mode, '1', str(runs), str(warmups),
                   str(args.llama_executable.parent.resolve())]
        with (args.output / (label + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=log, check=True)
        return np.fromfile(output, dtype='<f4'), json.loads(metrics.read_text())

    try:
        # Reference generation happens before timing, then weights are released.
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        torch.set_num_threads(1)
        model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                                                   dtype=torch.float32, attn_implementation='eager').eval()
        with torch.inference_mode():
            expected = [model(torch.tensor([ids]), use_cache=False).logits[0].numpy().copy() for ids in sequences]
        del model
        gc.collect()
        record.update(torch=torch.__version__, transformers=transformers.__version__)
        for index, ids in enumerate(sequences):
            for engine in ('leaf', 'llama'):
                values, measured = execute(engine, index, 'verify', f'preflight-{index}-{engine}', 1, 0)
                actual = values.reshape(expected[index].shape)
                fidelity = logit_diagnostics(actual, expected[index], [ids])
                checked = dict(engine=engine, prefix_tokens=len(ids) - 1, diagnostics=fidelity,
                               quality=quality(actual, expected[index], [ids]),
                               logits_close=fidelity['logit_tolerance_violations'] == 0)
                record['preflight'].append(checked)
                save()
                # A gross mismatch is not a runnable matched model. Smaller roundoff
                # failures remain visible and allow diagnostic timing, never qualification.
                if checked['quality']['next_token_agreement'] < 0.95 and not args.diagnostic_timing:
                    raise ValueError('Gross model mismatch; see retained preflight')
                print('preflight', len(ids)-1, engine, 'logits_close', checked['logits_close'], flush=True)
        del expected
        gc.collect()
        for iteration in range(args.rounds):
            for engine in (('leaf', 'llama') if iteration % 2 == 0 else ('llama', 'leaf')):
                for index, ids in enumerate(sequences):
                    _, measured = execute(engine, index, 'bench', f'{iteration+1}-{index}-{engine}',
                                          args.runs, args.warmups)
                    for phase in ('prefill', 'decode'):
                        if len(measured[phase + '_samples_ms']) != args.runs:
                            raise ValueError('Incomplete timing samples')
                    checked = dict(engine=engine, round=iteration+1, prefix_tokens=len(ids)-1,
                                   metrics=measured, stability=benchmark_stability(measured))
                    record['passes'].append(checked)
                    save()
                    print(iteration+1, len(ids)-1, engine, measured['prefill_p50_ms'],
                          measured['decode_p50_ms'], checked['stability']['passed'], flush=True)
        summary = []
        for length in record['lengths']:
            item = dict(prefix_tokens=length, phases={})
            passes = {engine: [x for x in record['passes'] if x['engine']==engine and x['prefix_tokens']==length]
                      for engine in ('leaf', 'llama')}
            for phase in ('prefill', 'decode'):
                field = phase + '_samples_ms'
                pooled = {}
                for engine in passes:
                    values = [v for x in passes[engine] for v in x['metrics'][field]]
                    pooled[engine] = dict(**latency_stability(values), median_ms=float(np.median(values)))
                ratios = [np.median(a['metrics'][field])/np.median(b['metrics'][field])
                          for a,b in zip(passes['leaf'], passes['llama'])]
                stable = all(x['stability'][phase]['passed'] for rows in passes.values() for x in rows)
                stable = stable and all(x['passed'] for x in pooled.values())
                item['phases'][phase] = dict(pooled=pooled, median_paired_leaf_over_llama=float(np.median(ratios)),
                                             stable=bool(stable), observed_leaf_faster=bool(np.median(ratios)<1),
                                             timing_win_gate=bool(stable and np.median(ratios)<=.98
                                                                  and args.rounds>=6 and args.runs>=11))
            summary.append(item)
        if any(digest(Path(path)) != value for path,value in hashes.items()):
            raise ValueError('Benchmark input changed')
        record.update(complete=True, summary=summary,
                      numerical_preflight_passed=all(x['logits_close'] for x in record['preflight']))
        # This bounded baseline does not supersede the failed larger quality suite.
        save()
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        record['error'] = str(error) or type(error).__name__
        save()
        raise


if __name__ == '__main__':
    with keep_awake():
        main()
