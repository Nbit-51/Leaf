"""Compare cached Leaf and llama.cpp FP32 CLI latency on Windows.

Prepare and audit both artifacts against the same checkpoint before running.
Use a held-out token fixture and fresh output directory. This measures fresh
processes with a warm OS file cache and checks exact continuation text and work
counts. It does not establish full-logit parity or measure peak memory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

from leaf.affinity import pin_cpu
from leaf.cli import encode_prompt, model_directory
from leaf.evaluation import write_record
from leaf.power import WINDOWS_POWER_MODE_NAMES, benchmark_power_mode_guard, keep_awake, require_benchmark_ac
from tools.benchmark_cli import CHILD_PROGRAM, token_list
from tools.decoder_plan import load_snapshot_plan
from tools.decoder_validation import latency_stability
from tools.verify_package_runtime import digest, isolated_environment, run_process

ROOT = Path(__file__).resolve().parents[1]
TIMERS = ('external_first_visible_text_ms', 'external_wall_ms')


def continuation(stdout: str, engine: str) -> str:
    newline = '\n' if engine == 'leaf' else '\n\n'
    normalized = stdout.replace('\r\n', '\n')
    if not normalized.endswith(newline) or '\ufffd' in normalized:
        raise ValueError(f'Invalid {engine} continuation framing or encoding')
    return normalized.removesuffix(newline)


def llama_work(stderr: str) -> tuple[int, int]:
    prompt = re.findall(r'prompt eval time\s*=\s*[0-9.]+ ms /\s*(\d+) tokens', stderr)
    decode = re.findall(r'(?m)^.*\bcommon_perf_print:\s+eval time\s*=\s*[0-9.]+ ms /\s*(\d+) runs', stderr)
    if len(prompt) != 1 or len(decode) != 1:
        raise ValueError('Missing or ambiguous llama.cpp prompt/decode timing counters')
    return int(prompt[0]), int(decode[0])


def summarize(samples: list[dict], pairs: int) -> dict:
    expected = [(pair, engine) for pair in range(1, pairs + 1)
                for engine in (('leaf', 'llama') if pair % 2 else ('llama', 'leaf'))]
    if [(sample['pair'], sample['engine']) for sample in samples] != expected:
        raise ValueError('Incomplete or out-of-order comparison pairs')
    result = {}
    for field in TIMERS:
        values = {engine: [sample[field] for sample in samples if sample['engine'] == engine]
                  for engine in ('leaf', 'llama')}
        spread = {engine: latency_stability(series) for engine, series in values.items()}
        if any(record['p10_ms'] is None for record in spread.values()):
            raise ValueError('Invalid external timing samples')
        result[field] = dict(p50_ms={engine: statistics.median(series) for engine, series in values.items()},
                             median_paired_leaf_over_llama=statistics.median(
                                 a / b for a, b in zip(values['leaf'], values['llama'])),
                             stable=all(row['passed'] for row in spread.values()), stability=spread)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'tokens', 'leaf-binary', 'leaf-artifact', 'leaf-provenance',
                 'llama-completion', 'llama-tokenize', 'gguf', 'gguf-audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--lengths', type=int, nargs='+', default=[63, 127])
    parser.add_argument('--generate', type=int, default=8)
    parser.add_argument('--pairs', type=int, default=6)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--required-power-mode', choices=WINDOWS_POWER_MODE_NAMES, required=True)
    args = parser.parse_args()
    if (sys.platform != 'win32' or args.pairs < 6 or args.pairs % 2 or args.generate < 2 or args.output.exists() or
            min(args.lengths) < 2 or len(set(args.lengths)) != len(args.lengths)):
        parser.error('Use Windows, unique prompt lengths >=2, even >=6 pairs, >=2 generated tokens and a fresh output')
    paths = {name: value.resolve() for name, value in vars(args).items() if isinstance(value, Path)}
    output, snapshot = paths.pop('output'), paths.pop('snapshot')
    completion, tokenize = paths['llama_completion'], paths['llama_tokenize']
    if completion.parent != tokenize.parent:
        parser.error('Use llama.cpp completion and tokenize from the same binary directory')
    plan = load_snapshot_plan(snapshot)
    row = json.loads(paths['tokens'].read_text(encoding='utf-8'))['sequences'][0]
    if not token_list(row) or max(args.lengths) > len(row) or max(args.lengths) + args.generate - 1 > plan['config']['max_positions']:
        parser.error('The token fixture and model context must cover every requested workload')
    provenance = json.loads(paths['leaf_provenance'].read_text(encoding='utf-8'))
    audit = json.loads(paths['gguf_audit'].read_text(encoding='utf-8'))
    checkpoint = {path.name: digest(path) for path in snapshot.glob('*.safetensors')}
    config_hash = digest(snapshot / 'config.json')
    request = provenance['request']
    if (audit.get('complete') is not True or audit.get('passed') is not True or
            audit.get('config_sha256') != config_hash or request.get('model_config_sha256') != config_hash or
            len(checkpoint) != 1 or audit.get('source_weight_sha256') not in checkpoint.values() or
            request.get('source_weight_sha256') != checkpoint or request.get('weight_bits') != 32 or
            provenance.get('artifact_sha256') != digest(paths['leaf_artifact']) or
            audit.get('gguf_sha256') != digest(paths['gguf'])):
        raise ValueError('Artifact audits must bind matching unquantized checkpoint values and configuration')
    output.mkdir(parents=True, exist_ok=False)
    cache, scratch = output / 'cache', output / 'tmp'
    scratch.mkdir()
    directory = model_directory(snapshot, cache)
    directory.mkdir(parents=True)
    staged = directory / 'decoder-32.leaf'
    try:
        os.link(paths['leaf_artifact'], staged)
    except OSError:
        shutil.copyfile(paths['leaf_artifact'], staged)
    if digest(staged) != provenance['artifact_sha256']:
        raise ValueError('Staged artifact differs from the audited input')
    environment = isolated_environment(cache)
    environment.update(LEAF_DECODER_BIN=str(paths['leaf_binary']), TEMP=str(scratch), TMP=str(scratch))
    llama_env = {key: value for key, value in environment.items() if not key.upper().startswith(('LLAMA_ARG_', 'GGML_'))}
    llama_env['PATH'] = str(completion.parent) + os.pathsep + llama_env['PATH']
    inputs = [*paths.values(), staged, *completion.parent.glob('*.dll'), *snapshot.glob('*.json'), *snapshot.glob('*.safetensors'),
              Path(__file__).resolve(), ROOT / 'leaf/cli.py', ROOT / 'tools/benchmark_cli.py', ROOT / 'tools/verify_package_runtime.py']
    hashes = {str(path): digest(path) for path in inputs}
    report = dict(complete=False, scope=__doc__.strip(), snapshot=str(snapshot), model=plan.get('source_architecture'),
                  hashes=hashes, cpu=args.cpu, threads=1, pairs=args.pairs, generated_tokens_requested=args.generate,
                  power={}, workloads={})
    report_path = output / 'comparison.json'
    tokenizer, _ = encode_prompt(snapshot, 'Leaf', raw=True)

    def execute(engine, prompt, length, pair):
        ac_before = require_benchmark_ac()
        if engine == 'leaf':
            metrics_path = output / f'leaf-{length}-{pair}.json'
            arguments = ['run', str(snapshot), '--offline', '--raw', '--prompt', prompt, '--max-tokens', str(args.generate),
                         '--bits', '32', '--fp32-math', 'accurate', '--threads', '1', '--worker-cpus', str(args.cpu),
                         '--metrics', str(metrics_path)]
            payload = json.dumps(dict(source_root=str(ROOT), arguments=arguments))
            sample = run_process([sys.executable, '-I', '-u', '-c', CHILD_PROGRAM, payload], environment, output)
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
            if (sample['preparation_message_seen'] or metrics['source_bf16_tensors'] or
                    metrics['fp32_math'] != 'accurate' or metrics['worker_cpus'] != [args.cpu] or
                    metrics['weight_bits'] != 32 or metrics['activation_bits'] != 32):
                raise ValueError('Leaf did not run the staged FP32 configuration')
            sample.update(metrics=metrics, generated_count=len(metrics['generated_tokens']))
        else:
            context = str(max(256, length + args.generate))
            command = [str(completion), '-m', str(paths['gguf']), '-p', prompt, '--no-escape', '-ngl', '0', '-n',
                       str(args.generate), '-t', '1', '-tb', '1', '-c', context, '-b', str(length), '-ub', str(length),
                       '-ctk', 'f32', '-ctv', 'f32', '-fa', 'off', '--temp', '0', '--repeat-penalty', '1',
                       '--no-conversation', '--no-display-prompt', '--no-warmup', '--no-context-shift', '--simple-io', '--color', 'off']
            sample = run_process(command, llama_env, output, require_package_check=False)
            prompt_count, decode = llama_work(sample['stderr'])
            if prompt_count != length:
                raise ValueError('llama.cpp reported a different prefill token count')
            sample.update(prompt_count=prompt_count, decode_forwards=decode, generated_count=decode + 1)
        ac_after = require_benchmark_ac()
        if any(type(sample[field]) not in (int, float) or not math.isfinite(sample[field]) or sample[field] <= 0 for field in TIMERS):
            raise ValueError('A command produced no visible continuation or invalid latency')
        return dict(sample, pair=pair, engine=engine, text=continuation(sample['stdout'], engine),
                    ac_before=ac_before, ac_after=ac_after)

    try:
        pin_cpu(args.cpu)
        with keep_awake(display=True), benchmark_power_mode_guard(report['power'], required_mode=args.required_power_mode):
            for length in args.lengths:
                ids = row[:length]
                prompt = tokenizer.decode(ids, skip_special_tokens=False)
                if encode_prompt(snapshot, prompt, raw=True)[1] != ids:
                    prompt = tokenizer.decode(ids, skip_special_tokens=True)
                if encode_prompt(snapshot, prompt, raw=True)[1] != ids:
                    raise ValueError(f'Token fixture cannot round-trip as raw text at {length} tokens')
                checked = subprocess.run([str(tokenize), '-m', str(paths['gguf']), '-p', prompt, '--no-escape', '-ngl', '0',
                                          '--ids', '--log-disable'], env=llama_env, cwd=output,
                                         capture_output=True, check=True, timeout=60)
                if json.loads(checked.stdout) != ids:
                    raise ValueError(f'Leaf and llama.cpp tokenizers disagree at {length} tokens')
                workload = report['workloads'][str(length)] = dict(prompt_ids=ids, preflight={}, samples=[])
                for engine in ('leaf', 'llama'):
                    workload['preflight'][engine] = execute(engine, prompt, length, 0)
                for pair in range(1, args.pairs + 1):
                    for engine in (('leaf', 'llama') if pair % 2 else ('llama', 'leaf')):
                        sample = execute(engine, prompt, length, pair)
                        workload['samples'].append(sample)
                        write_record(report_path, report)
                        print(length, pair, engine, round(sample['external_wall_ms'], 2), flush=True)
                expected = workload['preflight']['leaf']['text']
                observed = [*workload['preflight'].values(), *workload['samples']]
                workload['matched_workload'] = all(sample['text'] == expected and sample['generated_count'] == args.generate
                                                   for sample in observed)
                workload['summary'] = summarize(workload['samples'], args.pairs)
        if not all(digest(Path(name)) == checksum for name, checksum in hashes.items()):
            raise ValueError('A benchmark input changed during execution')
        for workload in report['workloads'].values():
            workload['timing_qualified'] = (report['power']['passed'] and workload['matched_workload'] and
                                            all(row['stable'] for row in workload['summary'].values()))
        report['complete'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        write_record(report_path, report)


if __name__ == '__main__':
    main()
