"""Serial Leaf/Ollama FP32 generation comparison with checked cache/output work.

Scope: backend-reported uncached prefill and generation of two output tokens
(one cached forward after the prefill token). HTTP/load time is separate. Six
balanced pairs, 3 warmups/11 measurements by default. No sample filtering.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
import time

import psutil
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.evaluation import PROMPTS, write_record
from leaf.power import keep_awake, require_benchmark_ac
from tools.decoder_validation import latency_stability
from tools.ollama_validation import api, generation_request, local_server
from tools.validate_decoder import digest, utc_now


def continuation(tokenizer, ids, generated):
    prefix = tokenizer.decode(ids, skip_special_tokens=True)
    full = tokenizer.decode(ids + generated, skip_special_tokens=True)
    if not full.startswith(prefix):
        raise ValueError('Continuation cannot be decoded independently of its prefix')
    return full[len(prefix):]


def prefix_workloads(tokenizer):
    body = (' A careful engineer measured the program on a local computer. '
            'The experiment used the same model, inputs and processor settings. ') * 10
    rows = []
    for lead in ('A', 'B'):
        ids = tokenizer.encode(lead + body).ids[:64]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        if len(ids) != 64 or tokenizer.encode(text).ids != ids:
            raise ValueError('Benchmark prompt does not roundtrip to the frozen 64 token IDs')
        rows.append(dict(prompt=text, ids=ids))
    common = 0
    for left, right in zip(rows[0]['ids'], rows[1]['ids']):
        if left != right:
            break
        common += 1
    if common > 1:
        raise ValueError('Benchmark prompts must differ immediately after their shared BOS, if any')
    return rows, common


def leaf_pass(args, rows, directory, *, maximum=2, quality=False, cached=1):
    directory.mkdir(parents=True, exist_ok=False)
    request, output = directory / 'request.bin', directory / 'results.jsonl'
    with request.open('wb') as stream:
        stream.write(struct.pack('<I', len(rows)))
        for row in rows:
            stream.write(struct.pack('<I', len(row['ids'])))
            stream.write(struct.pack('<' + 'I' * len(row['ids']), *row['ids']))
    parent = psutil.Process()
    previous = parent.cpu_affinity()
    environment = dict(os.environ)
    if not quality:
        require_benchmark_ac()
        environment['LEAF_BENCHMARK_REQUIRE_AC'] = '1'
    with (directory / 'stderr.log').open('wb') as error:
        try:
            parent.cpu_affinity([args.cpu])
            process = subprocess.Popen([str(args.leaf_executable.resolve()), str(args.leaf_artifact.resolve()),
                str(request.resolve()), str(output.resolve()), str(maximum), str(int(quality)), str(cached)],
                stdout=subprocess.DEVNULL, stderr=error, env=environment,
                creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW)
        finally:
            parent.cpu_affinity(previous)
        if process.wait():
            raise RuntimeError(f'Leaf pass failed; inspect {directory / "stderr.log"}')
    if not quality:
        require_benchmark_ac()
    results = [json.loads(line) for line in output.read_text().splitlines()]
    if len(results) != len(rows):
        raise ValueError('Incomplete Leaf pass')
    if not quality and any(row.get('ac_power_checked') is not True for row in results):
        raise ValueError('Native benchmark did not verify AC power at sample boundaries; rebuild the adapter')
    return results


def unload(base, model, parent):
    api(base, '/api/generate', dict(model=model, keep_alive=0))
    for _ in range(100):
        workers = [p for p in psutil.Process(parent.pid).children(recursive=True)
                   if 'llama-server' in p.name().lower()]
        if not api(base, '/api/ps').get('models') and not workers:
            return
        time.sleep(0.1)
    raise TimeoutError('Ollama model did not unload before Leaf execution')


def backend_port(process, cpu, *, configure=False):
    workers = [p for p in psutil.Process(process.pid).children(recursive=True)
               if 'llama-server' in p.name().lower()]
    if len(workers) != 1:
        raise ValueError('Expected one local CPU backend')
    worker = workers[0]
    if configure:
        worker.cpu_affinity([cpu])
        worker.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    if worker.cpu_affinity() != [cpu] or worker.nice() != psutil.ABOVE_NORMAL_PRIORITY_CLASS:
        raise ValueError('Ollama backend affinity/priority differs from Leaf')
    command = worker.cmdline()
    port = command[command.index('--port') + 1]
    return f'http://127.0.0.1:{port}', dict(pid=worker.pid, command=command,
        affinity=worker.cpu_affinity(), priority='above_normal')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ollama-executable', 'ollama-models', 'leaf-executable', 'leaf-artifact',
                 'snapshot', 'gguf-audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--ollama-model', required=True)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--pairs', type=int, default=6)
    parser.add_argument('--runs', type=int, default=11)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--diagnostic-timing', action='store_true')
    parser.add_argument('--preflight-record', type=Path,
                        help='reuse completed quality checks bound to identical runtimes, model and request policy')
    args = parser.parse_args()
    if os.name != 'nt':
        parser.error('This controlled Ollama comparison currently requires Windows')
    require_benchmark_ac()
    if args.output.exists() or args.pairs < 6 or args.pairs % 2 or args.runs < 11 or args.warmup < 3:
        parser.error('Require fresh output, even >=6 pairs, >=11 measurements and >=3 warmups')
    if any(value for name, value in os.environ.items()
           if name.startswith('LEAF_EXPERIMENTAL_') or name == 'LEAF_DECODER_PROFILE'):
        parser.error('Clear inherited experiment/profile environment overrides')
    adapter = json.loads(subprocess.check_output([str(args.leaf_executable.resolve()),'--runtime-info'],text=True))
    if adapter.get('benchmark_power_guard') is not True:
        parser.error('Rebuild the native generation adapter with the per-sample power guard')
    audit = json.loads(args.gguf_audit.read_text())
    if audit.get('passed') is not True or audit.get('source_weight_sha256') != digest(args.snapshot / 'model.safetensors'):
        parser.error('Require exact source-to-F32-GGUF parameter audit')
    provenance = json.loads(args.leaf_artifact.with_suffix('.provenance.json').read_text())
    source = provenance['request']['source_weight_sha256']
    if source != {'model.safetensors': audit['source_weight_sha256']} or provenance['request']['weight_bits'] != 32:
        parser.error('Leaf and Ollama do not bind the same FP32 source checkpoint')
    if provenance['artifact_sha256'] != digest(args.leaf_artifact):
        parser.error('Leaf artifact hash mismatch')
    if (audit['config_sha256'] != digest(args.snapshot/'config.json') or
            provenance['request']['model_config_sha256'] != audit['config_sha256']):
        parser.error('Leaf and Ollama configuration provenance differs')
    args.output.mkdir(parents=True)
    tokenizer = Tokenizer.from_file(str(args.snapshot / 'tokenizer.json'))
    workloads, cached = prefix_workloads(tokenizer)
    record = dict(format='leaf-ollama-cpu-generation-v1', measured_at_utc=utc_now(),
        scope=__doc__, cpu=args.cpu, threads=1, weight_bits=32, kv_bits=32,
        cached_prompt_tokens=cached, generated_tokens_per_sample=2, cached_decode_forwards=1,
        runs_per_pass=args.runs, warmup_per_pass=args.warmup,
        workload=workloads, complete=False, overall_speed_claim=False,
        criteria='Quality passed; each pass and pooled p90/p10 <=1.25; Leaf/Ollama median paired prefill <=0.98 and decode <=1.02',
        pairs=[], hashes={str(p): digest(p) for p in (args.ollama_executable, args.leaf_executable,
            args.leaf_artifact, args.gguf_audit, Path(__file__), ROOT/'tools/ollama_validation.py',
            ROOT/'benchmark/decoder_generation_benchmark.cpp', args.snapshot/'tokenizer.json')})
    path = args.output / 'comparison.json'
    write_record(path, record)
    with keep_awake(), local_server(args.ollama_executable, args.ollama_models,
                                   args.output / 'server', cpu=args.cpu) as (base, _, process):
        record['ollama_version'] = api(base, '/api/version')
        shown = api(base, '/api/show', dict(model=args.ollama_model))
        if shown['details']['quantization_level'] != 'F32':
            raise ValueError('Ollama model is not F32')
        blob = re.search(r'(?m)^FROM .*sha256-([0-9a-f]{64})$', shown['modelfile'])
        if blob is None or blob.group(1) != audit['gguf_sha256']:
            raise ValueError('Imported Ollama blob does not match the audited GGUF')
        record['ollama_model_info'] = shown
        api(base, '/api/generate', generation_request(args.ollama_model, 'Z', tokens=1))
        backend, policy = backend_port(process, args.cpu, configure=True)
        record['backend_policy'] = policy
        for row in workloads:
            actual = api(backend, '/tokenize', dict(content=row['prompt'], add_special=True))['tokens']
            if actual != row['ids']:
                raise ValueError('Actual Ollama backend tokenization differs from Leaf')
        if args.preflight_record:
            saved = json.loads(args.preflight_record.read_text())
            required = (args.ollama_executable, args.leaf_executable, args.leaf_artifact,
                        args.gguf_audit, args.snapshot/'tokenizer.json',
                        ROOT/'tools/ollama_validation.py', ROOT/'benchmark/decoder_generation_benchmark.cpp')
            old = {Path(p).resolve(): h for p,h in saved.get('hashes', {}).items()}
            if any(old.get(p.resolve()) != digest(p) for p in required):
                raise ValueError('Saved preflight does not match current execution inputs')
            if (saved['workload'] != workloads or saved['cached_prompt_tokens'] != cached
                    or saved['ollama_version'] != record['ollama_version']):
                raise ValueError('Saved preflight workload or Ollama version differs')
            quality = saved['quality']
            if len(quality.get('checks', [])) != 8:
                raise ValueError('Saved preflight is incomplete')
            record['reused_preflight'] = dict(path=str(args.preflight_record), sha256=digest(args.preflight_record))
            unload(base, args.ollama_model, process)
        else:
            quality_rows = [dict(prompt=text, ids=tokenizer.encode(text).ids) for text in PROMPTS]
            quality_rows += workloads
            ollama_quality = [api(base, '/api/generate', generation_request(args.ollama_model,
                row['prompt'], tokens=8, logprobs=True)) for row in quality_rows]
            (args.output / 'ollama-quality.json').write_text(json.dumps(ollama_quality, indent=2)+'\n')
            unload(base, args.ollama_model, process)
            leaf_quality = leaf_pass(args, quality_rows, args.output / 'leaf-quality', maximum=8,
                                     quality=True, cached=cached)
            checks = []
            for row, leaf, ollama in zip(quality_rows, leaf_quality, ollama_quality):
                text = continuation(tokenizer, row['ids'], leaf['generated_tokens'])
                logs = [v['logprob'] for v in ollama.get('logprobs', [])]
                expected = leaf['selected_logprobs'][:len(logs)]
                error = max((abs(a-b) for a,b in zip(logs, expected)), default=None)
                passed = (text == ollama['response'] and len(logs)>0 and len(logs)==len(expected)
                          and error is not None and error<=2e-4)
                checks.append(dict(prompt=row['prompt'], leaf_text=text, ollama_text=ollama['response'],
                    selected_logprob_max_abs_error=error, logprob_abs_limit=2e-4, passed=passed))
            quality = dict(scope='Eight greedy continuations and selected-token log probabilities; not full-logit/perplexity qualification',
                           checks=checks, passed=all(c['passed'] for c in checks))
        record['quality'] = quality
        write_record(path, record)
        print('Quality', quality['passed'], flush=True)
        if not quality['passed'] and not args.diagnostic_timing:
            raise ValueError('Numerical preflight failed; explicit --diagnostic-timing required for unqualified measurements')
        for index in range(args.pairs):
            pair = dict(order=['leaf', 'ollama'] if index%2==0 else ['ollama', 'leaf'])
            record['pairs'].append(pair)
            rows = [workloads[i%2] for i in range(args.warmup+args.runs)]
            for engine in pair['order']:
                if engine == 'leaf':
                    values = leaf_pass(args, rows, args.output / f'{index+1}-leaf', cached=cached)
                    for row, value in zip(rows, values):
                        value['response'] = continuation(tokenizer, row['ids'], value['generated_tokens'])
                        if value['cached_decode_forwards'] != 1:
                            raise ValueError('Leaf stopped before one cached decode step')
                else:
                    values = []
                    with (args.output / f'{index+1}-ollama.jsonl').open('x') as raw:
                        for sample, row in enumerate(rows):
                            power_before = require_benchmark_ac()
                            start = time.perf_counter()
                            value = api(base, '/api/generate', generation_request(args.ollama_model, row['prompt']))
                            value['external_request_ms'] = (time.perf_counter()-start)*1000
                            if sample == 0:
                                backend_port(process, args.cpu, configure=True)
                            value['prefill_ms'] = value['prompt_eval_duration']/1e6
                            value['generation_phase_ms'] = value['eval_duration']/1e6
                            values.append(value)
                            raw.write(json.dumps(value,allow_nan=False)+'\n')
                            raw.flush()
                            value['power_at_request_boundaries'] = dict(before=power_before, after=require_benchmark_ac())
                            if (value['prompt_eval_count'] != 64 or value.get('prompt_eval_cached_count') != cached
                                    or value['eval_count'] != 2):
                                if sample >= args.warmup:
                                    raise ValueError(f'Unmatched Ollama prompt/decode work: {value}')
                    backend_port(process, args.cpu)
                    ps = api(base, '/api/ps')
                    if len(ps['models']) != 1 or ps['models'][0].get('size_vram') != 0:
                        raise ValueError('Ollama CPU-only execution not confirmed')
                    record['ollama_process_state'] = ps
                    unload(base, args.ollama_model, process)
                measured = values[args.warmup:]
                pair[engine] = dict(samples=measured, warmups=values[:args.warmup],
                    stability={phase: latency_stability([v[key] for v in measured])
                               for phase,key in (('prefill','prefill_ms'),('decode','generation_phase_ms'))})
                write_record(path, record)
                print(index+1, engine, statistics.median(v['prefill_ms'] for v in measured),
                      statistics.median(v['generation_phase_ms'] for v in measured), flush=True)
        summary = {}
        for phase, key, limit in (('prefill','prefill_ms',0.98),('decode','generation_phase_ms',1.02)):
            pooled = {s: [v[key] for p in record['pairs'] for v in p[s]['samples']] for s in ('leaf','ollama')}
            medians = {s: statistics.median(v) for s,v in pooled.items()}
            ratio = medians['leaf']/medians['ollama']
            paired = statistics.median(statistics.median(v[key] for v in p['leaf']['samples'])/
                statistics.median(v[key] for v in p['ollama']['samples']) for p in record['pairs'])
            stable = {s: latency_stability(v) for s,v in pooled.items()}
            every_pass = all(p[s]['stability'][phase]['passed'] for p in record['pairs'] for s in pooled)
            summary[phase] = dict(medians_ms=medians, pooled_ratio=ratio, median_paired_ratio=paired,
                pooled_stability=stable, every_pass_stable=every_pass,
                passed=every_pass and all(v['passed'] for v in stable.values()) and paired<=limit)
        text_matches = all(all(p['leaf']['samples'][i]['response']==p['ollama']['samples'][i]['response']
                           for i in range(args.runs)) for p in record['pairs'])
        record.update(complete=True, gate=summary, timed_generated_text_matches=text_matches,
            overall_speed_claim=quality['passed'] and text_matches and all(p['passed'] for p in summary.values()))
        if any(digest(Path(p))!=h for p,h in record['hashes'].items()):
            raise ValueError('Inputs changed during comparison')
        write_record(path, record)
        print(json.dumps(summary, indent=2), flush=True)
        print('overall_speed_claim', record['overall_speed_claim'], flush=True)


if __name__ == '__main__':
    main()
