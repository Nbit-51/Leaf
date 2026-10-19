"""Warm-only per-projection diagnosis; generated builds never qualify a release."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from build import replace_once
from leaf.affinity import pin_cpu
from leaf.power import keep_awake, require_benchmark_ac
from tools.decoder_validation import run_native
from tools.validate_decoder import digest

GROUPS = ('q', 'k', 'v', 'o', 'up', 'gate', 'down', 'head')
REFERENCE = dict(before=ROOT / 'build/numerics-20261017/release/leaf_decoder.exe',
                 after=ROOT / 'build/linear-accumulation-20261018-k256/leaf_decoder.exe')


def instrument(source):
    enums = [prefix + '_' + group for prefix in ('Projection', 'Pack', 'Compute') for group in GROUPS]
    labels = [prefix + '_' + group for prefix in ('projection', 'pack', 'compute') for group in GROUPS]
    source = replace_once(source, 'Attention, Activation, KVAppend, CausalMask, Residual, Count',
                          'Attention, Activation, KVAppend, CausalMask, Residual, ' + ', '.join(enums) + ', Count')
    source = replace_once(source, 'enabled_ = value && value[0];', '''enabled_ = value && value[0];
        requested_ = enabled_;
        const auto skip = std::getenv("LEAF_PROFILE_WARMUP_FORWARDS");
        skip_forwards_ = skip ? std::stoull(skip) : 0;''')
    source = replace_once(source,
        'void forward_tokens(std::size_t tokens) noexcept { bucket_ = tokens == 1 ? 1 : 0; }',
        '''void forward_tokens(std::size_t tokens) noexcept {
        bucket_ = tokens == 1 ? 1 : 0;
        enabled_ = requested_ && forwards_++ >= skip_forwards_;
    }
    static unsigned projection_group(const std::string& name) {
        static const char* suffix[] = {"q_proj", "k_proj", "v_proj", "o_proj",
                                      "up_proj", "gate_proj", "down_proj", "lm_head"};
        for (unsigned i = 0; i < 8; ++i) {
            const std::string ending(suffix[i]);
            if (name.size() >= ending.size() && name.compare(name.size()-ending.size(), ending.size(), ending) == 0)
                return i;
        }
        throw std::runtime_error("Unclassified diagnostic projection: " + name);
    }
    static std::uint64_t thread_cycles() {
        ULONG64 cycles = 0;
        if (!QueryThreadCycleTime(GetCurrentThread(), &cycles)) throw std::runtime_error("Cannot query thread cycles");
        return cycles;
    }
    static std::uint64_t thread_cpu() {
        FILETIME created, exited, kernel, user;
        if (!GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user))
            throw std::runtime_error("Cannot query thread CPU time");
        return ((std::uint64_t(kernel.dwHighDateTime) << 32) | kernel.dwLowDateTime) +
               ((std::uint64_t(user.dwHighDateTime) << 32) | user.dwLowDateTime);
    }''')
    source = replace_once(source, 'if (profile_) start_ = Clock::now();', '''if (profile_) {
                start_ = Clock::now();
                cycles_ = thread_cycles();
                if (phase_ == Phase::Forward) cpu_ = thread_cpu();
            }''')
    source = replace_once(source, '++total.calls; total.tokens += tokens_;', '''++total.calls; total.tokens += tokens_;
                total.cycles += thread_cycles() - cycles_;
                if (phase_ == Phase::Forward) total.cpu_100ns += thread_cpu() - cpu_;''')
    source = replace_once(source, 'Clock::time_point start_{};',
                          'Clock::time_point start_{};\n        std::uint64_t cycles_ = 0, cpu_ = 0;')
    source = replace_once(source, '"attention", "activation", "kv_append", "causal_mask", "residual"',
                          '"attention", "activation", "kv_append", "causal_mask", "residual", ' +
                          ', '.join('"' + name + '"' for name in labels))
    source = replace_once(source, '"\\\"bucket_by\\\":\\\"forward_input_tokens\\\",\\\"timings\\\":{";',
        '"\\\"warm_only\\\":true,\\\"skipped_forwards\\\":" << skip_forwards_\n'
        '                   << ",\\\"bucket_by\\\":\\\"forward_input_tokens\\\",\\\"timings\\\":{";')
    source = replace_once(source, '<< ",\\\"milliseconds\\\":" << total.milliseconds << \'}\';',
        '<< ",\\\"milliseconds\\\":" << total.milliseconds\n'
        '                           << ",\\\"thread_cycles\\\":" << total.cycles\n'
        '                           << ",\\\"thread_cpu_ms\\\":" << double(total.cpu_100ns) / 10000 << \'}\';')
    source = replace_once(source, 'struct Total { std::uint64_t calls = 0, tokens = 0; double milliseconds = 0; };',
        'struct Total { std::uint64_t calls = 0, tokens = 0, cycles = 0, cpu_100ns = 0; double milliseconds = 0; };')
    source = replace_once(source, 'bool enabled_ = false;',
        'bool enabled_ = false, requested_ = false;\n    std::size_t skip_forwards_ = 0, forwards_ = 0;')
    source = replace_once(source, 'DecoderProfile::Scope scope(profile, phase, tokens);', '''DecoderProfile::Scope scope(profile, phase, tokens);
        const auto group = DecoderProfile::projection_group(name);
        DecoderProfile::Scope projection_scope(profile,
            static_cast<DecoderProfile::Phase>(static_cast<unsigned>(DecoderProfile::Phase::Projection_q) + group), tokens);''')
    pack = 'packed_float_input.pack(input, tokens, weight.cols, weight.cols, vector_pack);'
    source = replace_once(source, pack, '''{
                DecoderProfile::Scope pack_scope(profile,
                    static_cast<DecoderProfile::Phase>(static_cast<unsigned>(DecoderProfile::Phase::Pack_q) + group), tokens);
                ''' + pack + '\n            }')
    source = replace_once(source, '        workers.run(weight.rows, [&](std::size_t first, std::size_t last) {',
        '''        DecoderProfile::Scope compute_scope(profile,
            static_cast<DecoderProfile::Phase>(static_cast<unsigned>(DecoderProfile::Phase::Compute_q) + group), tokens);
        workers.run(weight.rows, [&](std::size_t first, std::size_t last) {''')
    return source


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    hashes = {str(Path(__file__)): digest(Path(__file__))}
    record = dict(complete=False, acceptance_timing=False, binaries={}, hashes=hashes)
    report = output / 'build.json'
    report.write_text(json.dumps(record, indent=2))
    try:
        for stage in ('before', 'after'):
            directory = output / stage
            directory.mkdir()
            original = ROOT / ('engine/src/decoder.cpp' if stage == 'before' else
                               'build/linear-accumulation-20261018-k256/decoder.cpp')
            source = directory / 'decoder.cpp'
            source.write_text(instrument(original.read_text()), newline='\n')
            executable = directory / 'leaf_decoder.exe'
            inputs = [original, source, ROOT / 'engine/src/main_decoder.cpp',
                      ROOT / 'engine/src/kv_cache.cpp', ROOT / 'engine/src/kernels/transformer.cpp',
                      ROOT / 'engine/include/leaf/kernels/token_panel.h',
                      ROOT / 'engine/include/leaf/kernels/row_reuse.h',
                      ROOT / 'engine/include/leaf/runtime/decoder_config.h', REFERENCE[stage]]
            command = ['C:/msys64/mingw64/bin/g++.exe', '-std=c++17', '-O3', '-DNDEBUG', '-pthread',
                       '-static', '-static-libgcc', '-static-libstdc++', '-I', str(ROOT / 'engine/include'),
                       '-I', str(HERE)]
            if stage == 'after':
                command.extend(('-DLEAF_ACCUMULATION_INLINE_PARTIAL=1', '-DLEAF_ACCUMULATION_TREE_OUTER=1',
                                '-DLEAF_ACCUMULATION_TREE_CHUNK=256'))
                inputs.extend((HERE / 'tree_sum.h', HERE / 'blocked_sum.h'))
            command.extend((str(source), str(ROOT / 'engine/src/main_decoder.cpp'),
                            str(ROOT / 'engine/src/kv_cache.cpp'), str(ROOT / 'engine/src/kernels/transformer.cpp'),
                            '-o', str(executable), '-lpsapi'))
            hashes.update({str(p): digest(p) for p in inputs})
            subprocess.run(command, check=True, cwd=ROOT)
            record['binaries'][stage] = dict(path=str(executable), sha256=digest(executable), command=command)
            report.write_text(json.dumps(record, indent=2))
        if any(digest(Path(p)) != h for p, h in hashes.items()):
            raise ValueError('Sources changed while building')
        record['complete'] = True
    finally:
        report.write_text(json.dumps(record, indent=2) + '\n')


def prepare(output):
    record = json.loads((output / 'build.json').read_text())
    if not record['complete']:
        raise ValueError('Require complete build')
    for info in record['binaries'].values():
        if digest(Path(info['path'])) != info['sha256']:
            raise ValueError('Changed diagnostic binary')
    scratch = output / 'tmp'
    scratch.mkdir(exist_ok=True)
    tempfile.tempdir = str(scratch.resolve())
    os.environ['LEAF_DECODER_PROFILE'] = '1'
    os.environ['LEAF_PROFILE_WARMUP_FORWARDS'] = '0'
    return record


def check(output):
    build_record = prepare(output)
    report = output / 'parity.json'
    if report.exists():
        raise ValueError('Refuse to overwrite parity results')
    paths = json.loads((ROOT / 'build/numerics-20261017/model-paths.json').read_text())
    record = dict(complete=False, passed=False, acceptance_timing=False, cases=[])
    try:
        for model in ('smol', 'qwen'):
            artifact = Path(paths[model]['artifact'])
            tokens = ROOT / 'build/family-models-20261017' / (model + '-quality-8/tokens.json')
            ids = json.loads(tokens.read_text())['sequences'][1][:17]
            for stage in ('before', 'after'):
                binary = Path(build_record['binaries'][stage]['path'])
                for mode in ('verify', 'generate'):
                    left, lm = run_native(REFERENCE[stage], artifact, [ids], mode=mode, generate=4)
                    right, rm = run_native(binary, artifact, [ids], mode=mode, generate=4, capture_profile=True)
                    exact = np.array_equal(left.view(np.uint32), right.view(np.uint32))
                    exact = exact and lm['generated_tokens'] == rm['generated_tokens']
                    record['cases'].append(dict(model=model, stage=stage, mode=mode, exact=exact,
                        reference_sha256=digest(REFERENCE[stage]), diagnostic_sha256=digest(binary),
                        artifact_sha256=digest(artifact), tokens_sha256=digest(tokens)))
                    report.write_text(json.dumps(record, indent=2))
                    print(model, stage, mode, 'exact', exact, flush=True)
                    if not exact:
                        raise ValueError('Instrumentation changed output')
        record.update(complete=True, passed=True)
    finally:
        report.write_text(json.dumps(record, indent=2) + '\n')


def measure(output):
    builds = prepare(output)
    if not json.loads((output / 'parity.json').read_text()).get('passed'):
        raise ValueError('Require exact instrumentation parity first')
    report = output / 'projection-costs.json'
    if report.exists():
        raise ValueError('Refuse to overwrite measurements')
    pin_cpu(2)
    os.environ['LEAF_PROFILE_WARMUP_FORWARDS'] = '6'
    record = dict(complete=False, acceptance_timing=False, runs=7, warmup=3, cpu=2,
                  scope='Two balanced diagnostic pairs, warm-only scopes; nested times must not be summed twice',
                  hashes={str(output / name): digest(output / name) for name in ('build.json', 'parity.json')}, models={})
    try:
        with keep_awake():
            for model, layers in (('smol', 32), ('qwen', 24)):
                work = ROOT / 'build/numerics-20261017/decode-silu' / (model + '-work')
                artifact, tokens = work / 'decoder-32.leaf', work / 'tokens.json'
                record['hashes'].update({str(p): digest(p) for p in (artifact, tokens)})
                ids = json.loads(tokens.read_text())['benchmark_ids']
                passes = record['models'][model] = []
                for stage in ('before', 'after', 'after', 'before'):
                    power = require_benchmark_ac()
                    _, metrics = run_native(Path(builds['binaries'][stage]['path']), artifact, [ids], mode='bench',
                                            runs=7, warmup=3, threads=1, activation_bits=32,
                                            capture_profile=True, windows_above_normal=True)
                    profile = metrics['diagnostic_profile']
                    if not profile['warm_only'] or profile['skipped_forwards'] != 6:
                        raise ValueError('Warmup exclusion missing')
                    for phase in ('prefill', 'decode'):
                        rows = profile['timings'][phase]
                        if rows['forward']['calls'] != 7:
                            raise ValueError('Warmup or sample count mismatch')
                        for group in GROUPS:
                            expected = 7 * (1 if group == 'head' else layers)
                            if rows['projection_' + group]['calls'] != expected:
                                raise ValueError('Projection count mismatch')
                    passes.append(dict(stage=stage, raw=metrics, power_before=power, power_after=require_benchmark_ac()))
                    report.write_text(json.dumps(record, indent=2))
                    print(model, stage, metrics['prefill_p50_ms'], metrics['decode_p50_ms'], flush=True)
        if any(digest(Path(p)) != h for p, h in record['hashes'].items()):
            raise ValueError('Measurement inputs changed')
        record['complete'] = True
    finally:
        report.write_text(json.dumps(record, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('build', 'check', 'measure'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.name != 'nt':
        raise ValueError('These cycle/CPU counters require Windows')
    globals()[args.stage](args.output.resolve())


if __name__ == '__main__':
    main()
