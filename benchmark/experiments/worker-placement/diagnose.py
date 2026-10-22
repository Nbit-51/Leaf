"""Unchanged-binary A/A diagnostic with native placement and process counters.

This is not acceptance timing. Instrumentation is isolated in build/, verified
against full reference logits, and never compiled into the installed runtime.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'mlp-schedule'
sys.path.insert(0, str(ROOT))
from leaf.power import keep_awake
from tools.decoder_validation import run_native, latency_stability
from tools.diagnose_windows_benchmark import cpu_sets
from tools.validate_decoder import digest, utc_now, write_record

spec = importlib.util.spec_from_file_location('prior_process_diagnostic', PRIOR / 'diagnose.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
once = shared.replace_once
CORES = [2, 4, 6, 8]


def build(source, directory, queue_wait=False):
    directory.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / 'build.json').read_text())
    if not manifest['complete'] or any(digest(Path(p)) != h for p, h in manifest['source_sha256'].items()):
        raise ValueError('Source manifest changed')
    reference = source / 'leaf_decoder.exe'
    if digest(reference) != manifest['binary_sha256']['leaf_decoder.exe']:
        raise ValueError('Source binary changed')
    main = source / 'main_decoder.cpp'
    text = '#include "process_probe.h"\n#include <filesystem>\n#include <thread>\n' + main.read_text()
    text = once(text, '        const double load_ms = elapsed(load_start);', '''        const double load_ms = elapsed(load_start);
        if (const char* gate = std::getenv("LEAF_PROCESS_PROBE_GATE")) {
            const auto deadline = Clock::now() + std::chrono::seconds(30);
            while (!std::filesystem::exists(gate)) {
                if (Clock::now() >= deadline) throw std::runtime_error("Diagnostic gate timeout");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }''')
    text = once(text, '        std::vector<double> prefills, decodes;',
                '        std::vector<double> prefills, decodes;\n        leaf::experiments::ProcessSamples process_prefill, process_decode;')
    text = once(text, '''                    decoder.reset(); auto start = Clock::now(); decoder.forward(prefix); const double prefill = elapsed(start);
                    start = Clock::now(); decoder.forward(last); const double decode = elapsed(start);''', '''                    decoder.reset();
                    const auto before = leaf::experiments::process_stamp();
                    auto start = Clock::now(); decoder.forward(prefix); const double prefill = elapsed(start);
                    const auto middle = leaf::experiments::process_stamp();
                    start = Clock::now(); decoder.forward(last); const double decode = elapsed(start);
                    const auto after = leaf::experiments::process_stamp();
                    if (iteration >= warmup) {
                        process_prefill.add(before, middle); process_decode.add(middle, after);
                    }''')
    text = once(text, '        metrics << ",\\\"generated_tokens\\\":"; array(metrics, generated);',
        '        metrics << ",\\\"generated_tokens\\\":"; array(metrics, generated);\n'
        '        metrics << ",\\\"process_diagnostic_build\\\":true";\n'
        '        process_prefill.emit(metrics, "prefill"); process_decode.emit(metrics, "decode");')
    generated = directory / 'main_decoder.cpp'
    if queue_wait:
        text = '#include "decoder_diagnostic.h"\n' + text
        text = once(text, '        const auto load_start = Clock::now();',
            '        leaf::diagnostics::ThreadDispatch dispatch_counters;\n        const auto load_start = Clock::now();')
    generated.write_text(text, newline='\n')
    command = list(manifest['commands'][0])
    command[command.index(str(main))] = str(generated)
    command[command.index('-o') + 1] = str(directory / 'leaf_decoder.exe')
    command[1:1] = ['-I', str(PRIOR)]
    inputs = dict(manifest['source_sha256'])
    inputs.update({str(p): digest(p) for p in (Path(__file__), PRIOR / 'diagnose.py', PRIOR / 'process_probe.h', source / 'build.json', reference, generated)})
    if queue_wait:
        probe = directory / 'queue_probe.h'
        probe.write_text('#pragma once\n#include <chrono>\n#include <cstdint>\n'
            'namespace leaf::experiments { inline double pool_wait_ms = 0; inline std::uint64_t pool_blocked = 0, pool_calls = 0; }\n', newline='\n')
        host = next(Path(p) for p in manifest['source_sha256'] if Path(p).as_posix().endswith('/engine/include/leaf/runtime/host_support.h'))
        private_host = directory / 'include/leaf/runtime/host_support.h'
        private_host.parent.mkdir(parents=True)
        host_text = '#include "queue_probe.h"\n' + host.read_text()
        # Start before an optional caller spin, so both policies measure the
        # whole post-drain barrier interval rather than just the blocking tail.
        host_text = once(host_text, '        ready_.notify_all(); drain();',
            '        ready_.notify_all(); drain();\n'
            '        const auto wait_start = std::chrono::steady_clock::now();\n'
            '        ++leaf::experiments::pool_calls;\n'
        )
        host_text = once(host_text, '        std::unique_lock<std::mutex> lock(mutex_);\n        finished_.wait',
            '        std::unique_lock<std::mutex> lock(mutex_);\n'
            '        if (pending_) ++leaf::experiments::pool_blocked;\n        finished_.wait')
        host_text = once(host_text, '        const auto error = task_error_;',
            '        leaf::experiments::pool_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wait_start).count();\n'
            '        const auto error = task_error_;')
        private_host.write_text(host_text, newline='\n')
        process = directory / 'process_probe.h'
        process_text = '#include "queue_probe.h"\n#include "decoder_diagnostic.h"\n' + (PRIOR / 'process_probe.h').read_text()
        process_text = once(process_text, '    double unix_seconds;',
            '    double unix_seconds, pool_wait_ms;\n    std::uint64_t pool_blocked, pool_calls, main_switches, main_cycles, main_cpu_100ns;')
        process_text = once(process_text, '    return {file_ticks(kernel)',
            '    const auto main = leaf::diagnostics::stamp();\n    return {file_ticks(kernel)')
        process_text = once(process_text, 'double(file_ticks(now) - 116444736000000000ull) / 1e7};',
            'double(file_ticks(now) - 116444736000000000ull) / 1e7, pool_wait_ms, pool_blocked, pool_calls, main.context_switches, main.cycles, main.cpu_100ns};')
        process_text = once(process_text, '    void add(const ProcessStamp& a,',
            '    std::vector<double> wait_ms, blocked, calls, main_switches, main_cycles, main_cpu_ms;\n    void add(const ProcessStamp& a,')
        process_text = once(process_text, '        starts_unix_seconds.push_back(a.unix_seconds);',
            '        wait_ms.push_back(b.pool_wait_ms - a.pool_wait_ms);\n'
            '        blocked.push_back(double(b.pool_blocked - a.pool_blocked)); calls.push_back(double(b.pool_calls - a.pool_calls));\n'
            '        main_switches.push_back(double(b.main_switches - a.main_switches));\n'
            '        main_cycles.push_back(double(b.main_cycles - a.main_cycles));\n'
            '        main_cpu_ms.push_back(double(b.main_cpu_100ns - a.main_cpu_100ns) / 1e4);\n'
            '        starts_unix_seconds.push_back(a.unix_seconds);')
        process_text = once(process_text, '        field("cpu_ms", cpu_ms);',
            '        field("pool_wait_ms", wait_ms); field("pool_blocked", blocked); field("pool_calls", calls);\n'
            '        field("main_switches", main_switches); field("main_cycles", main_cycles); field("main_cpu_ms", main_cpu_ms);\n'
            '        field("cpu_ms", cpu_ms);')
        process.write_text(process_text, newline='\n')
        diagnostic_header = ROOT / 'tests/native/decoder_diagnostic.h'
        command[1:1] = ['-I', str(directory / 'include'), '-I', str(directory), '-I', str(diagnostic_header.parent)]
        inputs.update({str(p): digest(p) for p in (probe, private_host, process, diagnostic_header)})
    report = dict(complete=False, acceptance_timing=False, queue_wait=queue_wait, inputs=inputs, command=command, candidate_binary=str(reference))
    write_record(directory / 'build.json', report)
    try:
        subprocess.run(command, check=True)
        if any(digest(Path(p)) != h for p, h in inputs.items()):
            raise ValueError('Diagnostic build inputs changed')
        report.update(complete=True, binary_sha256=digest(directory / 'leaf_decoder.exe'))
    finally:
        write_record(directory / 'build.json', report)


def run(directory, model):
    output = directory / (model + '-diagnostic.json')
    if output.exists():
        raise ValueError('Retain previous records')
    build_record = json.loads((directory / 'build.json').read_text())
    binary, reference = directory / 'leaf_decoder.exe', Path(build_record['candidate_binary'])
    if not build_record['complete'] or digest(binary) != build_record['binary_sha256']:
        raise ValueError('Diagnostic build incomplete')
    if any(digest(Path(p)) != h for p, h in build_record['inputs'].items()):
        raise ValueError('Diagnostic sources changed')
    prepared = ROOT / 'build/source-bf16-20261020/prepared.json'
    row = json.loads(prepared.read_text())['models'][model]
    artifact = Path(row['after'])
    if digest(artifact) != row['after_sha256']:
        raise ValueError('Model artifact changed')
    tokens = ROOT / ('build/numerics-20261017/decode-silu/' + model + '-work/tokens.json')
    ids = json.loads(tokens.read_text())['benchmark_ids']
    scratch = directory / 'tmp'
    scratch.mkdir(exist_ok=True)
    os.environ['TEMP'] = os.environ['TMP'] = str(scratch)
    import tempfile
    tempfile.tempdir = str(scratch)
    paths = [Path(__file__), directory / 'build.json', binary, reference, prepared, artifact, tokens,
             PRIOR / 'diagnose.py', ROOT / 'tools/decoder_validation.py', ROOT / 'tools/diagnose_windows_benchmark.py',
             ROOT / 'leaf/power.py', HERE.parent / 'source-bf16/integration.py']
    record = dict(complete=False, acceptance_timing=False, utc=utc_now(),
        scope=__doc__, cores=CORES, monitor_cpu=27, hashes={str(p): digest(p) for p in paths}, passes=[])
    write_record(output, record)
    me = psutil.Process()
    affinity = me.cpu_affinity()
    counters = shared.CoreCounters(CORES)
    record['counter_unavailable'] = counters.unavailable
    spec = importlib.util.spec_from_file_location('placement_diag_guards', HERE.parent / 'source-bf16/integration.py')
    guards = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guards)
    try:
        with keep_awake(display=True):
            a, _ = run_native(reference, artifact, [ids[:33]], threads=4, worker_cpus=CORES)
            b, _ = run_native(binary, artifact, [ids[:33]], threads=4, worker_cpus=CORES)
            record['full_logit_bit_exact'] = bool(np.array_equal(a.view('<u4'), b.view('<u4')))
            if not record['full_logit_bit_exact']:
                raise ValueError('Instrumentation changed logits')
            os.environ['LEAF_BENCHMARK_REQUIRE_AC'] = '1'
            for index in range(4):
                telemetry = []
                me.cpu_affinity([27])
                counters.read()
                with tempfile.TemporaryDirectory(dir=scratch) as tmp:
                    work = Path(tmp)
                    request = work / 'tokens.bin'
                    request.write_bytes(struct.pack('<II', 1, len(ids)) + struct.pack(f'<{len(ids)}I', *ids))
                    gate = work / 'start.ready'
                    command = [str(binary), str(artifact), str(request), str(work / 'logits.bin'), str(work / 'metrics.json'),
                               'bench', '4', '31', '10', '16', '0', '1', '32', ','.join(map(str, CORES))]
                    me.cpu_affinity(CORES)
                    try:
                        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=dict(os.environ, LEAF_PROCESS_PROBE_GATE=str(gate)), creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS)
                    finally:
                        me.cpu_affinity([27])
                    try:
                        owned = psutil.Process(child.pid)
                        if owned.cpu_affinity() != CORES:
                            raise ValueError('Unexpected process affinity')
                        with guards.power_window() as power:
                            gate.touch()
                            while child.poll() is None:
                                try:
                                    child.wait(timeout=0.5)
                                except subprocess.TimeoutExpired:
                                    pass
                                topology = cpu_sets()
                                telemetry.append(dict(unix_seconds=time.time(), counters=counters.read(),
                                    parked={str(c['logical_cpu']): bool(c['flags'] & 1) for c in topology
                                            if c['group'] == 0 and c['logical_cpu'] in CORES}))
                            _, stderr = child.communicate()
                        if child.returncode:
                            raise RuntimeError(stderr)
                        metrics = json.loads((work / 'metrics.json').read_text())
                    finally:
                        if child.poll() is None:
                            child.kill()
                            child.communicate()
                if metrics.get('worker_cpus') != CORES or not metrics.get('process_diagnostic_build') or not metrics['benchmark_ac_checked']:
                    raise ValueError('Diagnostic policy or native AC guard missing')
                for phase in ('prefill', 'decode'):
                    for field in ('cpu_ms', 'cycles', 'faults', 'starts_unix_seconds', 'ends_unix_seconds'):
                        values = metrics['process_' + field + '_' + phase]
                        if len(values) != 31 or not all(np.isfinite(v) and v >= 0 for v in values):
                            raise ValueError('Invalid process samples')
                record['passes'].append(dict(index=index, metrics=metrics, telemetry=telemetry, power=power,
                    stability={p: latency_stability(metrics[p + '_samples_ms']) for p in ('prefill', 'decode')}))
                write_record(output, record)
                print(model, index + 1, {p: dict(wall_ms=metrics[p + '_p50_ms'], cpu_ms=statistics.median(metrics['process_cpu_ms_' + p]),
                    cycles=statistics.median(metrics['process_cycles_' + p]), faults=sum(metrics['process_faults_' + p])) for p in ('prefill', 'decode')}, flush=True)
            record['sleep_events'] = guards.sleep_events(record['utc'])
        if any(digest(Path(p)) != h for p, h in record['hashes'].items()):
            raise ValueError('Diagnostic inputs changed')
        record['complete'] = True
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        counters.close()
        me.cpu_affinity(affinity)
        write_record(output, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('build', 'run'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--model', choices=('smol', 'qwen'), default='smol')
    parser.add_argument('--queue-wait', action='store_true', help='build additional caller barrier/dispatch diagnostics')
    args = parser.parse_args()
    if args.stage == 'build':
        if args.source is None:
            parser.error('--source is required')
        build(args.source.resolve(), args.directory.resolve(), args.queue_wait)
    else:
        run(args.directory.resolve(), args.model)


if __name__ == '__main__':
    main()
