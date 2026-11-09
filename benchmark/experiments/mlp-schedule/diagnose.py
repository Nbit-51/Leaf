"""Four-worker A/A diagnostic: process work, clocks and parking, never a speed gate.

The owned decoder inherits its four-core mask at creation. The Python monitor
runs on a separate physical core; samples include load/warmup. No power settings
are changed, and no unrelated process inventory is collected.

Optional off/on/on/off placement pins the four owned threads individually after
construction, before warmup. It tests the parked-core hypothesis, not a kernel
release. It never changes affinity of unrelated processes.
"""
import argparse
import ctypes
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
sys.path.insert(0, str(ROOT))
from leaf.power import keep_awake
from tools.decoder_validation import run_native, latency_stability
from tools.diagnose_windows_benchmark import Counters, CounterValue, cpu_sets
from tools.validate_decoder import digest, utc_now, write_record


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Diagnostic anchor changed; inspect generation')
    return text.replace(before, after)


def build(source, directory):
    manifest = json.loads((source/'build.json').read_text())
    assert manifest['complete']
    assert all(digest(Path(p)) == h for p,h in manifest['source_sha256'].items())
    binary = source/'leaf_decoder.exe'
    assert digest(binary) == manifest['binary_sha256']['leaf_decoder.exe']
    command = list(manifest['commands'][0])
    main = source/'main_decoder.cpp'
    text = main.read_text()
    text = '#include "process_probe.h"\n#include <filesystem>\n#include <thread>\n' + text
    text = replace_once(text, '        const double load_ms = elapsed(load_start);',
        '''        const double load_ms = elapsed(load_start);
        if (const char* gate = std::getenv("LEAF_PROCESS_PROBE_GATE")) {
            const auto registration_deadline = Clock::now() + std::chrono::seconds(10);
            while (leaf::experiments::process_worker_ids().size() != threads - 1) {
                if (Clock::now() >= registration_deadline) throw std::runtime_error("Worker registration timed out");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            {
                std::ofstream waiting(std::string(gate) + ".waiting");
                waiting << GetCurrentThreadId();
                for (auto tid : leaf::experiments::process_worker_ids()) waiting << ' ' << tid;
            }
            const auto deadline = Clock::now() + std::chrono::seconds(30);
            while (!std::filesystem::exists(gate)) {
                if (Clock::now() >= deadline) throw std::runtime_error("Process probe gate timed out");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }''')
    text = replace_once(text, '        std::vector<double> prefills, decodes;',
        '        std::vector<double> prefills, decodes;\n'
        '        leaf::experiments::ProcessSamples process_prefill, process_decode;')
    old = '''                    decoder.reset(); auto start = Clock::now(); decoder.forward(prefix); const double prefill = elapsed(start);
                    start = Clock::now(); decoder.forward(last); const double decode = elapsed(start);'''
    new = '''                    decoder.reset();
                    const auto process_start = leaf::experiments::process_stamp();
                    auto start = Clock::now(); decoder.forward(prefix); const double prefill = elapsed(start);
                    const auto process_middle = leaf::experiments::process_stamp();
                    start = Clock::now(); decoder.forward(last); const double decode = elapsed(start);
                    const auto process_end = leaf::experiments::process_stamp();
                    if (iteration >= warmup) {
                        process_prefill.add(process_start, process_middle);
                        process_decode.add(process_middle, process_end);
                    }'''
    text = replace_once(text, old, new)
    text = replace_once(text, '        metrics << ",\\\"generated_tokens\\\":"; array(metrics, generated);',
        '        metrics << ",\\\"generated_tokens\\\":"; array(metrics, generated);\n'
        '        metrics << ",\\\"process_diagnostic_build\\\":true";\n'
        '        process_prefill.emit(metrics, "prefill"); process_decode.emit(metrics, "decode");')
    generated = directory/'main_decoder.cpp'
    generated.write_text(text, newline='\n')
    # Register only Leaf's actual worker IDs, so Windows loader/service threads
    # in this process cannot be mistaken for inference workers by the monitor.
    host = next(Path(p) for p in manifest['source_sha256']
                if Path(p).as_posix().endswith('/engine/include/leaf/runtime/host_support.h'))
    private_host = directory/'include/leaf/runtime/host_support.h'
    private_host.parent.mkdir(parents=True)
    host_text = '#include "process_probe.h"\n' + host.read_text()
    host_text = replace_once(host_text, '            std::size_t observed = 0;',
        '            leaf::experiments::register_process_worker();\n            std::size_t observed = 0;')
    private_host.write_text(host_text,newline='\n')
    output = directory/'leaf_decoder.exe'
    command[command.index(str(main))] = str(generated)
    command[command.index('-o')+1] = str(output)
    command[1:1] = ['-I', str(directory/'include'), '-I', str(HERE)]
    inputs = dict(manifest['source_sha256'])
    inputs.update({str(p):digest(p) for p in [Path(__file__), HERE/'process_probe.h', source/'build.json', generated, private_host, binary]})
    report = dict(complete=False, acceptance_timing=False, inputs=inputs, command=command,
                  candidate_binary=str(binary))
    write_record(directory/'build.json', report)
    subprocess.run(command, check=True)
    assert all(digest(Path(p)) == h for p,h in inputs.items())
    report.update(complete=True, binary_sha256=digest(output))
    write_record(directory/'build.json', report)
    print('Built process-counter diagnostic', flush=True)


class CoreCounters(Counters):
    def __init__(self, cores):
        # Reuse the tested PDH ABI; omit wildcard process counters entirely.
        self.dll = ctypes.WinDLL('pdh')
        self.query = ctypes.c_void_p()
        self.dll.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_void_p)]
        self.dll.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_void_p)]
        self.dll.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        self.dll.PdhGetFormattedCounterValue.argtypes = [ctypes.c_void_p,ctypes.c_uint32,ctypes.c_void_p,ctypes.POINTER(CounterValue)]
        self.dll.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        for name in ('PdhOpenQueryW','PdhAddEnglishCounterW','PdhCollectQueryData','PdhGetFormattedCounterValue','PdhCloseQuery'):
            getattr(self.dll,name).restype = ctypes.c_uint32
        if self.dll.PdhOpenQueryW(None,0,ctypes.byref(self.query)):
            raise RuntimeError('Cannot open PDH query')
        self.handles, self.unavailable = {}, {}
        for cpu in cores:
            for key,suffix in [('performance_percent','% Processor Performance'),('busy_percent','% Processor Time')]:
                handle = ctypes.c_void_p()
                status = self.dll.PdhAddEnglishCounterW(self.query,
                    fr'\Processor Information(0,{cpu})\{suffix}',0,ctypes.byref(handle))
                name = f'cpu_{cpu}_{key}'
                if status: self.unavailable[name] = int(status)
                else: self.handles[name] = handle
        self.dll.PdhCollectQueryData(self.query)


def pin_threads(pid, threads, cores, strict):
    present = {t.id for t in psutil.Process(pid).threads()}
    if len(set(threads))!=len(cores) or not set(threads)<=present:
        raise ValueError('Expected exactly one inference thread per selected core')
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenThread.argtypes = [ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
    kernel.OpenThread.restype = ctypes.c_void_p
    kernel.SetThreadAffinityMask.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
    kernel.SetThreadAffinityMask.restype = ctypes.c_size_t
    class GroupAffinity(ctypes.Structure):
        _fields_ = [('mask',ctypes.c_size_t),('group',ctypes.c_uint16),('reserved',ctypes.c_uint16*3)]
    kernel.GetThreadGroupAffinity.argtypes = [ctypes.c_void_p,ctypes.POINTER(GroupAffinity)]
    kernel.GetThreadGroupAffinity.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]; kernel.CloseHandle.restype = ctypes.c_int
    placements = []
    for tid,cpu in zip(threads,cores):
        handle = kernel.OpenThread(0x20|0x40,False,tid)
        if not handle: raise OSError(ctypes.get_last_error(),'Cannot open owned inference thread')
        try:
            before,after = GroupAffinity(),GroupAffinity()
            if not kernel.GetThreadGroupAffinity(handle,ctypes.byref(before)):
                raise OSError(ctypes.get_last_error(),'Cannot read worker affinity')
            if before.group!=0 or not before.mask & (1<<cpu):
                raise ValueError('Requested worker core outside original allowed set')
            if strict and not kernel.SetThreadAffinityMask(handle,1<<cpu):
                raise OSError(ctypes.get_last_error(),'Cannot pin owned inference thread')
            if not kernel.GetThreadGroupAffinity(handle,ctypes.byref(after)):
                raise OSError(ctypes.get_last_error(),'Cannot verify worker affinity')
            if strict and (after.group!=0 or after.mask!=1<<cpu):
                raise ValueError('Worker affinity differs from request')
            placements.append(dict(thread_id=tid,cpu=cpu if strict else None,
                                   before_mask=int(before.mask),after_mask=int(after.mask)))
        finally: kernel.CloseHandle(handle)
    return placements


def run(directory, model, strict_abba=False):
    report = directory/(model+('-placement' if strict_abba else '')+'-diagnostic.json')
    if report.exists(): raise ValueError('Retain diagnostics; choose a fresh directory')
    build_record = json.loads((directory/'build.json').read_text())
    binary = directory/'leaf_decoder.exe'
    assert build_record['complete'] and build_record['binary_sha256'] == digest(binary)
    assert all(digest(Path(p)) == h for p,h in build_record['inputs'].items())
    prepared = ROOT/'build/source-bf16-20261020/prepared.json'
    artifact = Path(json.loads(prepared.read_text())['models'][model]['after'])
    token_path = ROOT/'build/numerics-20261017/decode-silu'/(model+'-work/tokens.json')
    ids = json.loads(token_path.read_text())['benchmark_ids']
    reference = Path(build_record['candidate_binary'])
    inputs = {str(p):digest(p) for p in [binary, reference, prepared, artifact, token_path,
              Path(__file__), directory/'build.json', ROOT/'tools/decoder_validation.py',
              ROOT/'tools/diagnose_windows_benchmark.py', ROOT/'leaf/power.py']}
    me = psutil.Process(); previous = me.cpu_affinity(); cores = [2,4,6,8]
    record = dict(complete=False, acceptance_timing=False, scope=__doc__, inputs=inputs,
                  utc=utc_now(), cores=cores, monitor_cpu=27, passes=[],strict_affinity_abba=strict_abba)
    write_record(report,record)
    os.environ['LEAF_BENCHMARK_REQUIRE_AC'] = '1'
    scratch = directory/'tmp'; scratch.mkdir(exist_ok=True)
    os.environ['TEMP'] = os.environ['TMP'] = str(scratch)
    import tempfile
    tempfile.tempdir = str(scratch)
    # Validate the instrumented executable against the same ordinary forward.
    a,_ = run_native(reference,artifact,[ids[:33]],threads=4)
    b,_ = run_native(binary,artifact,[ids[:33]],threads=4)
    record['full_logit_bit_exact'] = bool(np.array_equal(a.view('<u4'), b.view('<u4')))
    if not record['full_logit_bit_exact']: raise ValueError('Instrumentation changed logits')
    counters = CoreCounters(cores)
    record['counter_unavailable'] = counters.unavailable
    spec = importlib.util.spec_from_file_location('storage_guards',HERE.parent/'source-bf16/integration.py')
    guards = importlib.util.module_from_spec(spec); spec.loader.exec_module(guards)
    try:
        with keep_awake(display=True):
            for index in range(4):
                samples = []
                strict = strict_abba and index in (1,2)
                me.cpu_affinity([27]); counters.read()
                with tempfile.TemporaryDirectory(dir=scratch) as temporary:
                    work = Path(temporary)
                    request = work/'tokens.bin'
                    request.write_bytes(struct.pack('<II',1,len(ids))+struct.pack(f'<{len(ids)}I',*ids))
                    command = [str(binary),str(artifact),str(request),str(work/'logits.bin'),
                               str(work/'metrics.json'),'bench','4','31','10','16','0','1','32']
                    gate = work/'start.ready'
                    environment = dict(os.environ,LEAF_PROCESS_PROBE_GATE=str(gate))
                    # Set the creator's mask before spawn, then immediately
                    # restore the monitor to its separate physical core.
                    me.cpu_affinity(cores)
                    try:
                        child = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            text=True,env=environment,creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS)
                    finally: me.cpu_affinity([27])
                    try:
                        owned = psutil.Process(child.pid)
                        if owned.cpu_affinity()!=cores or owned.nice()!=psutil.ABOVE_NORMAL_PRIORITY_CLASS:
                            raise ValueError('Child affinity or priority differs from request')
                        deadline = time.monotonic()+15
                        waiting = Path(str(gate)+'.waiting')
                        while not waiting.exists() or not waiting.read_text():
                            if child.poll() is not None or time.monotonic()>=deadline:
                                raise ValueError('Owned decoder did not reach its placement gate')
                            time.sleep(0.01)
                        placements = pin_threads(child.pid,list(map(int,waiting.read_text().split())),cores,strict)
                        with guards.power_window() as power:
                            gate.touch()
                            while child.poll() is None:
                                try: child.wait(timeout=0.5)
                                except subprocess.TimeoutExpired: pass
                                start = time.perf_counter()
                                values = counters.read()
                                topology = cpu_sets()
                                samples.append(dict(unix_seconds=time.time(),counters=values,
                                    parked={str(c['logical_cpu']):bool(c['flags'] & 1) for c in topology
                                            if c['group']==0 and c['logical_cpu'] in cores},
                                    collection_seconds=time.perf_counter()-start))
                            stdout,stderr = child.communicate()
                        if child.returncode: raise RuntimeError(stderr)
                        metrics = json.loads((work/'metrics.json').read_text())
                    except BaseException:
                        if child.poll() is None: child.kill()
                        child.communicate()
                        raise
                if not metrics.get('process_diagnostic_build') or not metrics['benchmark_ac_checked']:
                    raise ValueError('Missing native diagnostic or AC guard')
                for phase in ('prefill','decode'):
                    for key in ('cpu_ms','cycles','faults','starts_unix_seconds','ends_unix_seconds'):
                        values = metrics['process_'+key+'_'+phase]
                        if len(values)!=31 or not all(np.isfinite(v) and v>=0 for v in values):
                            raise ValueError('Invalid diagnostic samples')
                item = dict(index=index, strict_affinity=strict, placements=placements,
                    metrics=metrics, telemetry=samples, power=power,
                    stability={p:latency_stability(metrics[p+'_samples_ms']) for p in ('prefill','decode')})
                record['passes'].append(item); write_record(report,record)
                print(model,index+1,'strict',strict,{p:dict(wall_ms=metrics[p+'_p50_ms'],
                    cpu_ms=statistics.median(metrics['process_cpu_ms_'+p]),
                    cycles=statistics.median(metrics['process_cycles_'+p])) for p in ('prefill','decode')},flush=True)
            record['sleep_events'] = guards.sleep_events(record['utc'])
        if any(digest(Path(p))!=h for p,h in inputs.items()): raise ValueError('Diagnostic inputs changed')
        record['complete'] = True
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        counters.close(); me.cpu_affinity(previous); write_record(report,record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('build','run'))
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--model',choices=('smol','qwen'),default='smol')
    parser.add_argument('--strict-affinity-abba',action='store_true')
    args = parser.parse_args(); directory = args.directory.resolve()
    if args.stage=='build':
        directory.mkdir(parents=True,exist_ok=False)
        if args.source is None: parser.error('--source is required to build')
        build(args.source.resolve(),directory)
    else: run(directory,args.model,args.strict_affinity_abba)


if __name__=='__main__': main()
