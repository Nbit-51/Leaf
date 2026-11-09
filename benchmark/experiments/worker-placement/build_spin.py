"""Private bounded-spin experiment; no release-source or model-math changes.

Only explicitly placed multi-thread pools spin, for at most about 50 us before
falling back to their condition variable. Unassigned and single-thread callers
keep the blocking policy. The timer is checked at most 64 pause instructions apart.
"""
import argparse
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tools.validate_decoder import digest, write_record


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Inspect changed worker source anchor')
    return text.replace(before, after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-ref', default='31fff42',
        help='historical placement implementation before bounded spinning')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / 'build').resolve()):
        parser.error('Use a fresh private build/ directory')
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    original = {str(p): digest(p) for p in (ROOT / 'engine').rglob('*') if p.is_file()}
    source_commit = subprocess.check_output(['git','rev-parse',args.source_ref],cwd=ROOT,text=True).strip()
    archive = subprocess.check_output(['git','archive','--format=zip',source_commit,'engine'],cwd=ROOT)
    source.mkdir()
    with zipfile.ZipFile(io.BytesIO(archive)) as saved:
        for item in saved.infolist():
            if not item.filename.startswith('engine/') or not (source/item.filename).resolve().is_relative_to(source):
                raise ValueError('Archive escapes private source tree')
        saved.extractall(source)
    record = dict(complete=False, release_default_changed=False, acceptance_timing=False,
        source_commit=source_commit, original_engine=original,
        inputs={str(Path(__file__)): digest(Path(__file__))})
    report = output / 'candidate-build.json'
    write_record(report, record)
    try:
        host = source / 'engine/include/leaf/runtime/host_support.h'
        text = host.read_text()
        text = once(text, '#include <condition_variable>',
            '#include <condition_variable>\n#include <chrono>\n'
            '#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)\n#include <immintrin.h>\n#endif')
        text = once(text, 'pending_ = threads_ - 1; ++epoch_; }',
            'pending_ = threads_ - 1; ++epoch_;\n'
            '          if (spins()) visible_epoch_.store(epoch_, std::memory_order_release); }')
        text = once(text, '        ready_.notify_all(); drain();',
            '        ready_.notify_all(); drain();\n        if (spins()) spin_until(completed_epoch_, epoch_, true);')
        text = once(text, '            if (--pending_ == 0) finished_.notify_one();',
            '            if (--pending_ == 0) {\n'
            '                if (spins()) completed_epoch_.store(observed, std::memory_order_release);\n'
            '                finished_.notify_one();\n'
            '            }\n'
            '            lock.unlock();\n'
            '            if (spins()) spin_until(visible_epoch_, observed, false);')
        text = once(text, '    void drain() {', '''    bool spins() const noexcept { return caller_cpu_ != unbound_cpu && threads_ > 1; }
    static void spin_until(const std::atomic<std::size_t>& counter, std::size_t target, bool equal) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(50);
        unsigned checks = 0;
        for (;;) {
            if ((counter.load(std::memory_order_acquire) == target) == equal) return;
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
            _mm_pause();
#else
            std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
            if ((++checks & 63) == 0 && std::chrono::steady_clock::now() >= deadline) return;
        }
    }
    void drain() {''')
        text = once(text, '    std::exception_ptr task_error_;',
            '    std::exception_ptr task_error_;\n    std::atomic<std::size_t> visible_epoch_{0}, completed_epoch_{0};')
        host.write_text(text, newline='\n')
        main_source = source / 'engine/src/main_decoder.cpp'
        main_source.write_text(once(main_source.read_text(),
            '        metrics << ",\\\"worker_cpus\\\":"; array(metrics, worker_cpus);',
            '        metrics << ",\\\"worker_cpus\\\":"; array(metrics, worker_cpus);\n'
            '        metrics << ",\\\"experimental_worker_spin_us\\\":50";'), newline='\n')
        builder = HERE.parent / 'linear-accumulation/build.py'
        record['inputs'][str(builder)] = digest(builder)
        spec = importlib.util.spec_from_file_location('spin_accurate_builder', builder)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT = source
        previous = sys.argv
        try:
            sys.argv = [str(builder), '--output', str(output / 'accurate'), '--precise-scalar',
                '--tree-outer', '--tree-chunk-size', '256', '--reuse-packing']
            module.main()
        finally:
            sys.argv = previous
        test_source = ROOT / 'tests/native/test_worker_affinity.cpp'
        record['inputs'][str(test_source)] = digest(test_source)
        test = output / 'worker_affinity_test.exe'
        subprocess.run(['C:/msys64/mingw64/bin/g++.exe', '-std=c++17', '-O3', '-pthread', '-static',
            '-I', str(source / 'engine/include'), str(test_source), '-o', str(test)], check=True)
        record['worker_test'] = subprocess.check_output([str(test)], text=True, timeout=60).strip()
        if any(digest(Path(p)) != h for p, h in original.items()):
            raise ValueError('Public engine changed')
        if any(digest(Path(p)) != h for p, h in record['inputs'].items()):
            raise ValueError('Builder inputs changed')
        record.update(complete=True, binary_sha256=digest(output / 'accurate/leaf_decoder.exe'))
        print(record['worker_test'], flush=True)
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        write_record(report, record)


if __name__ == '__main__':
    main()
