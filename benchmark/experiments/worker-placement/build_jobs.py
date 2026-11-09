"""Isolate tile-aligned FP32 jobs while keeping GEMV's fixed-32 queue loop.

The private source tree retains the qualified explicit-placement implementation.
No current engine file is edited. All reductions, weights and kernel math stay
unchanged. Build only when no controlled timing is running.
"""
import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from tools.validate_decoder import digest,write_record


def once(text,before,after):
    if text.count(before)!=1:raise ValueError('Inspect changed job-scheduling source anchor')
    return text.replace(before,after)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to((ROOT/'build').resolve()):parser.error('Use an isolated repository build/ directory')
    output.mkdir(parents=True,exist_ok=False);source=output/'source'
    original={str(p):digest(p) for p in (ROOT/'engine').rglob('*') if p.is_file()}
    shutil.copytree(ROOT/'engine',source/'engine')
    record=dict(complete=False,release_default_changed=False,performance_qualified=False,
        inputs={str(Path(__file__)):digest(Path(__file__))},original_engine=original)
    report=output/'candidate-build.json';write_record(report,record)
    try:
        header=source/'engine/include/leaf/runtime/host_support.h';text=header.read_text()
        text=once(text,'    AffinityScope bind_caller() const',
            '    unsigned thread_count() const noexcept { return threads_; }\n    AffinityScope bind_caller() const')
        text=once(text,'std::function<void(std::size_t, std::size_t)> task) {',
            'std::function<void(std::size_t, std::size_t)> task, std::size_t grain = 32) {\n'
            '        check(grain > 0 && grain <= std::numeric_limits<unsigned>::max(), "invalid worker grain");')
        text=once(text,'count_ = count; next_ = 0;',
            'count_ = count; grain_ = static_cast<unsigned>(grain); next_ = 0;')
        text=once(text,'    unsigned pending_ = 0;',
            '    unsigned pending_ = 0;\n    unsigned grain_ = 32;')
        text=once(text,'    void drain() {\n        for (;;) {',
            '    void drain() {\n'
            '        if (grain_ == 32) drain_jobs<32>(); else drain_jobs<0>();\n'
            '    }\n'
            '    template <unsigned FixedGrain> void drain_jobs() {\n'
            '        const std::size_t grain = FixedGrain ? FixedGrain : grain_;\n'
            '        for (;;) {')
        text=once(text,'next_.fetch_add(32, std::memory_order_relaxed)',
            'next_.fetch_add(grain, std::memory_order_relaxed)')
        text=once(text,'std::min(first + 32, count_)','std::min(first + grain, count_)')
        header.write_text(text,newline='\n')
        decoder=source/'engine/src/decoder.cpp';text=decoder.read_text()
        text=once(text,'    std::size_t source_bf16_tensors = 0, source_bf16_linear_calls = 0;',
            '    std::size_t source_bf16_tensors = 0, source_bf16_linear_calls = 0;\n'
            '    std::size_t tile_aligned_linear_calls = 0;')
        text=once(text,'        workers.run(weight.rows, [&](std::size_t first, std::size_t last) {',
            '''        std::size_t job_grain = 32;
        if (packed_float && workers.thread_count() > 1) {
            const auto rows_per_worker = weight.rows / workers.thread_count();
            job_grain = rows_per_worker >= 384 ? 96 : rows_per_worker >= 192 ? 48 : 32;
            if (job_grain != 32) ++tile_aligned_linear_calls;
        }
        workers.run(weight.rows, [&](std::size_t first, std::size_t last) {''')
        text=once(text,'        });\n    }\n    void prepare_rotary_tables',
            '        }, job_grain);\n    }\n    void prepare_rotary_tables')
        text=once(text,'bool Decoder::uses_vnni() const',
            'std::size_t Decoder::tile_aligned_linear_calls() const { return impl_->tile_aligned_linear_calls; }\n'
            'bool Decoder::uses_vnni() const')
        decoder.write_text(text,newline='\n')
        public=source/'engine/include/leaf/runtime/decoder.h'
        public.write_text(once(public.read_text(),'    bool uses_vnni() const;',
            '    std::size_t tile_aligned_linear_calls() const;\n    bool uses_vnni() const;'),newline='\n')
        main=source/'engine/src/main_decoder.cpp'
        main.write_text(once(main.read_text(),'        metrics << ",\\\"worker_cpus\\\":"; array(metrics, worker_cpus);',
            '        metrics << ",\\\"worker_cpus\\\":"; array(metrics, worker_cpus);\n'
            '        metrics << ",\\\"tile_aligned_linear_calls\\\":" << decoder.tile_aligned_linear_calls();'),newline='\n')
        builder=HERE.parent/'linear-accumulation/build.py'
        record['inputs'][str(builder)]=digest(builder)
        spec=importlib.util.spec_from_file_location('placed_job_builder',builder)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.ROOT=source
        previous=sys.argv
        try:
            sys.argv=[str(builder),'--output',str(output/'accurate'),'--precise-scalar',
                '--tree-outer','--tree-chunk-size','256','--reuse-packing']
            module.main()
        finally:sys.argv=previous
        # The old exhaustive worker test also checks the new fixed-32 fast path.
        test_source=HERE.parent/'mlp-schedule/worker_grain_test.cpp'
        record['inputs'][str(test_source)]=digest(test_source)
        compiler='C:/msys64/mingw64/bin/g++.exe';test=output/'worker_grain_test.exe'
        subprocess.run([compiler,'-std=c++17','-O3','-pthread','-static','-I',str(source/'engine/include'),
                        str(test_source),'-o',str(test)],check=True)
        record['worker_test']=subprocess.check_output([str(test)],text=True).strip()
        if any(digest(Path(p))!=h for p,h in original.items()):raise ValueError('Current engine changed')
        if any(digest(Path(p))!=h for p,h in record['inputs'].items()):raise ValueError('Builder inputs changed')
        record.update(complete=True,binary_sha256=digest(output/'accurate/leaf_decoder.exe'))
        print(record['worker_test'],flush=True)
    except BaseException as error:
        record['error']=repr(error);raise
    finally:write_record(report,record)


if __name__=='__main__':main()
