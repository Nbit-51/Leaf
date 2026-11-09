"""Compare existing 32-row jobs with tile-aligned 96-row jobs, same GEMM math."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
LINEAR=HERE.parent/'linear-accumulation'


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate-source',type=Path,required=True,help='private source/ tree from build_candidate.py')
    args=p.parse_args(); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    source=args.candidate_source.resolve()
    host=source/'engine/include/leaf/runtime/host_support.h'
    worker='''class AlignedWorkers {
    leaf::runtime::detail::Workers workers_;
public:
    explicit AlignedWorkers(unsigned threads) : workers_(threads) {}
    void run(std::size_t count, std::function<void(std::size_t,std::size_t)> task) {
        const auto per_worker=count/workers_.thread_count();
        const auto grain=per_worker>=384?96:per_worker>=192?48:32;
        workers_.run(count,std::move(task),grain);
    }
};'''
    header=output/'aligned_workers.h'
    header.write_text('#pragma once\n#include "leaf/runtime/host_support.h"\n'
        'namespace leaf::experiments {\nusing leaf::runtime::detail::check;\n'+worker+'\n}\n',newline='\n')
    compiler='C:/msys64/mingw64/bin/g++.exe'
    command=[compiler,'-std=c++17','-O3','-DNDEBUG','-pthread','-static','-static-libgcc','-static-libstdc++',
        '-I',str(source/'engine/include'),'-I',str(LINEAR),'-I',str(output),
        '-DLEAF_ACCUMULATION_INLINE_PARTIAL=1','-DLEAF_ACCUMULATION_TREE_CHUNK=256',
        str(HERE/'queue_probe.cpp'),'-o',str(output/'probe.exe')]
    sources=[Path(__file__),HERE/'queue_probe.cpp',host,header,LINEAR/'tree_sum.h',LINEAR/'blocked_sum.h',
             source/'engine/include/leaf/kernels/token_panel.h',source/'engine/include/leaf/kernels/row_reuse.h']
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    hashes={str(p):digest(p) for p in sources}
    record=dict(complete=False,release_default_changed=False,scope='4-thread job grain, unchanged GEMM arithmetic',
                command=command,source_sha256=hashes)
    report=output/'build.json'; report.write_text(json.dumps(record,indent=2)+'\n')
    subprocess.run(command,check=True)
    assert all(digest(Path(p))==h for p,h in hashes.items())
    record.update(complete=True,binary_sha256=digest(output/'probe.exe'))
    report.write_text(json.dumps(record,indent=2)+'\n'); print('Built tile-aligned worker probe',output,flush=True)


if __name__=='__main__': main()
