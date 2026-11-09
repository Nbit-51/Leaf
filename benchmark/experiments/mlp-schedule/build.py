"""Generate an isolated two-step K loop; released sources are never edited."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
LINEAR = HERE.parent / 'linear-accumulation'


def generate(output, tile_rows=6):
    panel = (ROOT/'engine/include/leaf/kernels/token_panel.h').read_text()
    first = panel.index('template <unsigned Rows>\nLEAF_TOKEN_PANEL_TARGET inline void accumulate_kblock(')
    last = panel.index('\nLEAF_TOKEN_PANEL_TARGET inline void gemm_kblocked(',first)
    partial = panel[first:last].replace('accumulate_kblock(', 'scheduled_accumulate(')
    partial = partial.replace('LEAF_TOKEN_PANEL_TARGET inline',
                              '__attribute__((target("avx2,fma"), always_inline)) inline',1)
    anchor = '    for (std::size_t col = 0; col < columns; ++col) {'
    if partial.count(anchor) != 1:
        raise ValueError('Inspect the changed partial-loop source before generating a candidate')
    partial = partial.replace(anchor,'#pragma GCC unroll 2\n'+anchor)
    tree = (LINEAR/'tree_sum.h').read_text()
    tree = tree.replace('tree_sum_detail','scheduled_sum_detail').replace(
        'gemm_tree_sum','gemm_scheduled_sum').replace('tree_chunk_size','scheduled_chunk_size')
    call = 'blocked_sum_detail::partial<Rows>('
    if tree.count(call) != 2:
        raise ValueError('Inspect the changed tree reduction before generating a candidate')
    tree = tree.replace(call,'scheduled_partial<Rows>(')
    anchor = 'const auto live = std::min<std::size_t>(6, rows - row);'
    if tree.count(anchor) != 1 or tile_rows not in (5,6):
        raise ValueError('Inspect row dispatch before changing scheduling')
    tree = tree.replace(anchor,anchor.replace('(6,','('+str(tile_rows)+','))
    wrapper = '''
template <unsigned Rows>
__attribute__((target("avx2,fma"), always_inline)) inline void scheduled_partial(
    const float* weights, std::size_t columns, std::size_t stride,
    const float* packed, float* result) {
    scheduled_accumulate<Rows>(weights, columns, stride, packed, result, true);
}
'''
    header = output/'scheduled_tree.h'
    header.write_text('#pragma once\n#include "tree_sum.h"\n'
        '#if defined(__x86_64__) && defined(__GNUC__)\nnamespace leaf::experiments {\n'
        +partial+wrapper+'}\n#endif\n'+tree.replace('#pragma once','',1),newline='\n')
    return header


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tile-rows',type=int,choices=(5,6),default=6)
    args=parser.parse_args(); output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    header=generate(output,args.tile_rows)
    command=['C:/msys64/mingw64/bin/g++.exe','-std=c++17','-O3','-DNDEBUG','-pthread',
        '-static','-static-libgcc','-static-libstdc++','-I',str(ROOT/'engine/include'),
        '-I',str(LINEAR),'-I',str(output),'-DLEAF_ACCUMULATION_INLINE_PARTIAL=1',
        '-DLEAF_ACCUMULATION_TREE_CHUNK=256',str(HERE/'probe.cpp'),'-o',str(output/'probe.exe')]
    sources=[Path(__file__),HERE/'probe.cpp',LINEAR/'tree_sum.h',LINEAR/'blocked_sum.h',header,
             ROOT/'engine/include/leaf/kernels/token_panel.h',ROOT/'engine/include/leaf/kernels/row_reuse.h']
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    hashes={str(p):digest(p) for p in sources}
    record=dict(complete=False,release_default_changed=False,tile_rows=args.tile_rows,command=command,source_sha256=hashes)
    report=output/'build.json'; report.write_text(json.dumps(record,indent=2)+'\n')
    subprocess.run(command,check=True)
    assert all(digest(Path(p))==h for p,h in hashes.items())
    record.update(complete=True,binary_sha256=digest(output/'probe.exe'))
    report.write_text(json.dumps(record,indent=2)+'\n')
    print('Built isolated scheduling probe:',output,flush=True)


if __name__=='__main__': main()
