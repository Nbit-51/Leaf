"""Build an instrumented source copy and profile frozen embedding inputs on Windows.

Never changes production sources. Timers overlap: attention children are subsets
of attention, and pack/GEMM are subsets of projection totals. Not acceptance timing.
"""
import argparse
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.power import keep_awake
from tools.embedding_validation import parity
from tools.validate_decoder import digest, utc_now

HEADER = r'''
#pragma once
#include <chrono>
#include <iostream>
#include <map>
#include <string>
namespace leaf_profile {
inline std::map<std::string, double> times;
struct Timer {
    std::string name;
    std::chrono::steady_clock::time_point start;
    explicit Timer(std::string n): name(std::move(n)), start(std::chrono::steady_clock::now()) {}
    ~Timer() { times[name] += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(); }
};
}
'''


def instrument(source, replacements):
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Profile anchor no longer unique: ' + old)
        source = source.replace(old, new)
    return '#include "leaf_profile.h"\n' + source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifact', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--build-only', action='store_true', help='check instrumentation compilation without profiling')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    args.output.mkdir(parents=True)
    saved = json.loads((args.reference / 'reference.json').read_text())
    for name, sha in saved['fixture_sha256'].items():
        if digest(args.reference / name) != sha:
            raise ValueError('Frozen reference changed')
    with np.load(args.reference / 'reference.npz') as data:
        ids = [row[mask.astype(bool)].tolist() for row, mask in zip(data['input_ids'], data['attention_mask'])]
        ids.append(data['long_input_ids'][0][data['long_attention_mask'][0].astype(bool)].tolist())
        expected = np.concatenate([data['embeddings'], data['long_embeddings']])
    encoder = instrument((ROOT / 'engine/src/embedding.cpp').read_text(), [
        ('float epsilon) {', 'float epsilon) { leaf_profile::Timer timer("norm");'),
        ('const auto &w = weights.at(name);', 'const auto group = name.find(".mlp.") != std::string::npos ? "mlp" : name.find("pool.") == 0 ? "output" : "projection";\n leaf_profile::Timer total(std::string(group)+"_total");\n const auto &w = weights.at(name);'),
        ('panels.pack(x.data(), tokens, w.cols, w.cols);', '{ leaf_profile::Timer timer(std::string(group)+"_pack"); panels.pack(x.data(), tokens, w.cols, w.cols); }\n leaf_profile::Timer compute(std::string(group)+"_gemm");'),
        ('std::vector<float> &sine) {', 'std::vector<float> &sine) { leaf_profile::Timer timer("rope_tables");'),
        ('const float *sine = nullptr) {', 'const float *sine = nullptr) { leaf_profile::Timer timer("reorder_rope");'),
        ('const auto tokens = ids.size();', 'leaf_profile::times.clear(); const auto start = std::chrono::steady_clock::now();\n const auto tokens = ids.size();'),
        ('attention.resize(tokens * heads * dim);', '{ leaf_profile::Timer timer(local[i] ? "attention_local" : "attention_global");\n attention.resize(tokens * heads * dim);'),
        ('linear(attention, tokens, p + "self_attn.o_proj.weight", projected);', '}\n linear(attention, tokens, p + "self_attn.o_proj.weight", projected);'),
        ('kernels::gelu_new_inplace(gate.data(), gate.size(), avx);', '{ leaf_profile::Timer timer("gelu_gate"); kernels::gelu_new_inplace(gate.data(), gate.size(), avx);'),
        ('linear(gate, tokens, p + "mlp.down_proj.weight", projected);', '}\n linear(gate, tokens, p + "mlp.down_proj.weight", projected);'),
        ('return result;', r'''leaf_profile::times["total"] = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        std::cerr << "{\"tokens\":" << tokens;
        for (const auto& item : leaf_profile::times) std::cerr << ",\"" << item.first << "\":" << item.second;
        std::cerr << "}\n";
        return result;'''),
    ])
    attention = instrument((ROOT / 'engine/include/leaf/kernels/attention_vector.h').read_text(), [
        ('float denominator = 0;', 'denominator = 0;'),
        ('std::size_t count = 0;', 'std::size_t count = 0;\n { leaf_profile::Timer score_timer("attention_scores");'),
        ('exp_scores(scores.data(), keys, maximum);', '}\n float denominator;\n { leaf_profile::Timer softmax_timer("attention_softmax");\n float maximum = -std::numeric_limits<float>::infinity();\n for (auto score : scores) maximum = std::max(maximum, score);\n exp_scores(scores.data(), keys, maximum);'),
        ('// Finite values and a finite softmax', '}\n leaf_profile::Timer value_timer("attention_values");\n // Finite values and a finite softmax'),
    ])
    # The extra maximum pass above is instrumentation overhead; preserve output
    # but disclose it, and never use this binary for acceptance timing.
    include = args.output / 'include'
    shutil.copytree(ROOT / 'engine/include', include)
    (include / 'leaf/kernels/attention_vector.h').write_text(attention)
    (include / 'leaf_profile.h').write_text(HEADER)
    source = args.output / 'embedding.cpp'
    source.write_text(encoder)
    exe = args.output / 'profile.exe'
    subprocess.run(['g++', '-std=c++17', '-O3', '-DNDEBUG', '-ffp-contract=off', '-pthread', '-static',
                    '-I', str(include), str(ROOT / 'benchmark/embedding_runtime_benchmark.cpp'), str(source),
                    str(ROOT / 'engine/src/kernels/transformer.cpp'), '-o', str(exe)], check=True)
    if args.build_only:
        print('Instrumented build passed; no timing samples collected')
        return
    request = args.output / 'request.bin'
    with request.open('wb') as stream:
        stream.write(b'LEAFER01' + struct.pack('<I', len(ids)))
        for row in ids:
            stream.write(struct.pack('<I', len(row)) + np.asarray(row, dtype='<u4').tobytes())
    output = args.output / 'vectors.bin'
    hashes = {str(p): digest(p) for p in [source, exe, include / 'leaf/kernels/attention_vector.h',
              include / 'leaf/kernels/row_reuse.h', args.artifact, request, Path(__file__)]}
    with keep_awake(), (args.output / 'profile.jsonl').open('w') as err:
        parent = psutil.Process()
        affinity = parent.cpu_affinity()
        try:
            parent.cpu_affinity([args.cpu])
            process = subprocess.Popen([str(exe.resolve()), str(args.artifact.resolve()), str(request.resolve()),
                                        str(output.resolve()), '1', '1', '3', '.'], stdout=subprocess.PIPE, stderr=err, text=True)
        finally:
            parent.cpu_affinity(affinity)
        stdout, _ = process.communicate()
        if process.returncode:
            raise RuntimeError('Profile failed; inspect retained log')
    quality = parity(np.fromfile(output, dtype='<f4').reshape(len(ids), 768), expected)
    samples = [json.loads(line) for line in (args.output / 'profile.jsonl').read_text().splitlines()]
    if len(samples) != len(ids) * 4:
        raise ValueError('Incomplete operation samples')
    summaries = [{key: float(np.median([s[key] for s in samples[4*i+1:4*i+4]]))
                  for key in samples[4*i]} for i in range(len(ids))]
    if any(digest(Path(p)) != sha for p, sha in hashes.items()):
        raise ValueError('Profile inputs changed')
    record = dict(format='leaf-embedding-operation-profile-v2', measured_at_utc=utc_now(), hashes=hashes,
                  cpu=args.cpu, threads=1, warmups=1, runs=3, complete=True, timing_qualified=False,
                  scope=__doc__, overhead='Per-query attention timers and an extra maximum reduction; diagnostic only.',
                  parity=quality, adapter=json.loads(stdout), samples=samples, medians=summaries)
    (args.output / 'profile.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
