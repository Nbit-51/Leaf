"""Build an isolated candidate from current sources; never edits release code."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Source anchor changed; inspect the candidate transformation')
    return text.replace(before, after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiler', default='C:/msys64/mingw64/bin/g++.exe')
    parser.add_argument('--inline-partial', action='store_true',
                        help='eliminate the partial-call boundary without changing arithmetic')
    parser.add_argument('--cache-blocked', action='store_true',
                        help='reuse each K chunk across the existing 96-row block; implies inline partial')
    parser.add_argument('--precise-scalar', action='store_true',
                        help='use FP64 dot reductions only for pure-FP32 scalar execution')
    parser.add_argument('--tree-outer', action='store_true',
                        help='combine K=128 partials with a balanced FP32 tree instead of FP64')
    parser.add_argument('--reuse-packing', action='store_true',
                        help='explicitly reuse identical unscaled FP32 Q/K/V and up/gate input panels')
    parser.add_argument('--tree-chunk-size', type=int, choices=(128, 256), default=128,
                        help='global FP32 tree chunk width, independent of model identity')
    parser.add_argument('--tree-cache', action='store_true',
                        help='reuse each input chunk across twelve output rows with independent FP32 trees')
    args = parser.parse_args()
    if args.tree_outer and args.cache_blocked:
        parser.error('Choose the FP32 tree or the historical cache variant, not both')
    if args.tree_chunk_size != 128 and not args.tree_outer:
        parser.error('--tree-chunk-size requires --tree-outer')
    if args.tree_cache and not args.tree_outer:
        parser.error('--tree-cache requires --tree-outer')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = ROOT / 'engine/src/decoder.cpp'
    text = original.read_text()
    header = 'tree_sum.h' if args.tree_outer else 'blocked_sum.h'
    text = replace_once(text, '#include "leaf/kernels/row_reuse.h"', '#include "' + header + '"')
    text = replace_once(text, 'std::vector<float>& output, std::size_t width) {',
                        'std::vector<float>& output, std::size_t width, bool precise_rms) {')
    old = '''        for (std::size_t col = 0; col < width; ++col) {
            const float value = input[row * width + col] - mean; square += value * value;
        }'''
    text = replace_once(text, old, '''        if (kind == 0 && precise_rms) {
            square = leaf::experiments::rms_square_sum(input.data() + row * width, width);
        } else {
''' + old + '''
        }''')
    text = replace_once(text, 'epsilon, norm_kind, output, h);',
                        'epsilon, norm_kind, output, h, fp32_artifact);')
    if args.precise_scalar:
        text = replace_once(text, 'void quant_dot_scalar(', '''void dot_scalar_precise(const Weight& w, std::size_t row, const float* inputs,
                        std::size_t tokens, float* result) {
    const auto* values = w.bits == 32 ? reinterpret_cast<const float*>(w.data) + row * w.cols : nullptr;
    for (std::size_t token = 0; token < tokens; ++token) {
        double total = 0.0;
        for (std::size_t col = 0; col < w.cols; ++col)
            total += static_cast<double>(w.bits == 32 ? values[col] : w.value(row, col)) * inputs[token * w.cols + col];
        result[token] = static_cast<float>(total);
    }
}

void quant_dot_scalar(''')
        text = replace_once(text, '        optimized_fp32 = LEAF_OPTIMIZED_FP32 && avx && fp32_artifact;',
            '''        if (fp32_artifact && !avx) dot = dot_scalar_precise;
        optimized_fp32 = LEAF_OPTIMIZED_FP32 && avx && fp32_artifact;''')
    # Restrict changed arithmetic to the pure-FP32 policy, including research builds.
    kernel = 'gemm_tree_sum' if args.tree_outer else 'gemm_blocked_sum'
    text = replace_once(text, 'leaf::kernels::gemm_token_panels_f32_row_reuse(',
        '(fp32_artifact ? leaf::experiments::' + kernel + ' : leaf::kernels::gemm_token_panels_f32_row_reuse)(')
    text = replace_once(text, 'bias ? reinterpret_cast<const float*>(bias->data) + begin : nullptr);',
                        'bias ? reinterpret_cast<const float*>(bias->data) + begin : nullptr, true);')
    if args.reuse_packing:
        text = replace_once(text, '    leaf::kernels::TokenPanelsF32 packed_float_input;',
            '''    leaf::kernels::TokenPanelsF32 packed_float_input;
    const float* packed_source = nullptr;
    std::size_t packed_tokens = 0, packed_columns = 0;''')
        text = replace_once(text, 'bool mlp_projection = false) {',
                            'bool mlp_projection = false, bool reuse_packed = false) {')
        text = replace_once(text, '        bool vector_pack = false;',
                            '        bool vector_pack = optimized_fp32 && avx;')
        text = replace_once(text,
            '            packed_float_input.pack(input, tokens, weight.cols, weight.cols, vector_pack);',
            '''            // The caller explicitly promises unchanged input only within a
            // Q/K/V or up/gate group. Pointer equality alone is never a ticket.
            const bool plain = fp32_artifact && avx && scaling == weights.end();
            const bool reuse = plain && reuse_packed && packed_source == input &&
                               packed_tokens == tokens && packed_columns == weight.cols;
            if (!reuse) packed_float_input.pack(input, tokens, weight.cols, weight.cols, vector_pack);
            packed_source = plain ? input : nullptr;
            packed_tokens = tokens;
            packed_columns = weight.cols;''')
        for projection, target in [('k', 'k'), ('v', 'v')]:
            old_call = f'linear(normalized.data(), tokens, prefix + "self_attn.{projection}_proj", {target});'
            text = replace_once(text, old_call, old_call[:-2] + ', false, true);')
        text = replace_once(text,
            'linear(normalized.data(), mlp_tokens, prefix + "mlp.gate_proj", gate, true);',
            'linear(normalized.data(), mlp_tokens, prefix + "mlp.gate_proj", gate, true, true);')
    generated = output / 'decoder.cpp'
    generated.write_text(text, newline='\n')
    main_source = (ROOT / 'engine/src/main_decoder.cpp').read_text()
    anchor = '        metrics << ",\\\"generated_tokens\\\":"; array(metrics, generated);'
    main_source = replace_once(main_source, anchor,
        '        metrics << ",\\\"experimental_accumulation_build\\\":true";\n' + anchor)
    if args.reuse_packing:
        main_source = replace_once(main_source, anchor,
            '        metrics << ",\\\"experimental_input_pack_reuse_build\\\":true";\n' + anchor)
    generated_main = output / 'main_decoder.cpp'
    generated_main.write_text(main_source, newline='\n')
    common = [args.compiler, '-std=c++17', '-O3', '-DNDEBUG', '-pthread',
              '-static', '-static-libgcc', '-static-libstdc++', '-I', str(ROOT / 'engine/include'), '-I', str(HERE)]
    if args.inline_partial or args.cache_blocked or args.tree_outer:
        common.append('-DLEAF_ACCUMULATION_INLINE_PARTIAL=1')
    if args.cache_blocked:
        common.append('-DLEAF_ACCUMULATION_CACHE_BLOCKED=1')
    if args.tree_outer:
        common.append('-DLEAF_ACCUMULATION_TREE_OUTER=1')
        common.append('-DLEAF_ACCUMULATION_TREE_CHUNK=' + str(args.tree_chunk_size))
    if args.tree_cache:
        common.append('-DLEAF_ACCUMULATION_TREE_CACHE=1')
    jobs = [common + [str(generated), str(generated_main), str(ROOT / 'engine/src/kv_cache.cpp'),
                     str(ROOT / 'engine/src/kernels/transformer.cpp'), '-o', str(output / 'leaf_decoder.exe'), '-lpsapi'],
            common + [str(HERE / 'probe.cpp'), '-o', str(output / 'probe.exe')]]
    sources = [original, ROOT / 'engine/src/main_decoder.cpp', ROOT / 'engine/include/leaf/kernels/token_panel.h',
               ROOT / 'engine/include/leaf/kernels/row_reuse.h', HERE / 'blocked_sum.h', HERE / 'probe.cpp',
               Path(__file__), generated, generated_main]
    if args.tree_outer:
        sources.append(HERE / 'tree_sum.h')
    # The decoder includes activation, attention and runtime headers as well.
    # Retain all engine headers and translation units so a kernel-only change
    # cannot produce an apparently identical build manifest.
    sources.extend([ROOT / 'engine/src/kv_cache.cpp', ROOT / 'engine/src/kernels/transformer.cpp'])
    sources.extend(sorted((ROOT / 'engine/include').rglob('*.h')))
    sources = list(dict.fromkeys(sources))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest = dict(complete=False, release_default_changed=False, acceptance_timing=False,
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256=hashes, commands=jobs,
        inline_partial=args.inline_partial or args.cache_blocked or args.tree_outer,
        cache_blocked=args.cache_blocked, precise_scalar=args.precise_scalar, tree_outer=args.tree_outer,
        reuse_packing=args.reuse_packing, tree_chunk_size=args.tree_chunk_size if args.tree_outer else None,
        tree_cache=args.tree_cache)
    (output / 'build.json').write_text(json.dumps(manifest, indent=2))
    try:
        for command in jobs:
            subprocess.run(command, check=True, cwd=ROOT)
    except BaseException as error:
        manifest['error'] = str(error) or type(error).__name__
        (output / 'build.json').write_text(json.dumps(manifest, indent=2))
        raise
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    manifest.update(complete=True, binary_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in (output / 'leaf_decoder.exe', output / 'probe.exe')})
    (output / 'build.json').write_text(json.dumps(manifest, indent=2))
    print('Built isolated candidate and small probe:', output, flush=True)


if __name__ == '__main__':
    main()
