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
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = ROOT / 'engine/src/decoder.cpp'
    text = original.read_text()
    text = replace_once(text, '#include "leaf/kernels/row_reuse.h"', '#include "blocked_sum.h"')
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
    # Restrict changed arithmetic to the pure-FP32 policy, including research builds.
    text = replace_once(text, 'leaf::kernels::gemm_token_panels_f32_row_reuse(',
        '(fp32_artifact ? leaf::experiments::gemm_blocked_sum : leaf::kernels::gemm_token_panels_f32_row_reuse)(')
    text = replace_once(text, 'bias ? reinterpret_cast<const float*>(bias->data) + first : nullptr);',
                        'bias ? reinterpret_cast<const float*>(bias->data) + first : nullptr, true);')
    generated = output / 'decoder.cpp'
    generated.write_text(text, newline='\n')
    main_source = (ROOT / 'engine/src/main_decoder.cpp').read_text()
    anchor = '        metrics << ",\\\"generated_tokens\\\":"; array(metrics, generated);'
    main_source = replace_once(main_source, anchor,
        '        metrics << ",\\\"experimental_accumulation_build\\\":true";\n' + anchor)
    generated_main = output / 'main_decoder.cpp'
    generated_main.write_text(main_source, newline='\n')
    common = [args.compiler, '-std=c++17', '-O3', '-DNDEBUG', '-pthread',
              '-static', '-static-libgcc', '-static-libstdc++', '-I', str(ROOT / 'engine/include'), '-I', str(HERE)]
    jobs = [common + [str(generated), str(generated_main), str(ROOT / 'engine/src/kv_cache.cpp'),
                     str(ROOT / 'engine/src/kernels/transformer.cpp'), '-o', str(output / 'leaf_decoder.exe'), '-lpsapi'],
            common + [str(HERE / 'probe.cpp'), '-o', str(output / 'probe.exe')]]
    sources = [original, ROOT / 'engine/src/main_decoder.cpp', ROOT / 'engine/include/leaf/kernels/token_panel.h',
               ROOT / 'engine/include/leaf/kernels/row_reuse.h', HERE / 'blocked_sum.h', HERE / 'probe.cpp',
               Path(__file__), generated, generated_main]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest = dict(complete=False, release_default_changed=False, acceptance_timing=False,
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256=hashes, commands=jobs)
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
