"""Check the native SiLU contract and compare against an immutable Git header."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from leaf.power import keep_awake, require_benchmark_ac
from tools.decoder_validation import latency_stability
from tools.validate_decoder import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-ref', default='f814501886b94eb8717b8782ef2f76da447240c2',
                        help='immutable pre-pipeline header revision, independent of current HEAD')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git','rev-parse','--verify','--end-of-options',
                                     args.baseline_ref + '^{commit}'], cwd=ROOT, text=True).strip()
    header = subprocess.check_output(['git','show',commit + ':engine/include/leaf/kernels/silu_gate.h'], cwd=ROOT, text=True)
    previous = output / 'silu_before.h'
    previous.write_text(header.replace('namespace leaf::kernels {', 'namespace leaf::before_kernels {'), newline='\n')
    paths = [Path(__file__), Path(__file__).with_name('probe.cpp'), previous,
             ROOT/'engine/include/leaf/kernels/silu_gate.h', ROOT/'tests/native/test_silu_gate.cpp']
    hashes = {str(p):digest(p) for p in paths}
    report = dict(complete=False, acceptance_timing=False, baseline_commit=commit, hashes=hashes)
    record = output/'results.json'
    record.write_text(json.dumps(report, indent=2))
    common = ['C:/msys64/mingw64/bin/g++.exe','-std=c++17','-O3','-DNDEBUG','-static','-ffp-contract=off',
              '-I',str(ROOT/'engine/include'),'-I',str(output)]
    try:
        test, probe = output/'test_silu_gate.exe', output/'probe.exe'
        subprocess.run(common+[str(ROOT/'tests/native/test_silu_gate.cpp'),'-o',str(test)], check=True)
        subprocess.run(common+[str(Path(__file__).with_name('probe.cpp')),'-o',str(probe)], check=True)
        checked = subprocess.check_output([str(test)],text=True)
        report['native_correctness'] = checked.strip()
        report['binary_sha256'] = {p.name:digest(p) for p in (test,probe)}
        print(checked, end='', flush=True)
        with keep_awake():
            report['power_before'] = require_benchmark_ac()
            result = subprocess.run([str(probe)],capture_output=True,text=True)
            report['power_after'] = require_benchmark_ac()
        (output/'stdout.txt').write_text(result.stdout)
        (output/'stderr.txt').write_text(result.stderr)
        result.check_returncode()
        report['native'] = json.loads(result.stdout)
        for shape in report['native']['shapes']:
            samples = {s:[v for p in shape['passes'] if p['stage']==s for v in p['ms']] for s in ('before','after')}
            medians = {s:statistics.median(v) for s,v in samples.items()}
            shape['median_ms'] = medians
            shape['ratio'] = medians['after']/medians['before']
            shape['stability'] = {s:latency_stability(v) for s,v in samples.items()}
            print(shape['elements'], medians, 'ratio',round(shape['ratio'],5), flush=True)
        if any(digest(Path(p)) != h for p,h in hashes.items()):
            raise ValueError('Probe sources changed during execution')
        report['complete'] = True
    finally:
        record.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
