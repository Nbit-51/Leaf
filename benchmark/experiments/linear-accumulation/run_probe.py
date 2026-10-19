"""Run a short isolated kernel probe, retaining stdout even if validation fails."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from leaf.affinity import pin_cpu
from tools.validate_decoder import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--timing', action='store_true')
    args = parser.parse_args()
    output = args.build / ('probe-timing.json' if args.timing else 'probe-correctness.json')
    if output.exists():
        raise ValueError('Refuse to overwrite retained probe results')
    runtime = (args.build / 'probe.exe').resolve()
    if digest(runtime) != json.loads((args.build / 'build.json').read_text())['binary_sha256'][runtime.name]:
        raise ValueError('Probe binary differs from completed build')
    affinity = None
    if args.timing:
        import psutil
        pin_cpu(2)
    child = subprocess.Popen([str(runtime)] + (['timing'] if args.timing else []),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if args.timing:
            affinity = psutil.Process(child.pid).cpu_affinity()
        stdout, stderr = child.communicate()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    output.with_suffix('.stdout.txt').write_text(stdout)
    output.with_suffix('.stderr.txt').write_text(stderr)
    if child.returncode:
        raise subprocess.CalledProcessError(child.returncode, child.args, stdout, stderr)
    record = json.loads(stdout)
    record.update(binary_sha256=digest(runtime), acceptance_timing=False,
                  full_model_speed_qualified=False, observed_child_affinity=affinity)
    if args.timing:
        if affinity != [2]:
            raise ValueError('Probe did not inherit the requested CPU affinity')
        record['pinned_cpu'] = 2
    output.write_text(json.dumps(record, indent=2) + '\n')
    print('Saved', output, flush=True)


if __name__ == '__main__':
    main()
