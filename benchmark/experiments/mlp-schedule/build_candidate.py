"""Reconstruct the unqualified worker-grain candidate under ignored build/.

Never edits the default engine. The private source tree must be fresh, and the
saved patch must apply cleanly; changed anchors fail instead of approximating.
"""
import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from tools.validate_decoder import digest, write_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/'build').resolve()):
        parser.error('Private candidates must stay under this repository build/ directory')
    output.mkdir(parents=True,exist_ok=False)
    source = output/'source'
    shutil.copytree(ROOT/'engine',source/'engine')
    patch = HERE/'candidate.patch'
    original = {str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'engine').rglob('*') if p.is_file()}
    record = dict(complete=False,release_default_changed=False,performance_qualified=False,
                  base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  original_engine=original,inputs={str(p):digest(p) for p in
                    [Path(__file__),patch,HERE/'worker_grain_test.cpp']})
    report = output/'candidate-build.json'
    write_record(report,record)
    try:
        command = ['git','apply','--directory',str(source.relative_to(ROOT)).replace('\\','/'),str(patch)]
        subprocess.run([*command[:2],'--check',*command[2:]],cwd=ROOT,check=True)
        subprocess.run(command,cwd=ROOT,check=True)
        builder = HERE.parent/'linear-accumulation/build.py'
        record['inputs'][str(builder)] = digest(builder)
        spec = importlib.util.spec_from_file_location('isolated_linear_builder',builder)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.ROOT = source
        previous = sys.argv
        try:
            sys.argv = [str(builder),'--output',str(output/'accurate'),'--precise-scalar',
                        '--tree-outer','--tree-chunk-size','256','--reuse-packing']
            module.main()
        finally: sys.argv = previous
        compiler = 'C:/msys64/mingw64/bin/g++.exe'
        test = output/'worker_grain_test.exe'
        subprocess.run([compiler,'-std=c++17','-O3','-pthread','-static',
            '-I',str(source/'engine/include'),str(HERE/'worker_grain_test.cpp'),'-o',str(test)],check=True)
        result = subprocess.run([str(test)],check=True,capture_output=True,text=True)
        record['worker_test'] = result.stdout.strip()
        record['binary_sha256'] = digest(output/'accurate/leaf_decoder.exe')
        if any(digest(ROOT/p)!=h for p,h in original.items()):
            raise ValueError('Default engine changed during private reconstruction')
        if any(digest(Path(p))!=h for p,h in record['inputs'].items()):
            raise ValueError('Candidate build inputs changed')
        record['complete'] = True
        print(record['worker_test'],flush=True)
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally: write_record(report,record)


if __name__=='__main__': main()
