"""Bind the include-only portability fix to the already measured Windows source."""
import hashlib
import json
from pathlib import Path
import subprocess

directory=Path(__file__).resolve().parent
root=directory.parents[1]
source=root/'engine/src/decoder.cpp'
new=source.read_bytes()
after=b'#include "leaf/kernels/row_reuse.h"\n#endif\n#include "leaf/kernels/source_bf16.h"'
before=b'#include "leaf/kernels/row_reuse.h"\n#include "leaf/kernels/source_bf16.h"\n#endif'
assert new.count(after)==1
old=new.replace(after,before)
build=json.loads((directory/'accurate-final/build.json').read_text())
digest=lambda b:hashlib.sha256(b).hexdigest()
assert digest(old)==build['source_sha256'][str(source)]
original=directory/'before-portability-decoder.cpp'
original.write_bytes(old)
commands=[]
preprocessed=[]
for label,path in [('before',original),('after',source)]:
    output=directory/('preprocessed-'+label+'.cpp')
    command=['C:/msys64/mingw64/bin/g++.exe','-std=c++17','-O3','-DNDEBUG','-pthread',
             '-I',str(root/'engine/include'),'-E','-P',str(path),'-o',str(output)]
    subprocess.run(command,check=True)
    commands.append(command)
    preprocessed.append(digest(output.read_bytes()))
assert preprocessed[0]==preprocessed[1], 'Windows preprocessed source changed'
record=dict(passed=True,scope='Header moved outside Windows optimized guard; default Windows preprocessed source is identical',
            before_source_sha256=digest(old),after_source_sha256=digest(new),
            preprocessed_sha256=preprocessed,commands=commands)
(directory/'include-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print('PASS: include-only fix retains identical Windows preprocessed source')
