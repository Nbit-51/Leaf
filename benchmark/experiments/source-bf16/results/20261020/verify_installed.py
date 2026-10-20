"""Verify the installed wheel in an isolated child without compiler/decoder overrides."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
directory=Path(__file__).resolve().parent
phase=sys.argv[1] if len(sys.argv)>1 else 'installed'
spec=importlib.util.spec_from_file_location('storage_fixture',root/'tests/unit/test_source_bf16.py')
fixture=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
before,_,_=fixture.fixture_model(directory/(phase+'-model'))
from tokenizers import Tokenizer, models, pre_tokenizers
t=Tokenizer(models.WordLevel({'[UNK]':0,**{f'tok{i}':i for i in range(1,101)}},unk_token='[UNK]'))
t.pre_tokenizer=pre_tokenizers.Whitespace()
t.save(str(before.parent/'tokenizer.json'))
installed=directory/phase
environment={k:v for k,v in os.environ.items() if k not in ('LEAF_DECODER_BIN','LEAF_EMBED_BIN','CXX','PYTHONPATH','PYTHONHOME')
             and not k.startswith('LEAF_EXPERIMENTAL_') and k!='LEAF_DECODER_PROFILE'}
environment.update(PATH=str(Path(os.environ['SystemRoot'])/'System32'),
                   LEAF_CACHE_DIR=str(directory/(phase+'-cache')),HF_HUB_OFFLINE='1')
code="""import sys
from pathlib import Path
sys.path.insert(0,sys.argv.pop(1))
import leaf.cli
assert Path(leaf.cli.__file__).is_relative_to(Path(sys.path[0]))
sys.argv[0]='leaf'
leaf.cli.main()
assert not any(m in sys.modules for m in ('torch','transformers'))
"""
records=[]
for arguments in [['doctor'],['run',str(before.parent),'--offline','--bits','32',
    '--source-bf16-head','--raw','--prompt','tok5 tok6 tok7','--max-tokens','2',
    '--metrics',str(directory/(phase+'-metrics.json'))]]:
    p=subprocess.run([sys.executable,'-I','-c',code,str(installed),*arguments],env=environment,
                     cwd=directory,check=True,capture_output=True,text=True,timeout=60)
    records.append(dict(arguments=arguments,stdout=p.stdout,stderr=p.stderr))
m=json.loads((directory/(phase+'-metrics.json')).read_text())
assert m['source_bf16_tensors']==2 and m['source_bf16_linear_calls']>0
assert m['weight_bits']==m['activation_bits']==32
r=dict(passed=True,compiler_path_removed=True,decoder_override_removed=True,
       isolated_installed_import=True,records=records,metrics=m)
(directory/(phase+'-check.json')).write_text(json.dumps(r,indent=2)+'\n')
print('PASS: installed doctor and offline source-BF16 generation; no compiler or binary setup')
