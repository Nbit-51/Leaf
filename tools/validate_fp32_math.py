"""Rebind native accurate FP32 to the complete retained trained qualification.

Developer migration audit: requires the October 19/22 local artifacts and saved
oracles. It does not download models, rerun PyTorch, or qualify performance.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.decoder_validation import run_native
from tools.validate_decoder import digest,utc_now,write_record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--full-fast-reference',type=Path)
    parser.add_argument('--full-fast-quality',type=Path)
    args=parser.parse_args()
    if (args.full_fast_reference is None)!=(args.full_fast_quality is None):
        parser.error('Provide the fast reference and its bound quality report together')
    output=args.directory.resolve()
    if not output.is_relative_to((ROOT/'build').resolve()) or output==(ROOT/'build').resolve():
        parser.error('Use a fresh directory inside build/')
    output.mkdir(exist_ok=False,parents=True)
    scratch=output/'tmp';scratch.mkdir()
    os.environ['TEMP']=os.environ['TMP']=str(scratch)
    tempfile.tempdir=str(scratch)
    base=ROOT/'build/worker-placement-20261022'
    reference=base/'bounded-spin/accurate/leaf_decoder.exe'
    ordinary=Path((base/'final-runtime.txt').read_text().strip())
    qualified=base/'spin-model/quality.json'
    prepared=ROOT/'build/source-bf16-20261020/prepared.json'
    source_quality=ROOT/'build/source-bf16-20261020/quality.json'
    binary=args.binary.resolve()
    record=dict(complete=False,passed=False,timing_qualified=False,utc=utc_now(),
        scope=__doc__,binary=str(binary),models={},regressions=[])
    report=output/'quality.json'
    write_record(report,record)
    try:
        proof=json.loads(qualified.read_text());storage_proof=json.loads(source_quality.read_text())
        models=json.loads(prepared.read_text())
        if not (proof['complete'] and proof['passed'] and storage_proof['complete'] and storage_proof['passed'] and models['complete']):
            raise ValueError('Incomplete trained reference qualification')
        if proof['hashes'][str(reference)]!=digest(reference) or proof['hashes'][str(prepared)]!=digest(prepared):
            raise ValueError('Qualified reference identity changed')
        fast_reference=args.full_fast_reference.resolve() if args.full_fast_reference else ordinary
        fast_proof=None
        paths=[binary,reference,ordinary,qualified,prepared,source_quality,Path(__file__),ROOT/'tools/decoder_validation.py']
        if args.full_fast_quality:
            fast_proof=json.loads(args.full_fast_quality.read_text())
            if not (fast_proof.get('complete') is True and fast_proof.get('passed') is True and
                    fast_proof.get('hashes',{}).get(str(fast_reference))==digest(fast_reference)):
                raise ValueError('Require complete bound fast-reference qualification')
            paths.extend([fast_reference,args.full_fast_quality.resolve()])
            record['full_fast_reference']=str(fast_reference)
            record['fast_scope']='Exact regression to previous ordinary Leaf; does not resolve its existing PyTorch tolerance misses'
        record['hashes']={str(p):digest(p) for p in paths}

        def compare(artifact,ids,*,policy='accurate',**kwargs):
            prior=reference if policy=='accurate' else fast_reference
            a,_=run_native(prior,artifact,[ids],**kwargs)
            b,metrics=run_native(binary,artifact,[ids],fp32_math=policy,**kwargs)
            exact=bool(np.array_equal(a.view('<u4'),b.view('<u4')))
            return dict(exact=exact,logits=int(a.size),fp32_math=metrics.get('fp32_math'))

        for name,row in models['models'].items():
            artifact=Path(row['before'])
            if digest(artifact)!=row['before_sha256']:raise ValueError('FP32 artifact changed')
            if fast_proof and fast_proof['hashes'].get(str(artifact))!=digest(artifact):
                raise ValueError('Fast-reference artifact differs from its qualified proof')
            # Qualification also proved source-preserving storage exact. Audit
            # actual FP32 and source-BF16 inputs rather than assuming interchange.
            stored=Path(row['after'])
            if digest(stored)!=row['after_sha256']:raise ValueError('Source-storage artifact changed')
            work=ROOT/'build/linear-accumulation-20261019-pipeline'/('full-quality-'+name)
            tokens,quality=work/'tokens.json',work/'quality.json'
            gold=json.loads(quality.read_text());sequences=json.loads(tokens.read_text())['sequences']
            if not (gold['complete'] and gold['passed'] and len(sequences)==8 and
                    len(gold['generations'])==6 and gold['tokens_sha256']==digest(tokens)):
                raise ValueError('Require all trained blocks and reference continuations')
            record['hashes'].update({str(p):digest(p) for p in (artifact,stored,tokens,quality)})
            cases=record['models'][name]=dict(blocks=[],generations=[],controls=[],fast_regression=[])
            for index,ids in enumerate(sequences):
                item=compare(artifact,ids)
                cases['blocks'].append(dict(index=index,**item));write_record(report,record)
                print(name,'block',index+1,'exact',item['exact'],flush=True)
                if not item['exact']:raise ValueError('Full trained logits differ from the qualified accurate reference')
            if fast_proof:
                cases['full_fast_blocks']=[]
                for index,ids in enumerate(sequences):
                    item=compare(artifact,ids,policy='fast')
                    cases['full_fast_blocks'].append(dict(index=index,**item));write_record(report,record)
                    print(name,'fast regression block',index+1,'exact',item['exact'],flush=True)
                    if not item['exact']:raise ValueError('Full ordinary FP32 logits changed')
                cases['fast_generations']=[]
                for case in gold['generations']:
                    _,a=run_native(fast_reference,artifact,[case['input_ids']],mode='generate',generate=16)
                    _,b=run_native(binary,artifact,[case['input_ids']],mode='generate',generate=16)
                    exact=a['generated_tokens']==b['generated_tokens']
                    cases['fast_generations'].append(dict(prompt=case['prompt'],exact=exact))
                    if not exact:raise ValueError('Ordinary generation changed')
                write_record(report,record)
            for case in gold['generations']:
                _,metrics=run_native(binary,artifact,[case['input_ids']],mode='generate',generate=16,fp32_math='accurate')
                exact=metrics['generated_tokens']==case['reference_tokens']
                cases['generations'].append(dict(prompt=case['prompt'],exact=exact))
                write_record(report,record)
                if not exact:raise ValueError('Saved reference continuation differs')
            for threads,scalar,mode,chunk in [(1,True,'verify',1),(2,False,'chunked',9),(4,False,'verify',1),(1,False,'chunked',1)]:
                item=compare(artifact,sequences[0][:33],threads=threads,scalar=scalar,mode=mode,chunk=chunk)
                cases['controls'].append(dict(threads=threads,scalar=scalar,mode=mode,chunk=chunk,**item))
                write_record(report,record)
                if not item['exact']:raise ValueError('Accurate scalar/cache/thread control differs')
            item=compare(stored,sequences[0][:33])
            cases['controls'].append(dict(source_bf16=True,**item))
            if not item['exact']:raise ValueError('Source-BF16 accurate control differs')
            for scalar,mode,chunk in [(False,'verify',1),(True,'chunked',9)]:
                item=compare(artifact,sequences[0][:33],policy='fast',scalar=scalar,mode=mode,chunk=chunk)
                cases['fast_regression'].append(dict(scalar=scalar,mode=mode,**item))
                if not item['exact']:raise ValueError('Default FP32 logits changed')
            write_record(report,record)
            print(name,'all continuations, cache/scalar/thread/storage controls and fast defaults pass',flush=True)

        for name,folder,filename in [('gpt2','gpt2_decoder','decoder-32.leaf'),
                                     ('tinyllama-w8a8','tinyllama_windows_fair','decoder-8-smooth.leaf')]:
            artifact=ROOT/'build'/folder/filename;tokens=artifact.parent/'tokens.json'
            ids=json.loads(tokens.read_text())['benchmark_ids'][:17]
            record['hashes'].update({str(p):digest(p) for p in (artifact,tokens)})
            for scalar in (False,True):
                kw=dict(activation_bits=8) if name=='tinyllama-w8a8' else {}
                item=compare(artifact,ids,policy='fast',scalar=scalar,**kw)
                record['regressions'].append(dict(model=name,scalar=scalar,**item))
                if not item['exact']:raise ValueError('Earlier trained default regression')
            write_record(report,record)
        if any(digest(Path(p))!=h for p,h in record['hashes'].items()):raise ValueError('Qualification inputs changed')
        record.update(complete=True,passed=True)
        print('PASS complete trained accurate-policy rebinding; performance remains unqualified',flush=True)
    except BaseException as error:
        record['error']=repr(error);raise
    finally:
        write_record(report,record)


if __name__=='__main__':main()
