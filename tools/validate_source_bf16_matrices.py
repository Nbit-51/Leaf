"""Exact trained regression for original BF16 matrix storage with FP32 arithmetic.

Requires locally qualified FP32 artifacts. Does not rerun PyTorch, loosen an
accuracy tolerance, download a model, or qualify model/competitor latency.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.decoder_validation import run_native
from tools.export_decoder import DESCRIPTOR, HEADER, export_decoder
from tools.validate_decoder import digest, utc_now, write_record


def metadata(mapped):
    """Decode actual artifact bytes independently of exporter tensor selection."""
    header = HEADER.unpack_from(mapped)
    magic, version, count, base = header[:4]
    if version not in (2, 3, 4, 5, 6) or magic != f'LEAFDC0{version}'.encode():
        raise ValueError('Unsupported artifact header')
    position, eos = HEADER.size, ([] if header[11] == 0xffffffff else [header[11]])
    if version >= 4:
        eos_count = struct.unpack_from('<I', mapped, position)[0]
        if eos_count > 1024:
            raise ValueError('Invalid complete EOS metadata')
        position += 4
        eos = list(struct.unpack_from(f'<{eos_count}I', mapped, position))
        position += 4*eos_count
    descriptors = {}
    for _ in range(count):
        length = struct.unpack_from('<I', mapped, position)[0]
        position += 4
        if not length or position+length+DESCRIPTOR.size > base:
            raise ValueError('Descriptor exceeds metadata extent')
        name = mapped[position:position+length].decode('utf-8')
        position += length
        row = DESCRIPTOR.unpack_from(mapped, position)
        position += DESCRIPTOR.size
        if name in descriptors:
            raise ValueError('Duplicate tensor descriptor')
        descriptors[name] = row
    if position > base or base > len(mapped) or base % 64:
        raise ValueError('Invalid artifact data base')
    return header, eos, descriptors


def audit_payloads(before: Path, after: Path, exported: dict) -> dict:
    """Bounded mmap/chunk audit of every stored value, including FP32 fallback."""
    checked, compact = 0, []
    with before.open('rb') as a, after.open('rb') as b:
        with mmap.mmap(a.fileno(), 0, access=mmap.ACCESS_READ) as old:
            with mmap.mmap(b.fileno(), 0, access=mmap.ACCESS_READ) as new:
                ah, aeos, ad = metadata(old)
                bh, beos, bd = metadata(new)
                if (ah[2] != bh[2] or old[24:HEADER.size] != new[24:HEADER.size] or
                        aeos != beos or list(ad) != list(bd) or bh[1] != 6):
                    raise ValueError('Configuration, EOS policy, descriptor set/order or expected v6 differs')
                for name, source in ad.items():
                    target = bd[name]
                    if (source[:3] != target[:3] or source[0] != 2 or source[3] != 32 or
                            target[3] not in (16, 32) or source[4] or target[4] or source[8] or target[8]):
                        raise ValueError(f'Unexpected unquantized descriptor: {name}')
                    values = source[1]*source[2]
                    for mapped, header, descriptor in ((old, ah, source), (new, bh, target)):
                        start, size = header[3]+descriptor[5], descriptor[6]
                        if (not values or size != values*descriptor[3]//8 or descriptor[5] % 64 or
                                start < header[3] or start+size > len(mapped)):
                            raise ValueError(f'Invalid tensor payload range: {name}')
                    if target[3] == 16:
                        if target[1] <= 1 or not name.endswith('.weight'):
                            raise ValueError('Compact storage appeared outside a matrix weight')
                        compact.append(name)
                    for first in range(0, values, 1_048_576):
                        count = min(1_048_576, values-first)
                        start = ah[3]+source[5]+4*first
                        expected = np.frombuffer(old[start:start+4*count], dtype='<u4')
                        width = target[3]//8
                        start = bh[3]+target[5]+width*first
                        observed = np.frombuffer(new[start:start+width*count], dtype='<u2' if width == 2 else '<u4')
                        if width == 2:
                            observed = observed.astype('<u4') << 16
                        if not np.array_equal(expected, observed):
                            raise ValueError(f'Tensor bits changed: {name}, chunk {first}')
                        checked += count
                if sorted(compact) != sorted(exported.get('source_bf16_tensors', [])) or not compact:
                    raise ValueError('Export report does not match actual compact tensor descriptors')
                return dict(exact=True, values=checked, tensors=len(ad), source_bf16_tensors=len(compact),
                    bf16_names=compact, vocabulary=ah[9], before_bytes=len(old), after_bytes=len(new),
                    bytes_ratio=len(new)/len(old), config_and_eos_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'reference', 'reference-quality', 'models', 'directory'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(ROOT/'build') or directory == ROOT/'build':
        parser.error('Use a fresh directory inside build/')
    directory.mkdir(parents=True, exist_ok=False)
    scratch = directory/'tmp'
    scratch.mkdir()
    os.environ['TEMP'] = os.environ['TMP'] = str(scratch)
    tempfile.tempdir = str(scratch)
    binary, reference = args.binary.resolve(), args.reference.resolve()
    proof_path, models_path = args.reference_quality.resolve(), args.models.resolve()
    record = dict(format='leaf-source-bf16-matrices-quality-v1', utc=utc_now(), complete=False,
        passed=False, timing_qualified=False, acceptance_timing=False, scope=__doc__, models={}, hashes={})
    report = directory/'quality.json'
    write_record(report, record)
    try:
        if any(value for key, value in os.environ.items() if key.startswith('LEAF_EXPERIMENTAL_') or
               key in ('LEAF_DECODER_PROFILE', 'LEAF_DISABLE_VNNI', 'LEAF_DIAGNOSTIC_START_GATE')):
            raise ValueError('Clear inherited experimental/profiling policies')
        proof = json.loads(proof_path.read_text())
        if not (proof.get('complete') is True and proof.get('passed') is True and
                proof.get('hashes', {}).get(str(reference)) == digest(reference)):
            raise ValueError('Require a complete passing qualification bound to the reference runtime')
        models = json.loads(models_path.read_text())['models']
        if not isinstance(models, dict) or not models:
            raise ValueError('Require a nonempty model manifest')
        bound = [binary, reference, proof_path, models_path, Path(__file__), ROOT/'tools/export_decoder.py',
                 ROOT/'tools/decoder_plan.py', ROOT/'tools/decoder_validation.py']
        record['hashes'] = {str(p.resolve()): digest(p) for p in bound}
        for name, row in models.items():
            if not isinstance(name, str) or not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):
                raise ValueError('Model name must be a safe directory component')
            snapshot, original, tokens = (Path(row[k]).resolve() for k in ('snapshot', 'fp32_artifact', 'tokens'))
            if proof['hashes'].get(str(original)) != digest(original):
                raise ValueError('FP32 artifact differs from its qualified reference proof')
            if proof['hashes'].get(str(tokens)) != digest(tokens):
                raise ValueError('Trained token blocks differ from their qualified reference proof')
            sequences = json.loads(tokens.read_text())['sequences']
            if (not isinstance(sequences, list) or len(sequences) != 8 or
                    any(not isinstance(ids, list) or len(ids) != 128 or
                        any(type(x) is not int or not 0 <= x <= 0xffffffff for x in ids) for ids in sequences)):
                raise ValueError('Require eight complete 128-token trained blocks')
            source_paths = [snapshot/'config.json', *sorted(snapshot.glob('*.safetensors'))]
            source_paths += [snapshot/p for p in ('generation_config.json', 'leaf_plan.json', 'model.safetensors.index.json')
                             if (snapshot/p).is_file()]
            if len(source_paths) < 2:
                raise ValueError('Model snapshot has no source safetensors')
            record['hashes'].update({str(p): digest(p) for p in [original, tokens, *source_paths]})
            work = directory/name
            work.mkdir()
            stored = work/'decoder-32-source-bf16.leaf'
            exported = export_decoder(snapshot, stored, bits=32, source_bf16=True)
            export_path = work/'export.json'
            write_record(export_path, exported)
            payload = audit_payloads(original, stored, exported)
            record['hashes'].update({str(p): digest(p) for p in (stored, export_path)})
            result = record['models'][name] = dict(before=str(original), after=str(stored), payload=payload,
                                                   blocks=[], generations=[], controls=[])
            write_record(report, record)

            def compare(ids, policy, **options):
                a, am = run_native(reference, original, [ids], fp32_math=policy, activation_bits=32, **options)
                b, bm = run_native(binary, stored, [ids], fp32_math=policy, activation_bits=32, **options)
                if any(m.get('fp32_math') != policy or m.get('activation_bits') != 32 for m in (am, bm)):
                    raise ValueError('Native math/activation policy differs')
                if (am.get('source_bf16_tensors') != 0 or
                        bm.get('source_bf16_tensors') != payload['source_bf16_tensors'] or
                        bm.get('source_bf16_linear_calls', 0) <= 0):
                    raise ValueError('Native compact-matrix path was not executed')
                if options.get('mode') == 'generate':
                    generated = am.get('generated_tokens')
                    exact = (isinstance(generated, list) and 0 < len(generated) <= options['generate'] and
                             all(type(x) is int and 0 <= x < payload['vocabulary'] for x in generated) and
                             generated == bm.get('generated_tokens'))
                    return dict(exact=exact, fp32_math=policy, generated_tokens=generated)
                exact = (a.size == b.size == len(ids)*payload['vocabulary'] and np.isfinite(a).all() and
                         np.isfinite(b).all() and np.array_equal(a.view('<u4'), b.view('<u4')))
                return dict(exact=bool(exact), fp32_math=policy, logits=int(a.size))

            for policy in ('fast', 'accurate'):
                for index, ids in enumerate(sequences):
                    item = dict(index=index, **compare(ids, policy))
                    result['blocks'].append(item)
                    write_record(report, record)
                    print(name, policy, 'block', index+1, 'exact', item['exact'], flush=True)
                    if not item['exact']:
                        raise ValueError('Full trained FP32/source-BF16 logits differ')
                prefixes = [sequences[0][:63], sequences[1][:7], sequences[2][:17], sequences[3][:33]]
                for index, ids in enumerate(prefixes):
                    item = dict(index=index, input_ids=ids, **compare(ids, policy, mode='generate', generate=8))
                    result['generations'].append(item)
                    write_record(report, record)
                    if not item['exact']:
                        raise ValueError('Greedy continuation differs')
                for scalar, threads, mode, chunk in ((False, 1, 'chunked', 1), (True, 1, 'chunked', 9),
                                                       (False, 2, 'chunked', 9), (True, 1, 'verify', 1)):
                    item = dict(scalar=scalar, threads=threads, mode=mode, chunk=chunk,
                                **compare(sequences[0][:33], policy, scalar=scalar, threads=threads, mode=mode, chunk=chunk))
                    result['controls'].append(item)
                    write_record(report, record)
                    if not item['exact']:
                        raise ValueError('Cache/scalar/two-worker control differs')
            print(name, 'complete exact storage/math regression', flush=True)
        if any(digest(Path(p)) != h for p, h in record['hashes'].items()):
            raise ValueError('Qualification input changed')
        record.update(complete=True, passed=True)
        print('PASS original BF16 matrix storage; latency remains unqualified', flush=True)
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        write_record(report, record)


if __name__ == '__main__':
    main()
