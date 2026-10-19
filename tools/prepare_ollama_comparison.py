"""Prepare, audit and import a cached Llama/Qwen2 FP32 comparison checkpoint."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.validate_decoder import digest
from tools.ollama_validation import api, local_server


def audit(snapshot, gguf_path, source, output):
    sys.path.insert(0, str(source / 'gguf-py'))
    import gguf
    import numpy as np
    from safetensors import safe_open

    config = json.loads((snapshot / 'config.json').read_text())
    kind = config.get('model_type')
    if kind not in ('llama', 'qwen2'):
        raise ValueError('Audit supports Llama and Qwen2 semantics only; no silent architecture guess')
    arch = gguf.MODEL_ARCH.LLAMA if kind == 'llama' else gguf.MODEL_ARCH.QWEN2
    reader = gguf.GGUFReader(gguf_path)
    if reader.get_field('general.architecture').contents() != kind:
        raise ValueError('GGUF architecture differs from source config')
    tensors = {t.name: t for t in reader.tensors}
    remaining = set(tensors)
    mapping = gguf.get_tensor_name_map(arch, config['num_hidden_layers'])
    checks = []
    weight_file = snapshot / 'model.safetensors'
    hashes = {str(p): digest(p) for p in (weight_file, snapshot/'config.json', gguf_path, Path(__file__),
        source/'convert_hf_to_gguf.py', source/'conversion/llama.py', source/'conversion/qwen.py')}
    with safe_open(weight_file, framework='pt', device='cpu') as weights:
        for name in weights.keys():
            target = mapping.get_name(name, try_suffixes=('.weight', '.bias'))
            if target is None:
                raise ValueError('Unmapped source parameter: ' + name)
            value = weights.get_tensor(name).float().numpy()
            permuted = kind == 'llama' and name.endswith(('self_attn.q_proj.weight', 'self_attn.k_proj.weight'))
            if permuted:
                heads = config['num_attention_heads'] if name.endswith('q_proj.weight') else config['num_key_value_heads']
                value = value.reshape(heads, 2, value.shape[0]//heads//2, *value.shape[1:]).swapaxes(1, 2).reshape(value.shape)
            alias = False
            if target not in tensors and name == 'lm_head.weight' and config.get('tie_word_embeddings'):
                target, alias = 'token_embd.weight', True
            tensor = tensors[target]
            if (tensor.tensor_type != gguf.GGMLQuantizationType.F32 or tensor.data.shape != value.shape or
                    not np.array_equal(tensor.data.view(np.uint32), value.view(np.uint32))):
                raise ValueError('FP32 parameter mismatch: ' + name)
            remaining.discard(target)
            checks.append(dict(source=name, gguf=target, exact=True, rope_permuted=permuted, tied_alias=alias))
        if remaining == {'output.weight'} and config.get('tie_word_embeddings'):
            tensor, embedding = tensors['output.weight'], tensors['token_embd.weight']
            if (tensor.tensor_type != gguf.GGMLQuantizationType.F32 or
                    not np.array_equal(tensor.data.view(np.uint32), embedding.data.view(np.uint32))):
                raise ValueError('Tied output does not equal token embedding')
            checks.append(dict(source='model.embed_tokens.weight', gguf='output.weight', exact=True, tied_alias=True))
            remaining.clear()
    if remaining:
        raise ValueError('Unexpected GGUF tensors: ' + ', '.join(sorted(remaining)))
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError('Inputs changed during parameter audit')
    record = dict(complete=True, passed=True, architecture=kind, parameters=len(checks), checks=checks,
                  scope='Every source parameter verified bit-for-bit as F32; Llama Q/K permutation and declared tied aliases only',
                  source_weight_sha256=hashes[str(weight_file)], config_sha256=hashes[str(snapshot/'config.json')],
                  gguf_sha256=hashes[str(gguf_path)], hashes=hashes)
    output.write_text(json.dumps(record, indent=2) + '\n')
    print('PASS', len(checks), 'parameters, F32', kind, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('convert', 'audit', 'import', 'export-leaf'))
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT/'build/competitors/llama-b11540/source')
    parser.add_argument('--ollama-executable', type=Path, default=ROOT/'build/ollama-20261015/runtime/ollama.exe')
    parser.add_argument('--ollama-models', type=Path, default=ROOT/'build/ollama-family-20261019/models')
    parser.add_argument('--model-name')
    parser.add_argument('--reference-artifact', type=Path,
                        help='require an exact match to an already quality-checked Leaf artifact')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot, directory = args.snapshot.resolve(), args.output.resolve()
    gguf_path, audit_path = directory/'model-f32.gguf', directory/'gguf-audit.json'
    if args.stage == 'convert':
        if gguf_path.exists():
            raise ValueError('Refuse to overwrite an existing GGUF; audit it or choose a fresh output')
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        with (directory/'conversion.log').open('x') as log:
            subprocess.run([sys.executable, str(args.source/'convert_hf_to_gguf.py'), str(snapshot),
                            '--outfile', str(gguf_path), '--outtype', 'f32'], check=True, env=env,
                           stdout=log, stderr=subprocess.STDOUT)
        print('Converted local snapshot:', gguf_path, flush=True)
    elif args.stage == 'audit':
        if audit_path.exists():
            raise ValueError('Refuse to overwrite retained audit')
        audit(snapshot, gguf_path, args.source.resolve(), audit_path)
    elif args.stage == 'export-leaf':
        from types import SimpleNamespace
        from tools.export_decoder import export_decoder
        from tools.validate_decoder import decoder_plan_identity, export_request, write_export_provenance
        artifact = directory/'decoder-32.leaf'
        if artifact.exists():
            raise ValueError('Refuse to overwrite an existing Leaf artifact')
        bound = {'model_config_sha256': digest(snapshot/'config.json'),
                 'source_weight_sha256': {p.name: digest(p) for p in sorted(snapshot.glob('*.safetensors'))}}
        plan, identity = decoder_plan_identity(SimpleNamespace(model=snapshot))
        export_decoder(snapshot, artifact, bits=32)
        if (digest(snapshot/'config.json') != bound['model_config_sha256'] or
                {p.name: digest(p) for p in sorted(snapshot.glob('*.safetensors'))} != bound['source_weight_sha256']):
            raise ValueError('Source changed during Leaf export; no provenance written')
        if args.reference_artifact and digest(artifact) != digest(args.reference_artifact):
            raise ValueError('Fresh export differs from the quality-checked reference artifact')
        write_export_provenance(artifact, export_request(bound, plan, identity, 32, 32, [], 0))
        print('Exported source-bound FP32 Leaf artifact:', artifact, flush=True)
    else:
        record = json.loads(audit_path.read_text())
        if not record.get('passed') or digest(gguf_path) != record['gguf_sha256']:
            raise ValueError('Require a complete exact audit for the imported file')
        if not args.model_name:
            parser.error('--model-name is required for import')
        modelfile = directory/'Modelfile'
        modelfile.write_text('FROM ' + gguf_path.as_posix() + '\n')
        with local_server(args.ollama_executable, args.ollama_models, directory/'import-server') as (base, env, _):
            tags = api(base, '/api/tags').get('models', [])
            existing = any(row['name'].removesuffix(':latest') == args.model_name.removesuffix(':latest') for row in tags)
            if not existing:
                with (directory/'import.log').open('x') as log:
                    subprocess.run([str(args.ollama_executable.resolve()), 'create', args.model_name,
                                    '-f', str(modelfile)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            shown = api(base, '/api/show', dict(model=args.model_name))
            blob = re.search(r'(?m)^FROM .*sha256-([0-9a-f]{64})$', shown['modelfile'])
            if shown['details']['quantization_level'] != 'F32' or not blob or blob.group(1) != record['gguf_sha256']:
                raise ValueError('Imported model differs from the exact audited FP32 GGUF')
            (directory/'model-info.json').write_text(json.dumps(shown, indent=2) + '\n')
            print('Imported audited FP32 model:', args.model_name, flush=True)


if __name__ == '__main__':
    main()
