"""Locate Pythia FP32 divergence on a retained quality fixture; no timing claims."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers.models.gpt_neox import modeling_gpt_neox

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leaf import cli
from leaf.evaluation import write_record
from tools.decoder_validation import run_native
from tools.validate_decoder import digest


def compare(actual, expected):
    actual, expected = actual.reshape(-1), expected.reshape(-1)
    if (actual.shape != expected.shape or not actual.size
            or not np.isfinite(actual).all() or not np.isfinite(expected).all()):
        raise ValueError("Invalid trace tensor")
    delta = np.abs(actual.astype(np.float64) - expected)
    tolerance = 2e-4 + 2e-4 * np.abs(expected.astype(np.float64))
    worst = int(np.argmax(delta / tolerance))
    return dict(size=int(actual.size), max_abs_error=float(delta.max()),
                tolerance_violations=int(np.sum(delta > tolerance)),
                max_tolerance_ratio=float((delta / tolerance).max()), worst_flat_index=worst,
                reference_at_worst=float(expected[worst]), native_at_worst=float(actual[worst]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quality-record', type=Path, required=True)
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--block', type=int, required=True, help='zero-based saved block index')
    p.add_argument('--executable', type=Path, required=True, help='LEAF_DIAGNOSTIC_TRACE build')
    p.add_argument('--baseline', type=Path, required=True, help='previous normal runtime')
    p.add_argument('--output', type=Path, required=True, help='fresh directory')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.quality_record.read_text())
    if digest(args.tokens) != source['tokens_sha256']:
        raise ValueError('Token fixture identity mismatch')
    snapshot = Path(source['snapshot'])
    for name, checksum in source['source_sha256'].items():
        if digest(snapshot / name) != checksum:
            raise ValueError('Model source identity mismatch')
    rows = json.loads(args.tokens.read_text())['sequences']
    if not 0 <= args.block < len(rows):
        raise ValueError('Block index outside the saved fixture')
    row = rows[args.block]
    artifact = cli.prepare(snapshot, cli.model_directory(snapshot, cli.cache_root()), 32)
    if digest(artifact) != source['artifact_sha256']:
        raise ValueError('Native artifact identity mismatch')
    record = dict(complete=False, block=args.block, quality_record_sha256=digest(args.quality_record),
                  executable_sha256=digest(args.executable), baseline_sha256=digest(args.baseline),
                  torch=torch.__version__, transformers=transformers.__version__, tensors={})
    # Persist the identity before inference; an interrupted trace is never complete.
    write_record(args.output / 'trace.json', record)
    run_trace(args, record, row, artifact, snapshot)


def run_trace(args, record, row, artifact, snapshot):
    try:
        capture_trace(args, record, row, artifact, snapshot)
        record['complete'] = True
    except BaseException as error:
        record['error'] = str(error) or type(error).__name__
        raise
    finally:
        write_record(args.output / 'trace.json', record)


def capture_trace(args, record, row, artifact, snapshot):
    native_dir = args.output / 'native'
    native_dir.mkdir()
    prior = os.environ.get('LEAF_TRACE_DIRECTORY')
    os.environ['LEAF_TRACE_DIRECTORY'] = str(native_dir.resolve())
    try:
        native, _ = run_native(args.executable, artifact, [row])
    finally:
        if prior is None:
            os.environ.pop('LEAF_TRACE_DIRECTORY')
        else:
            os.environ['LEAF_TRACE_DIRECTORY'] = prior
    baseline, _ = run_native(args.baseline, artifact, [row])
    record['instrumentation_bit_exact'] = native.shape == baseline.shape and native.tobytes() == baseline.tobytes()
    if not record['instrumentation_bit_exact']:
        raise ValueError('Instrumentation changed baseline outputs')
    torch.set_num_threads(1)
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                                                dtype=torch.float32, attn_implementation='eager').eval()
    if model.config.model_type != 'gpt_neox':
        raise ValueError('This reference adapter requires GPT-NeoX/Pythia')
    tensors = {}
    hooks = []
    def capture(name):
        def save(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            tensors[name] = value.detach().numpy().copy()
        return save
    hooks.append(model.gpt_neox.embed_in.register_forward_hook(capture('embedding')))
    for i, layer in enumerate(model.gpt_neox.layers):
        prefix = f'model.layers.{i}.'
        for name, module in (('input_norm', layer.input_layernorm), ('post_norm', layer.post_attention_layernorm),
                             ('attn_projected', layer.attention.dense), ('up', layer.mlp.dense_h_to_4h),
                             ('activation', layer.mlp.act), ('down', layer.mlp.dense_4h_to_h), ('hidden', layer)):
            hooks.append(module.register_forward_hook(capture(prefix + name)))
        def save_qkv(module, inputs, output, prefix=prefix):
            value = output.detach().numpy().reshape(len(row), model.config.num_attention_heads, 3, -1)
            for index, name in enumerate(('q', 'k', 'v')):
                tensors[prefix + name] = value[:, :, index, :].copy()
        hooks.append(layer.attention.query_key_value.register_forward_hook(save_qkv))
        def save_attention(module, inputs, prefix=prefix):
            tensors[prefix + 'attention'] = inputs[0].detach().numpy().copy()
        hooks.append(layer.attention.dense.register_forward_pre_hook(save_attention))
    hooks.append(model.gpt_neox.final_layer_norm.register_forward_hook(capture('final_norm')))
    original_attention = modeling_gpt_neox.eager_attention_forward
    def trace_attention(module, query, key, value, *args, **kwargs):
        prefix = f'model.layers.{module.layer_idx}.'
        tensors[prefix + 'q_rotary'] = query.detach().numpy().copy()
        tensors[prefix + 'k_rotary'] = key.detach().numpy().copy()
        return original_attention(module, query, key, value, *args, **kwargs)
    modeling_gpt_neox.eager_attention_forward = trace_attention
    try:
        with torch.inference_mode():
            tensors['logits'] = model(torch.tensor([row]), use_cache=False).logits.numpy().copy()
    finally:
        modeling_gpt_neox.eager_attention_forward = original_attention
        for hook in hooks:
            hook.remove()
    for name, expected in tensors.items():
        actual = np.fromfile(native_dir / (name + '.f32'), dtype='<f4')
        np.save(args.output / (name + '.npy'), expected)
        record['tensors'][name] = compare(actual, expected)
        print(name, record['tensors'][name]['max_abs_error'], record['tensors'][name]['tolerance_violations'], flush=True)


if __name__ == '__main__':
    main()
