"""Paired logit/cache regression for the final-layer FP32 row-selection experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from transformers import AutoModelForCausalLM, MistralConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leaf.evaluation import write_record
from tools.export_decoder import export_decoder
from tools.validate_decoder import digest, utc_now
from tools.verify_decoder_architectures import configurations


def reference_logits(model, lengths, context):
    """Full-prefix oracle, independent of the native row-selection/cache path."""
    vocab = model.config.vocab_size
    outputs = []
    with torch.inference_mode():
        for length in lengths:
            prefix = [(i * 17 + 4) % vocab for i in range(length)]
            history = []
            def append(ids, all_logits=False):
                history.extend(ids)
                values = model(torch.tensor([history]), use_cache=False).logits[0]
                outputs.append(values[-len(ids):].numpy().copy() if all_logits else values[-1:].numpy().copy())
            append(prefix)
            append([7 % vocab])
            if len(history) + 36 <= context:
                append([(i * 11 + 3) % vocab for i in range(32)])
                append([3, 4, 5], True)
            history = []
            append(prefix, True)
    return np.concatenate(outputs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, help='optional paired native driver; otherwise FP32 reference checks only')
    p.add_argument('--after', type=Path, required=True, help='same driver linked to candidate decoder')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh output directory')
    args.output.mkdir(parents=True)
    report = dict(complete=False, passed=False, measured_at_utc=utc_now(),
                  before_sha256=digest(args.before) if args.before else None, after_sha256=digest(args.after),
                  scope=__doc__, timing_qualified=False, checks=[])
    path = args.output / 'validation.json'
    write_record(path, report)
    torch.set_num_threads(1)
    lengths = ('1,7,8,15,16,17,31,32,33,47,48,49,63,64,65,95,127'
               if args.before else '31,32,33,63,64,65')
    configs = configurations()
    configs.append(MistralConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, sliding_window=8, bos_token_id=1, eos_token_id=2))
    try:
        with tempfile.TemporaryDirectory(prefix='leaf_final_rows_') as temporary:
            root = Path(temporary)
            for index, config in enumerate(configs):
                if config.model_type == 'gpt2':
                    config.n_positions = 256
                else:
                    config.max_position_embeddings = 256
                config._attn_implementation = 'eager'
                torch.manual_seed(901 + index)
                model = AutoModelForCausalLM.from_config(config).float().eval()
                expected = reference_logits(model, list(map(int, lengths.split(','))), 256)
                snapshot = root / str(index)
                model.save_pretrained(snapshot, safe_serialization=True)
                del model
                for bits in ((32, 8, 4) if args.before else (32,)):
                    artifact = snapshot / f'decoder-{bits}.leaf'
                    export_decoder(snapshot, artifact, bits)
                    policies = [(1, False, 32), (2, False, 32), (1, True, 32)]
                    if bits != 32:
                        policies.append((1, False, 8))
                    for threads, scalar, activation in policies:
                        outputs = []
                        binaries = [('before', args.before)] if args.before else []
                        binaries.append(('after', args.after))
                        max_error = None
                        for label, exe in binaries:
                            target = root / (label + '.f32')
                            command = [str(exe.resolve()), str(artifact), str(target), str(threads),
                                       str(int(scalar)), str(activation), '256', lengths, '0']
                            result = subprocess.run(command, capture_output=True, text=True, check=True)
                            outputs.append((target.read_bytes(), json.loads(result.stdout)))
                            if bits == 32:
                                actual = np.frombuffer(outputs[-1][0], dtype='<f4').reshape(expected.shape)
                                np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-4)
                                max_error = float(np.max(np.abs(actual - expected)))
                        if args.before and outputs[0] != outputs[1]:
                            raise AssertionError(f'Logit bytes or cache checks differ: {index}/{bits}/{threads}/{scalar}/{activation}')
                        report['checks'].append(dict(config_index=index, architecture=config.model_type,
                            bits=bits, threads=threads, scalar=scalar, activation_bits=activation,
                            calls=outputs[-1][1]['calls'], bit_exact=True if args.before else None,
                            reference_max_abs_error=max_error))
                        write_record(path, report)
                print(index, config.model_type, 'passed', flush=True)
        report.update(complete=True, passed=True)
    except BaseException as error:
        report['error'] = str(error) or type(error).__name__
        raise
    finally:
        write_record(path, report)


if __name__ == '__main__':
    main()
