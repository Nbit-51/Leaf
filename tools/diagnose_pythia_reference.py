"""Compare retained Pythia failures with alternative arithmetic; not a release gate."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leaf import cli
from leaf.evaluation import logit_diagnostics, write_record
from tools.decoder_validation import quality, run_native
from tools.trace_pythia_logits import compare
from tools.validate_decoder import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality-record', type=Path, required=True)
    parser.add_argument('--tokens', type=Path, required=True)
    parser.add_argument('--blocks', type=int, nargs='+', required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite diagnostic evidence')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.quality_record.read_text())
    if digest(args.tokens) != source['tokens_sha256']:
        raise ValueError('Token fixture identity mismatch')
    snapshot = Path(source['snapshot'])
    for name, checksum in source['source_sha256'].items():
        if digest(snapshot / name) != checksum:
            raise ValueError('Model source identity mismatch')
    artifact = cli.prepare(snapshot, cli.model_directory(snapshot, cli.cache_root()), 32)
    if digest(artifact) != source['artifact_sha256']:
        raise ValueError('Native artifact identity mismatch')
    rows = json.loads(args.tokens.read_text())['sequences']
    if any(not 0 <= index < len(rows) for index in args.blocks):
        raise ValueError('Block index outside the saved fixture')
    record = dict(complete=False, scope=__doc__, model=source['model'],
                  quality_record_sha256=digest(args.quality_record),
                  baseline_sha256=digest(args.baseline),
                  candidate_sha256=digest(args.candidate) if args.candidate else None,
                  torch=torch.__version__, transformers=transformers.__version__, checks=[])
    def save():
        write_record(args.output, record)
    save()
    torch.set_num_threads(1)
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                                                dtype=torch.float32, attn_implementation='eager').eval()
    if model.config.model_type != 'gpt_neox':
        raise ValueError('This diagnostic requires GPT-NeoX/Pythia')
    with torch.inference_mode():
        for index in args.blocks:
            row = rows[index]
            model.float()
            model.set_attn_implementation('eager')
            ids = torch.tensor([row])
            eager = model(ids, use_cache=False).logits[0].numpy().copy()
            model.set_attn_implementation('sdpa')
            sdpa = model(ids, use_cache=False).logits[0].numpy().copy()
            model.set_attn_implementation('eager')
            model.double()
            double = model(ids, use_cache=False).logits[0].numpy().copy()
            model.float()
            baseline, _ = run_native(args.baseline, artifact, [row])
            values = {'leaf_baseline': baseline.reshape(eager.shape),
                      'torch_sdpa': sdpa, 'torch_float64': double}
            if args.candidate:
                candidate, _ = run_native(args.candidate, artifact, [row])
                values['leaf_candidate'] = candidate.reshape(eager.shape)
            item = dict(block=index, comparisons={})
            for name, actual in values.items():
                item['comparisons'][name] = dict(vs_eager=compare(actual, eager),
                                                 quality_vs_eager=quality(actual, eager, [row]),
                                                 diagnostics_vs_eager=logit_diagnostics(actual, eager, [row]),
                                                 vs_float64=compare(actual, double))
            item['comparisons']['torch_eager'] = dict(vs_float64=compare(eager, double))
            record['checks'].append(item)
            save()
            print(json.dumps(item), flush=True)
    record['complete'] = True
    save()


if __name__ == '__main__':
    main()
