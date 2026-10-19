"""Compare a retained cross-runtime preflight with a third FP32 reference.

Diagnostic only: does not change or override the original comparison gate.
Runs only the failed prompts by default and makes no timing claim.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf.evaluation import write_record
from tools.validate_decoder import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True,
                        help='original preflight record with sibling raw quality outputs')
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output to retain earlier evidence')
    saved = json.loads(args.comparison.read_text())
    directory = args.comparison.parent
    leaf_path = directory/'leaf-quality/results.jsonl'
    ollama_path = directory/'ollama-quality.json'
    audit_path = next(Path(p) for p in saved['hashes'] if Path(p).name == 'gguf-audit.json')
    audit = json.loads(audit_path.read_text())
    snapshot = args.snapshot.resolve()
    if (digest(snapshot/'model.safetensors') != audit['source_weight_sha256'] or
            digest(snapshot/'config.json') != audit['config_sha256']):
        raise ValueError('Reference model differs from audited comparison source')
    if any(digest(Path(p)) != h for p, h in saved['hashes'].items()):
        raise ValueError('Original preflight inputs changed')
    paths = [args.comparison, leaf_path, ollama_path, audit_path, Path(__file__),
             snapshot/'config.json', snapshot/'model.safetensors', snapshot/'tokenizer.json']
    hashes = {str(p): digest(p) for p in paths}
    leaf = [json.loads(line) for line in leaf_path.read_text().splitlines()]
    ollama = json.loads(ollama_path.read_text())
    checks = saved['quality']['checks']
    if len(leaf) != len(checks) or len(ollama) != len(checks):
        raise ValueError('Retained raw quality output is incomplete')
    import torch
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(1)
    tokenizer = Tokenizer.from_file(str(snapshot/'tokenizer.json'))
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation='eager').eval()
    record = dict(complete=False, diagnostic_only=True, original_gate_unchanged=True,
        scope='Failed prompts only; cached eager FP32 reference, FP64 selected-token log-softmax',
        torch=torch.__version__, transformers=transformers.__version__, hashes=hashes, checks=[])
    write_record(args.output, record)
    with torch.inference_mode():
        for i, check in enumerate(checks):
            if check['passed']:
                continue
            ids = tokenizer.encode(check['prompt']).ids
            expected, probabilities = [], []
            past = None
            sequence = ids
            for token in leaf[i]['generated_tokens']:
                result = model(torch.tensor([sequence]), past_key_values=past, use_cache=True)
                logits = result.logits[0, -1].double()
                expected.append(int(logits.argmax()))
                probabilities.append(float(logits[token] - torch.logsumexp(logits, dim=-1)))
                past, sequence = result.past_key_values, [token]
            native = leaf[i]['selected_logprobs']
            reported = [row['logprob'] for row in ollama[i].get('logprobs', [])]
            leaf_errors = [abs(a-b) for a,b in zip(native, probabilities)]
            ollama_errors = [abs(a-b) for a,b in zip(reported, probabilities)]
            row = dict(index=i, prompt=check['prompt'], reference_greedy_tokens=expected,
                leaf_generated_tokens=leaf[i]['generated_tokens'], reference_logprobs=probabilities,
                leaf_logprobs=native, ollama_logprobs=reported,
                leaf_greedy_matches_reference=expected == leaf[i]['generated_tokens'],
                leaf_max_abs_error=max(leaf_errors),
                ollama_reported_max_abs_error=max(ollama_errors) if ollama_errors else None,
                ollama_unreported_positions=len(probabilities)-len(reported),
                ollama_note='Errors use aligned visible-token positions; unreported terminal EOS cannot be scored')
            record['checks'].append(row)
            write_record(args.output, record)
            print(i, 'Leaf/reference', row['leaf_max_abs_error'],
                  'Ollama/reference', row['ollama_reported_max_abs_error'], flush=True)
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError('Diagnosis inputs changed')
    record['complete'] = True
    write_record(args.output, record)


if __name__ == '__main__':
    main()
