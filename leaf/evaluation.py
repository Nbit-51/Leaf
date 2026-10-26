"""Bounded-block FP32 fidelity evaluation, separate from speed qualification."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

PROMPTS = (
    "The purpose of a computer is",
    "Once upon a time, a small robot found",
    "Question: What is the capital of France?\nAnswer:",
    "Two plus two equals",
    "def add(a, b):\n    return",
    "Hello! Café, naïve, and Unicode text:",
)


def write_record(path: Path, record: dict) -> None:
    """Keep the last complete checkpoint if interrupted while writing JSON."""
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staged.replace(path)


def summarize(rows: list[dict]) -> dict:
    """Pool losses by scored tokens; never average block perplexities."""
    total = sum(row["evaluated_tokens"] for row in rows)
    if not rows or total <= 0:
        raise ValueError("No scored quality targets")
    for row in rows:
        if (row["evaluated_tokens"] <= 0 or
                not all(math.isfinite(row[key]) and row[key] > 0
                        for key in ("perplexity", "pytorch_perplexity"))):
            raise ValueError("Invalid quality measurements")
    native_loss = sum(math.log(row["perplexity"]) * row["evaluated_tokens"] for row in rows) / total
    reference_loss = sum(math.log(row["pytorch_perplexity"]) * row["evaluated_tokens"] for row in rows) / total
    result = dict(evaluated_tokens=total, perplexity=math.exp(native_loss),
                pytorch_perplexity=math.exp(reference_loss),
                perplexity_ratio=math.exp(native_loss - reference_loss),
                next_token_agreement=sum(row["next_token_agreement"] * row["evaluated_tokens"] for row in rows) / total,
                leaf_next_token_accuracy=sum(row["leaf_correct_targets"] for row in rows) / total,
                pytorch_next_token_accuracy=sum(row["pytorch_correct_targets"] for row in rows) / total,
                max_abs_logit_error=max(row["max_abs_logit_error"] for row in rows),
                all_logits_close=all(row["logits_allclose"] for row in rows))
    if all("diagnostics" in row for row in rows):
        result["diagnostics"] = dict(
            logit_tolerance_violations=sum(row["diagnostics"]["logit_tolerance_violations"] for row in rows),
            max_probability_error=max(row["diagnostics"]["max_probability_error"] for row in rows),
            max_total_variation=max(row["diagnostics"]["max_total_variation"] for row in rows),
            max_kl_reference_to_leaf=max(row["diagnostics"]["max_kl_reference_to_leaf"] for row in rows),
            mean_kl_reference_to_leaf=sum(row["diagnostics"]["mean_kl_reference_to_leaf"] * row["evaluated_tokens"]
                                         for row in rows) / total)
    return result


def generation_comparison(expected: list[int], actual: list[int]) -> dict:
    prefix = 0
    for left, right in zip(expected, actual):
        if left != right:
            break
        prefix += 1
    return dict(exact_match=expected == actual, matching_prefix_tokens=prefix,
                first_mismatch_index=None if expected == actual else prefix,
                reference_tokens=expected, leaf_tokens=actual)


def logit_diagnostics(actual: np.ndarray, expected: np.ndarray,
                      sequences: list[list[int]]) -> dict:
    """Explain fidelity failures without changing acceptance gates.

    Elementwise checks include every logit. Distribution and argmax diagnostics
    include only rows with a next-token target, matching the likelihood metric.
    Process a few rows at a time to bound the float64 probability workspace.
    """
    if (actual.ndim != 2 or actual.shape != expected.shape or actual.shape[1] < 2
            or actual.shape[0] != sum(map(len, sequences))
            or not np.isfinite(actual).all() or not np.isfinite(expected).all()):
        raise ValueError("Invalid diagnostic logits")
    scored = np.ones(actual.shape[0], dtype=bool)
    offset = 0
    for row in sequences:
        if len(row) < 2:
            raise ValueError("Diagnostics require a next-token target per sequence")
        offset += len(row)
        scored[offset - 1] = False
    count = int(scored.sum())
    if count == 0:
        raise ValueError("No scored diagnostic targets")
    violations, worst_ratio, worst = 0, -1.0, None
    kl_sum, max_kl, max_tv, max_probability_error = 0.0, 0.0, 0.0, 0.0
    disagreements = []
    for start in range(0, len(actual), 8):
        a = actual[start:start + 8].astype(np.float64)
        e = expected[start:start + 8].astype(np.float64)
        ratio = np.abs(a - e) / (2e-4 + 2e-4 * np.abs(e))
        violations += int(np.sum(ratio > 1))
        index = np.unravel_index(int(ratio.argmax()), ratio.shape)
        if float(ratio[index]) > worst_ratio:
            worst_ratio = float(ratio[index])
            worst = dict(row=start + int(index[0]), token=int(index[1]),
                         reference=float(e[index]), leaf=float(a[index]))
        valid = scored[start:start + len(a)]
        positions = np.flatnonzero(valid)
        a, e = a[valid], e[valid]
        if not len(a):
            continue
        predicted, reference = a.argmax(-1), e.argmax(-1)
        for local in np.flatnonzero(predicted != reference):
            top_two = np.partition(e[local], -2)[-2:]
            disagreements.append(dict(row=start + int(positions[local]),
                                      reference_token=int(reference[local]), leaf_token=int(predicted[local]),
                                      reference_top2_margin=float(top_two[1] - top_two[0])))
        # Log-softmax avoids log(0) for underflowed probabilities.
        a -= a.max(-1, keepdims=True)
        e -= e.max(-1, keepdims=True)
        a -= np.log(np.exp(a).sum(-1, keepdims=True))
        e -= np.log(np.exp(e).sum(-1, keepdims=True))
        pa, pe = np.exp(a), np.exp(e)
        kl = np.maximum(0.0, (pe * (e - a)).sum(-1))
        difference = np.abs(pa - pe)
        kl_sum += float(kl.sum())
        max_kl = max(max_kl, float(kl.max()))
        max_tv = max(max_tv, float(0.5 * difference.sum(-1).max()))
        max_probability_error = max(max_probability_error, float(difference.max()))
    return dict(logit_count=int(actual.size), logit_tolerance_violations=violations,
                max_logit_tolerance_ratio=worst_ratio, worst_logit=worst,
                scored_rows=count, mean_kl_reference_to_leaf=kl_sum / count,
                max_kl_reference_to_leaf=max_kl, max_total_variation=max_tv,
                max_probability_error=max_probability_error, argmax_disagreements=disagreements)


def evaluate(args):
    from leaf import cli
    from tools.datasets import token_prefix
    from tools.decoder_plan import generation_policy_identity, greedy_reference_tokens, load_snapshot_plan
    from tools.decoder_validation import quality, run_native
    from tools.validate_decoder import GATES, digest, utc_now
    fp32_math = getattr(args, 'fp32_math', 'fast')
    if fp32_math not in ('fast', 'accurate'):
        raise ValueError('FP32 math must be fast or accurate')
    native_options = dict(threads=args.threads)
    if fp32_math == 'accurate':
        native_options['fp32_math'] = fp32_math
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError('Quality evaluation requires validation dependencies: '
                           'python -m pip install "leaf-cpu[validation]" (or ".[validation]" from source).') from error
    if args.output.exists():
        raise ValueError("Use a fresh output directory to preserve previous evidence")
    if not args.dataset.is_file():
        raise ValueError("Dataset file does not exist")
    args.output.mkdir(parents=True)
    report = args.output / "quality.json"
    record = dict(format="leaf-output-quality-v1", complete=False, passed=False,
                  model=args.model, measured_at_utc=utc_now(), precision="FP32", timing_qualified=False,
                  scope="Deterministic corpus prefix in independent fixed-length blocks; raw greedy prompts. "
                        "Measures model likelihood and runtime fidelity, not factual/chat task accuracy or speed.",
                  requested_blocks=args.blocks, sequence_length=args.sequence_length, threads=args.threads,
                  fp32_math=fp32_math,
                  dataset_source=args.dataset_source, dataset_sha256=digest(args.dataset), text_column=args.text_column,
                  torch=torch.__version__, transformers=transformers.__version__,
                  evaluator_sha256=digest(Path(__file__)),
                  gates=dict(**GATES["32"], logit_atol=2e-4, logit_rtol=2e-4, exact_generation_required=True),
                  blocks=[], generations=[])
    write_record(report, record)
    try:
        torch.set_num_threads(args.threads)
        cache = cli.cache_root()
        runtime = cli.executable(cache)
        if fp32_math == 'accurate':
            from leaf.runtime_check import check
            modes = check(runtime, 'leaf_decoder').get('fp32_math_modes')
            if not isinstance(modes, list) or 'accurate' not in modes:
                raise RuntimeError('The selected native runtime does not support --fp32-math accurate')
        snapshot = cli.resolve_model(args.model, cache, args.offline)
        cli.validate_model_generation(snapshot)
        plan = load_snapshot_plan(snapshot)
        config = plan["config"]
        record["generation_policy"] = generation_policy_identity(plan, args.generate)
        if args.sequence_length > config["max_positions"]:
            raise ValueError("Evaluation sequence length exceeds model context")
        artifact = cli.prepare(snapshot, cli.model_directory(snapshot, cache), 32)
        record.update(snapshot=str(snapshot), artifact_sha256=digest(artifact), executable_sha256=digest(runtime),
                      source_sha256={p.name: digest(p) for p in sorted(snapshot.iterdir())
                                     if p.is_file() and p.suffix in (".json", ".safetensors")})
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        ids = token_prefix(tokenizer, args.dataset, args.blocks * args.sequence_length, args.text_column)
        sequences = [ids[start:start + args.sequence_length] for start in range(0, len(ids), args.sequence_length)]
        fixture = args.output / "tokens.json"
        write_record(fixture, dict(sequences=sequences, raw_generation_prompts=PROMPTS))
        record["tokens_sha256"] = digest(fixture)
        write_record(report, record)
        model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                                                    dtype=torch.float32, attn_implementation="eager").eval()
        # One reference block at a time: no corpus-sized logits allocation or giant .npy cache.
        # Co-resident reference/native weights are suitable for small models; this is not timing.
        with torch.inference_mode():
            for index, row in enumerate(sequences):
                expected = model(torch.tensor([row]), use_cache=False).logits[0].numpy().copy()
                actual, _ = run_native(runtime, artifact, [row], **native_options)
                actual = actual.reshape(expected.shape)
                measured = quality(actual, expected, [row])
                targets = np.asarray(row[1:])
                measured.update(index=index, logits_allclose=bool(np.allclose(actual, expected, atol=2e-4, rtol=2e-4)),
                                leaf_correct_targets=int(np.sum(actual[:-1].argmax(-1) == targets)),
                                pytorch_correct_targets=int(np.sum(expected[:-1].argmax(-1) == targets)),
                                diagnostics=logit_diagnostics(actual, expected, [row]))
                record["blocks"].append(measured)
                record["summary"] = summarize(record["blocks"])
                write_record(report, record)
                print(f"Quality block {index + 1}/{args.blocks}: agreement {measured['next_token_agreement']:.4%}", flush=True)
                del actual, expected
            for prompt in PROMPTS:
                _, prompt_ids = cli.encode_prompt(snapshot, prompt, raw=True)
                if prompt_ids != tokenizer.encode(prompt):
                    raise ValueError("Leaf/reference prompt tokenization differs")
                if len(prompt_ids) + args.generate - 1 > config["max_positions"]:
                    raise ValueError("Generation check exceeds model context")
                expected = greedy_reference_tokens(model, prompt_ids, plan, args.generate)
                _, metrics = run_native(runtime, artifact, [prompt_ids], mode="generate", generate=args.generate, **native_options)
                compared = generation_comparison(expected, metrics["generated_tokens"])
                record["generations"].append(dict(prompt=prompt, input_ids=prompt_ids, **compared,
                                                 reference_text=tokenizer.decode(expected),
                                                 leaf_text=tokenizer.decode(metrics["generated_tokens"])))
                write_record(report, record)
                print(f"Generation {len(record['generations'])}/{len(PROMPTS)}: exact={compared['exact_match']}", flush=True)
        for path, checksum in ((runtime, record["executable_sha256"]), (artifact, record["artifact_sha256"]),
                               (args.dataset, record["dataset_sha256"]), (fixture, record["tokens_sha256"])):
            if digest(path) != checksum:
                raise ValueError("An evaluation input changed during execution")
        for name, checksum in record["source_sha256"].items():
            if digest(snapshot / name) != checksum:
                raise ValueError("Model snapshot changed during execution")
        summary = record["summary"]
        record.update(complete=True, passed=bool(summary["next_token_agreement"] >= GATES["32"]["min_next_token_agreement"]
                      and summary["perplexity_ratio"] <= GATES["32"]["max_perplexity_ratio"]
                      and summary["all_logits_close"] and all(row["exact_match"] for row in record["generations"])))
        if not record["passed"]:
            raise ValueError(f"Output fidelity gate failed; retained measurements: {report}")
    except BaseException as error:
        record["error"] = str(error) or type(error).__name__
        raise
    finally:
        write_record(report, record)
    print(json.dumps(record["summary"], indent=2))
    print(f"Quality passed; no timing qualification. Report: {report}")
