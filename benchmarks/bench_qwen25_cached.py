"""Offline causal-decoder CPU baseline with and without the KV cache.

The benchmark never downloads implicitly. Point ``--model`` at a complete
local snapshot, or populate the Hugging Face cache first. This is a PyTorch
baseline for the transformer work; Leaf does not yet execute the full model.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import time

import torch
from transformers import AutoModelForCausalLM
import transformers

try:
    import psutil
except ImportError:  # The WSL benchmark environment intentionally stays small.
    psutil = None


def _median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1000.0


def _model_bytes(model) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def _rss_bytes() -> int | None:
    if psutil is not None:
        return int(psutil.Process().memory_info().rss)
    try:
        # Linux reports resident pages in /proc; no third-party package needed.
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, IndexError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                        help="local snapshot path or cached Hugging Face model id")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark/results/qwen25_cached_cpu.json"))
    parser.add_argument("--benchmark-name", default="qwen2.5-0.5b-pytorch-cpu-kv-cache",
                        help="result label when benchmarking another local decoder")
    args = parser.parse_args()
    if args.sequence_length < 2 or args.runs < 1 or args.threads < 1:
        parser.error("sequence length must be >=2; runs and threads must be positive")

    torch.set_num_threads(args.threads)
    rss_before = _rss_bytes()
    load_start = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
    except OSError as error:
        raise SystemExit(
            "Model snapshot is incomplete. This benchmark is offline-only; cache the model "
            "weights first or pass --model with a complete local snapshot.\n"
            f"Original error: {error}"
        ) from error
    load_seconds = time.perf_counter() - load_start
    rss_loaded = _rss_bytes()

    vocab_size = int(model.config.vocab_size)
    token_ids = (torch.arange(args.sequence_length, dtype=torch.long) * 7919 + 17) % vocab_size
    input_ids = token_ids.unsqueeze(0)
    prefix, final_token = input_ids[:, :-1], input_ids[:, -1:]

    @torch.inference_mode()
    def uncached_forward():
        return model(input_ids=input_ids, use_cache=False).logits[:, -1, :]

    @torch.inference_mode()
    def cached_forward():
        state = model(input_ids=prefix, use_cache=True)
        return model(
            input_ids=final_token,
            past_key_values=state.past_key_values,
            use_cache=True,
        ).logits[:, -1, :]

    for _ in range(args.warmup):
        uncached_forward()
        cached_forward()

    uncached_logits = uncached_forward()
    cached_logits = cached_forward()
    maximum_error = float(torch.max(torch.abs(uncached_logits - cached_logits)))
    same_token = int(torch.argmax(uncached_logits)) == int(torch.argmax(cached_logits))

    uncached_times = []
    cached_times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        uncached_forward()
        uncached_times.append(time.perf_counter() - start)

        # Prefix construction is deliberately outside the timed region. The
        # measurement isolates one-token decode with an already populated KV cache.
        with torch.inference_mode():
            state = model(input_ids=prefix, use_cache=True)
        start = time.perf_counter()
        with torch.inference_mode():
            model(input_ids=final_token, past_key_values=state.past_key_values, use_cache=True)
        cached_times.append(time.perf_counter() - start)

    uncached_ms = _median_ms(uncached_times)
    cached_ms = _median_ms(cached_times)
    result = {
        "benchmark": args.benchmark_name,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "torch": torch.__version__, "transformers": transformers.__version__},
        "model": Path(args.model).name if Path(args.model).exists() else args.model,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "precision": "float32",
        "threads": args.threads,
        "sequence_length": args.sequence_length,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "model_parameter_bytes": _model_bytes(model),
        "load_seconds": load_seconds,
        "rss_before_load_bytes": rss_before,
        "rss_after_load_bytes": rss_loaded,
        "uncached_full_context_decode_p50_ms": uncached_ms,
        "cached_single_token_decode_p50_ms": cached_ms,
        "kv_cache_speedup": uncached_ms / cached_ms,
        "cached_vs_uncached_max_abs_logit_error": maximum_error,
        "cached_vs_uncached_argmax_match": same_token,
        "scope": "PyTorch CPU baseline; Leaf full-transformer execution is not implemented yet.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
