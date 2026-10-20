# Original BF16 storage, FP32 computation

`leaf run --source-bf16-head` retains the checkpoint's original BF16 vocabulary
projection weights, plus the embedding when tied to the same source tensor.
All other weights remain FP32. This is an optional storage policy, not a change
to the model's computation precision or an automatic quantization profile.

```sh
leaf run HuggingFaceTB/SmolLM2-360M-Instruct --bits 32 --source-bf16-head --prompt "Explain CPU inference." --max-tokens 32
leaf run Qwen/Qwen2.5-0.5B-Instruct --bits 32 --source-bf16-head --prompt "Explain CPU inference." --max-tokens 32
```

Complete local snapshots work with the same flag and `--offline`. The installed
runtime must support decoder artifact version 5; an older runtime is rejected
before downloading weights. Reinstall Leaf after updating its source checkout.
No decoder binary environment variable is required by a normal wheel install.

## Exact storage semantics

The exporter checks the original safetensors dtype. It never converts arbitrary
FP32 source weights to BF16, even when some values happen to be representable.
For an FP32 source it reuses the normal FP32 cache. For an eligible BF16 source,
it writes a separate `decoder-32-source-bf16.leaf` artifact and preparation report.
The existing FP32 artifact and precision-selection profiles remain separate.

BF16 values expand into FP32 by shifting their bits; no source information is
discarded. AVX2 paired GEMV preserves the FP32 kernel's accumulator and reduction
order. Multi-token projection expands bounded 96-row blocks for the existing
FP32 GEMM. Scalar and column-tail paths preserve their corresponding arithmetic.
CPU feature checks still select the portable fallback where needed.

The compact payload is shared for tied embedding/output weights. Protecting a
tied alias with `keep_fp32_tensors` protects the whole group instead of keeping
both copies. Only embedding/output tensors may have 16-bit storage in format 5;
integer activations and mixed quantized weights are rejected for this format.
Versions 2, 3 and 4 remain readable and their ordinary exports are unchanged.

Native metrics report `source_bf16_tensors` and `source_bf16_linear_calls` so the
selected path can be checked. `weight_bits` and `activation_bits` describe the
FP32 computation policy; the storage counters identify the compact payloads.

## Validation scope

The native primitive test checks all 65,536 BF16 expansion patterns and paired
GEMV equality, guards and unaligned inputs. Native model checks cover FP32 source
fallback, tied/untied heads, scalar/AVX2, one/two threads, token counts crossing
the GEMM dispatch boundary, odd vocabulary rows, column tails, chunked KV cache,
generation, end-token sets and causal sliding windows. Malformed storage metadata
and older-runtime handling are tested explicitly.

For the trained SmolLM2-360M and Qwen2.5-0.5B checkpoints, every exported tensor is
audited against the existing FP32 artifact. Eight held-out 128-token full-logit
blocks/model are bit-identical to the previously quality-qualified accurate
accumulation candidate. Six reference continuations/model and trained scalar,
chunked-cache and two-thread controls also pass. This reuses the prior PyTorch
qualification by exact output binding; it is not a fresh PyTorch model evaluation.

The accurate K=256 accumulation candidate remains experimental. Its measurements
must not be presented as timings of the ordinary installed default, or as a new
comparison with Ollama, llama.cpp or PyTorch. The new storage option is available
with the ordinary runtime, but its arithmetic policy is unchanged.

This reduces artifact bytes and output-head traffic. It is not a measured peak
resident-memory or cold-start improvement. MLP projections remain the main
prefill compute cost; vocabulary storage targets a separate decode bottleneck.

## Controlled October 20 results

Windows, one pinned P-core (CPU 2), one thread, warmed 63-token prefill and one
cached decode token with last-token logits. Six balanced AB/BA pairs/model,
31 measurements and ten warmups/pass. Load, tokenization and command startup
are excluded. Both sides use the same accurate K=256 accumulation policy.

| Model | FP32-storage decode | Source-BF16 decode | Median paired reduction | Prefill before / after |
|---|---:|---:|---:|---:|
| SmolLM2-360M-Instruct | 80.24445 ms | 76.20240 ms | **5.12%** | 493.38740 / 491.26880 ms |
| Qwen2.5-0.5B-Instruct | 110.50400 ms | 97.50815 ms | **11.65%** | 568.44785 / 557.07875 ms |

Every individual and pooled spread passes p90/p10 <=1.25. The predeclared gate
requires paired decode ratio <=0.98 and prefill <=1.02; both models pass both
phases. All samples are retained. AC monitoring reports no failures and Windows
sleep-event auditing reports no events during the repeat.

Artifacts shrink from 1,447,312,000 to 1,352,940,160 bytes for Smol (6.52%) and
1,976,158,464 to 1,703,889,152 bytes for Qwen (13.78%). The different decode gains
are consistent with the retained profile: the output head accounted for about
13% of Smol decode and 27% of Qwen decode. This supports targeting traffic in a
shared operation; it does not imply every model will gain the same percentage.

**1,125 unit tests pass**, plus 97 focused real-native checks with the conservative
build. Ten reduced architecture configurations pass. The final Windows wheel
passes isolated installed startup and offline compact-artifact preparation/run
without compiler paths, binary overrides or Torch/Transformers imports. Wheel
startup/generation checks are functional evidence, not a separate timing gate.

[Retained samples, source/binary identities and validation manifest](../benchmark/experiments/source-bf16/results/20261020/validation.json)
include the failed first attempt and exact original runner. The regular default
arithmetic remains unchanged; these gains are scoped to the tested accurate
candidate and original BF16 checkpoints.

## Benchmark sleep controls

The first October 20 timing attempt overlapped Windows Modern Standby from
14:49:15 to 14:54:45 IST. Qwen failed stability. The complete attempt is retained
as unqualified; no samples were removed and the apparently passing Smol summary
does not rescue the interrupted attempt.

The repeat explicitly requests both system and display activity for the duration
of timing, restores the previous thread state afterwards, records pass timestamps,
and checks Windows sleep events before qualifying results. Normal Leaf inference
does not request this display policy. The requests use Microsoft's documented
[SetThreadExecutionState flags](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate)
and do not change the persistent power plan. Manual sleep can still override
the request and would invalidate a timing run.
