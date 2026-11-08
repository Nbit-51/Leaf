# Original BF16 storage, FP32 computation

`leaf run --source-bf16-head` retains the checkpoint's original BF16 vocabulary
projection weights, plus the embedding when tied to the same source tensor.
In this head-only mode, all other weights remain FP32. This is an optional storage policy, not a change
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
order. Multi-token projection expands bounded row blocks for the existing
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

## Native accurate math compatibility

The October 26 native accurate-math option composes with this storage format:
`leaf run MODEL --bits 32 --source-bf16-head --fp32-math accurate --prompt "Hello"`.
Storage retains original source values; accumulation controls computation.
The combined offline installed-wheel path passes without compiler or binary
overrides. [Accurate-policy migration checks](shared-accumulation.md#native-accuracy-option--october-26-2026)
preserve the trained outputs. No fresh storage speedup is inferred from those
correctness checks. These October 26 checks cover the output head and tied embedding; the broader
optional matrix policy is described below.

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


## Broader matrix storage (October 30, 2026)

`leaf run MODEL --bits 32 --source-bf16 --fp32-math accurate --prompt "Hello"`
retains eligible attention, MLP and embedding matrix weights that were originally
stored in BF16. Norms and biases remain FP32. The existing `--source-bf16-head`
option keeps its narrower behavior; select one scope. Both are opt-in. Ordinary
FP32 preparation and automatic quantization profiles retain their existing policy.

The exporter reads original BF16 payloads as unsigned 16-bit values and applies
layout transforms without rounding. It checks source/transformed finite values
and shares identical transformed aliases; protecting one identical alias protects
the entire group. F32/F16 sources stay FP32 even when numerically representable
in BF16. If there is no eligible matrix, normal FP32 preparation is reused.

Interior BF16 matrices require artifact version 6. Older readers reject its magic;
the CLI verifies native capability before downloading weights. The new cache is
`decoder-32-source-bf16-matrices.leaf`, separate from FP32 and head-only caches.
Existing versions 2-5 remain readable. Integer activations, mixed quantized weights,
calibration vectors, BF16 norms/biases and quantization metadata are rejected.

Computation remains the selected FP32 policy. Single-token projection converts
original BF16 values directly into FP32 registers and uses the existing reduction.
Prefill expands six-row blocks into FP32 before the existing GEMM. It does
not retain a full FP32 weight copy. Conversion and scratch-buffer costs can affect
prefill; a smaller artifact does not itself establish a latency improvement.

`leaf evaluate` currently prepares ordinary FP32 artifacts. The separate developer
`tools/validate_source_bf16_matrices.py` audits the new storage against a bound,
previously qualified FP32 runtime/artifact, including all tensor bits, full trained
logits, generation and cache/scalar/thread controls. This is an exact regression
check, not a fresh PyTorch oracle or a relaxed accuracy gate. Ordinary fast math's
previous strict-reference tolerance misses remain open.


Current broader-storage validation: **1,513 unit checks** pass on the initial
96-row version; **176 focused checks** and the complete trained audit pass on
the selected six-row version. Both SmolLM2-360M and Qwen2.5-0.5B preserve all
trained logit bits in fast/accurate policies, generation and cache/scalar/thread
controls. Their artifacts occupy about half their original FP32 file sizes.
The installed six-row wheel also passes compiler-free offline execution.

The initial 96-row implementation's short, stable head-only-vs-matrix screens
show 33-36% lower decode time with 9-12% higher prefill time. These are **initial
candidate diagnostics**, not timing evidence for the selected six-row version.
The smaller buffer's latency measurement is pending. No new default, competitor
speed claim or full release timing qualification follows from these checks.

[Source, raw diagnostics, test reports and qualification identities](../benchmark/validation/source-bf16-matrices/20261030/manifest.json)
are retained with the [session checkpoint](checkpoint-2026-10-30-gemm-alignment.md).


## November 8: prefill expansion follow-up

The explicit BF16 storage path now uses guarded AVX2/FMA bulk expansion where
available; the original portable function remains available and forced-scalar
execution retains its policy. Eight original BF16 values widen by integer
operations, shift into the FP32 bit layout and store without rounding.
Single-token GEMV and the storage format are unchanged.

The resumed six-row-versus-head-only screen passes stability on both models:
Smol prefill 506.06->551.28ms/decode 76.40->51.09ms; Qwen 568.06->597.89ms/
96.06->65.27ms. These short results describe the prior six-row implementation,
not the subsequent AVX2 update. Smaller storage helps decode while conversion
and GEMM still cost prefill time.

Matched 96-row-versus 6-row accounting records approximately 7,700-14,900 process
page faults per prefill sample for the large temporary and zero for the small
one. These include soft/allocation faults; disk paging is not established.
The generated portable expansion uses SSE2. Isolated production-layout AVX2
checks lower copy time for 5,376/5,760/15,360 values, while 29,184 is neutral.

For the actual AVX2 update, all native bit/guard checks pass (**4,292,253**),
the full Python suite passes (**1,513**, two optional legacy skips), and the
complete trained audit preserves **411,828,224** logit values across fast and
accurate policies. Generation, cache/scalar/thread checks and actual installed
wheel execution also pass. This preserves prior qualification, rather than
constituting a new independent PyTorch reference run.

A full serial comparison against the previous six-row implementation uses
accurate/accurate math, identical BF16 values, CPU2, one thread, six balanced
pairs, 31samples and 10warmups/pass. All individual and pooled phases are stable;
AC/effective-mode audits pass and every sample is retained.

| Model | Prefill before->AVX2 | Observed reduction | Decode before->AVX2 |
|---|---:|---:|---:|
| SmolLM2-360M |521.22455->513.15220ms|1.55%|49.04615->48.82465ms|
| Qwen2.5-0.5B |590.55420->587.68990ms|0.49%|63.91170->63.88160ms|

Both miss the declared 2% prefill improvement threshold; decode passes
non-regression. The guarded SIMD expansion is retained with the verified
operator-level scope. No qualified 2% whole-model or competitor win is claimed.
The default storage policy remains opt-in; do not mix these incremental results
with head-only or other-runtime measurements.

[Raw records, sources, tests and hashes](../benchmark/validation/bf16-expansion/20261108/manifest.json)
and the [resume checkpoint](checkpoint-2026-11-08-bf16-expansion.md) preserve the
complete evidence. No unchanged acceptance run was repeated to seek a pass.
