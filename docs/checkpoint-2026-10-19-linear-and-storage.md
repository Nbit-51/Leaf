# October 19: exact kernels, linear cost and Smol/Qwen versus Ollama

This checkpoint records work completed on `codex/linear-accumulation` before
publication. Accurate accumulation remains separate from released defaults.
The user subsequently authorized merging after CI passes and deleting the branch.
No benchmark or local model server is left running.

## Implemented and checked

- SiLU interleaves four independent FP64 polynomial chains, preserving the
  previous per-lane operations, coefficients, rounding and four-lane exceptional
  fallback. The native test passes **5,257,011 checks**, maximum ULP error zero.
  Guards, aliases, unaligned arrays, special values and SIMD tails are covered.
- `TokenPanelsF32` retains constructed workspace across shrinking/growing widths,
  avoiding redundant clearing of a tail that packing immediately overwrites.
  Logical extents and active-panel padding are unchanged. Alternating dimensions,
  scalar/vector packing, guards and input preservation pass native checks.
- **1,048 unit tests** and ten reduced architecture configurations pass.
- Accurate K=256 plus these changes matches all sixteen prior full quality blocks
  bit-for-bit. All twelve reference continuations and twelve cache/scalar/thread/
  earlier-model/W8A8 checks pass. The complete trained checks remain 1,016 scored
  targets/model, 100% agreement and zero strict logit violations, with the original
  `2e-4 + 2e-4*abs(reference)` tolerance. Reduced architectures are not trained
  model evaluations.

The reversed-order SiLU probe (31 samples/pass) measures **17–19% lower operator
latency** and exact old/new output. Short model screens show smaller, mixed
effects and do not qualify a whole-model gain. No approximation was introduced.

## Where time goes

The new projection profiler excludes warmups and verifies exact instrumentation
parity. Two balanced diagnostic pairs, seven measured forwards/pass, one thread.
These are mean scoped times of the accurate K=256 candidate **before** today's
SiLU/workspace changes, not acceptance medians. Pack/compute and enclosing
projection scopes overlap; never sum them twice.

| Model | Prefill total | MLP up + gate + down | Decode total | Vocabulary output projection |
|---|---:|---:|---:|---:|
| SmolLM2-360M | 508.99 ms | 341.21 ms / 67.0% | 80.68 ms | 10.23 ms / 12.7% |
| Qwen2.5-0.5B | 582.65 ms | 449.46 ms / 77.1% | 112.44 ms | 30.57 ms / 27.2% |

Attention projections are the next prefill cost; RMSNorm is roughly 1% or less.
Explicit panel reuse saves about 5–6 ms, while accurate GEMM adds compute cycles.
Forward CPU time and cycle controls show that neither all noise nor exclusively
kernel work explains every wall-clock variation.

The fused partial/tree idea did not beat the existing kernel. Stop it; its
sources/results remain in `build/linear-accumulation-20261019-merged*`, and failed
code was removed from the working engine/builder. The older OpenBLAS and
twelve-row cache failures are completed; do not repeat without a new hypothesis.

## Fresh matched FP32 Ollama comparisons

These are the first completed local Smol/Qwen Ollama comparisons found in retained
evidence. Leaf uses the **accurate experimental K=256 candidate**, including today's
changes, not the released binary. Ollama is the pinned local v0.40.4 CPU runtime.

Every source tensor (290/model) matches the converted F32 GGUF exactly, permitting
only declared Llama Q/K layout permutation and tied aliases. Fresh Leaf exports
are byte-identical to the quality-checked artifacts. Both runtimes: one thread,
CPU 2, above-normal priority, FP32 weights/KV, identical 64-token IDs, zero shared
cached prompt tokens for these tokenizers, two output tokens and one cached decode
forward. Load/HTTP time is separate. Six balanced pairs, three warmups and eleven
measurements/pass; 66 measured samples/runtime/model. No filtering.

**Every pass and pooled stability check passes. Every timed visible output matches.**

| Model | Leaf prefill | Ollama prefill | Leaf decode | Ollama decode |
|---|---:|---:|---:|---:|
| SmolLM2-360M-Instruct | 497.6553 ms | 585.9160 ms | 81.53365 ms | 90.6600 ms |
| Qwen2.5-0.5B-Instruct | 572.2712 ms | 642.9490 ms | 111.74465 ms | 122.4930 ms |

Median paired Leaf/Ollama ratios: Smol 0.84722 prefill / 0.89919 decode;
Qwen 0.88828 / 0.91225. These are timing advantages, **not an overall qualified
speed claim under the unchanged cross-runtime preflight**.

All eight visible preflight outputs match/model, but selected log-probability
differences exceed 0.0002 (Smol max 0.00026337; Qwen 0.00085008). Smol's immediate
EOS also has no Ollama-reported log-probability. Failed preflights are retained;
timing explicitly uses `--diagnostic-timing`; `overall_speed_claim` stays false.

An independent cached eager FP32 PyTorch check on **the disputed prompts** finds
Leaf/reference selected-log-probability errors <=0.0000125, with all conditional
argmax tokens matching. Ollama's reported errors reach 0.00085123. This points
to a cross-runtime numerical/reporting difference rather than a demonstrated
Leaf accuracy regression. It does not override the original gate, establish
Ollama's internal cause, or justify calling Ollama's inference incorrect.

## Next measured opportunity: original BF16 storage, FP32 computation

Both snapshots have BF16 source weights, currently expanded exactly to FP32.
A separate native probe retains original BF16 vocabulary weights, expands eight
values into FP32 AVX2 registers, then uses the established paired-GEMV FMA and
reduction order. **No additional source-weight rounding.** All 49,152 Smol and
151,936 Qwen outputs are bit-identical for the tested input. Four reversed-order
passes, eleven samples and three warmups each; all pass/pooled spreads stable:

| Real vocabulary tensor | FP32 storage | Original BF16 storage | Ratio |
|---|---:|---:|---:|
| Smol 49,152 x 960 | 9.74360 ms | 5.98585 ms | 0.61434 |
| Qwen 151,936 x 896 | 29.34230 ms | 18.18425 ms | 0.61973 |

This **38% operator reduction is not integrated into the decoder** and is not
a model or resident-memory claim. Half the weight bytes are streamed, but the
probe holds both storage arrays. See `benchmark/experiments/source-bf16`.

Next: typed source-preserving storage with FP32 computation, starting at the
vocabulary projection. Preserve exact FP32 fallback for values not representable
in BF16; never silently truncate FP32 weights. Cover artifact compatibility,
scalar/AVX2, tails, all-logit verification, cache/threads, load time and memory.
Require full model quality and fresh whole-model timing before a default.
Prefill's MLP projections remain the largest compute target; this storage idea
addresses the separately measured decode/output-projection cost.

## Power and durable state

AC was lost after both comparisons completed. Windows confirmed unplugged;
only correctness work continued. After restoration, Windows confirmed AC;
the storage probe then ran with fresh warmups and native per-sample AC guards.
No battery timing was accepted. `power-interruption.json` also verifies that
all earlier Ollama request boundaries and native Leaf samples checked AC.

Compact raw records are retained under:

- [`linear-accumulation/results/20261019`](../benchmark/experiments/linear-accumulation/results/20261019)
- [`silu-pipeline/results`](../benchmark/experiments/silu-pipeline/results)
- [`source-bf16/results`](../benchmark/experiments/source-bf16/results)
- [`family Ollama records`](../benchmark/results/family-ollama-20261019)

Large binaries, model weights and logits remain in ignored `build/`. Binaries:

- Accurate: `build/linear-accumulation-20261019-pipeline/leaf_decoder.exe`, SHA256
  `af65268c755564834ae9c1d25d591d1480f4120c04ea8a4d1a8138110dcca7be`.
- Adapter: same folder, `generation-adapter.exe`, SHA256
  `02aa881dd3ad22f22ae257214f7f586f1053ce448e4b63d8257d230f1e8a8337`.
- Released arithmetic plus exact kernel changes:
  `build/silu-pipeline-20261019-release/leaf_decoder.exe`, SHA256
  `12727301cba1419cffa19d6daa8a15e457acffbc21944fd640551ab35667f03b`.

Old manifests are immutable. `verified-dependencies.json` supplements today's
build records at verification time. Future experiment builds hash all engine
headers, including activations, instead of only projection headers.
