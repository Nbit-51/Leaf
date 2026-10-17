# Shared FP32 decode SiLU — October 17, 2026

The Windows default now uses the existing vector SiLU gate for pure-FP32
single-token decoder calls. It changes shared workload dispatch, with no model
ID special case or new kernel math. It requires the existing AVX2/FMA checks,
a SiLU-gated MLP, FP32 activations and all-FP32 weights. W8A8 remains
prefill-only, and scalar, INT4, historical research builds and other OS policies
retain their existing paths. Short multi-token calls below eight tokens remain
scalar unless their size is one.

## Controlled incremental results

One Windows host, pinned logical CPU 2, one thread, FP32, batch one, 63-token
prefill plus one cached decode token, last-token logits. Six balanced AB/BA
pairs per model, ten warmups and 31 measured samples per pass. Both models
pass every individual and pooled stability check, including p90/p10 <= 1.25.
The predeclared decode gate requires a pooled ratio <= 0.98 and prefill <= 1.02.
The existing prefill-target default and stability rules were not relaxed.

| Model | Before decode | After decode | Reduction | Before / after prefill |
|---|---:|---:|---:|---:|
| SmolLM2-360M-Instruct | 83.4384 ms | 78.74105 ms | **5.63%** | 490.5763 / 489.08675 ms |
| Qwen2.5-0.5B-Instruct | 115.6098 ms | 109.27475 ms | **5.48%** | 563.0904 / 564.99775 ms |

Median paired decode ratios are 0.94847 and 0.94540 respectively. Prefill
passes non-regression; no prefill improvement is claimed. These compare with
Leaf at merged PR #7 (`1e0e57f`), not PyTorch, Ollama or llama.cpp. They exclude
loading, tokenization and command startup and do not establish long-generation
or multi-core latency.

Raw samples, frozen token fixtures and hashes:
[Smol acceptance](../benchmark/results/fp32-decode-silu/smol-acceptance.json),
[Qwen acceptance](../benchmark/results/fp32-decode-silu/qwen-acceptance.json).
Revisions are `a10cc1512eabd3dde888204e902eca88bddb4951` and
`7ae557604adf67be50417f59c2c2f167def9a775` respectively.

## Correctness and release binding

The measured candidate passes twenty exact-output cases across trained Smol,
Qwen and TinyLlama: full logits, chunked cache, tokenwise cache, two threads,
forced scalar and generation. W8A8 tokenwise and generation exclusions also
pass. The final source build is separately bound to that candidate on eight
trained cached/generation cases. The source adds a compile-time guard to
preserve research-build policy; it selects the same path as the measured
candidate in the normal Windows build.

The final build passes ten reduced architecture configurations with exact
before/after outputs, reference tolerances, cache resets, greedy generation,
FP32/scalar/two-thread checks, W8A8/INT4 and smoothed artifacts. The dispatch
audit additionally exercises FP32 weights with 8-bit activations and requires
that this mixed path remain excluded. All **1,040 unit tests pass**. An initial
sandbox run could not write native request files in its system temp directory;
the complete suite passed with a fresh workspace temp directory.

A freshly built and installed wheel passes GPT-2 first-preparation and cached
repeat execution against saved trained reference tokens. A Qwen Hub-ID offline
run selects its bundled binary, preserves both EOS IDs, executes vector SiLU,
and emits exactly the previous installed runtime's tokens. Both installed
checks have no compiler on PATH and import no Torch/model libraries. Package
smoke observations are not latency gates.

[Release evidence](../benchmark/results/fp32-decode-silu/release-validation.json)
links the exact binary, source and report identities. The earlier strict
elementwise quality failures remain unchanged: byte-identical execution
preserves current output quality; it does not resolve those failures.

## What the operation trace established

Two short default profiles per model put linear operations at 94–95% of
prefill and 88–91% of decode. Scalar SiLU accounts for about 7–8% of decode,
which motivated this bounded change. Profiles include first use and warmups;
their absolute timings are diagnostic, not acceptance measurements.

The new `tools/trace_decoder_logits.py` traces shared Llama, Qwen2 and NeoX
operator families. It binds saved model/input hashes, captures the same stages
in native and reference execution, and requires diagnostic native logits to be
byte-identical to the ordinary runtime. Its reference hooks are tested for
exact logits, full stage coverage, correct rotary token axes and cleanup after
failure.

Saved failing block 1 is reproduced for both trained models. Worst failures
are on row zero, where attention has only one key. The first stage exceeding
the final-logit tolerance on that row is Smol layer 31's down projection and
Qwen layer 22's gated activation. That threshold crossing alone does not
identify the operation that introduced the error. At Qwen's failing activation,
the observed difference is -0.000484511; recomputing from its differing operands
predicts -0.000484823. Activation arithmetic contributes about 0.000000243:
the dominant error was propagated into the operation.

Same-input RMSNorm probes reproduce Leaf's current norm bit-for-bit. Summing
FP32 squares with FP64 accumulation, then rounding the sum to FP32 before the
existing scale calculation, reduces norm error against Torch roughly fivefold.
Torch uses hierarchical FP32 partial sums rather than Leaf's serial sum:
[pinned CPU sum implementation](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/cpu/SumKernel.cpp#L287-L422).
Multiplication association already agrees; there is no evidence here for
changing the reciprocal-square-root formula.

| Saved failing block only | Current violations | FP64 norm sum | Norm + two-bank GEMM |
|---|---:|---:|---:|
| Smol block 1 | 154 | 68 | 2 |
| Qwen block 1 | 451 | 5 | 0 |

The two-bank GEMM improves synthetic error but shows higher observed micro
cost. Those micro timings have no qualified pinning/stability claim, and the
combined candidate still fails Smol. **Neither numerical candidate ships.**
No tolerance was increased, and a passing single Qwen block is not a complete
trained-model quality pass. [Retained diagnostic summary](../benchmark/results/fp32-decode-silu/numerical-diagnostics.json).

Next work should address the remaining shared accumulation error without
adding the observed GEMM cost, then validate all retained blocks before any
new competitor claim. Linear projection cost remains the main performance
target. There is no evidence here that a new model-specific attention pass is
needed to explain these particular failures.

## Reproduce

Build the default with `scripts/build_decoder.ps1`, then run the architecture
audit with `--expect-silu-defaults`. Use an independently retained prior binary
for `--before-executable`. For a controlled decode comparison, pass
`--target-phase decode` to `tools/benchmark_default_abba.py` with six pairs,
31 runs and ten warmups; keep the machine plugged in and otherwise idle.
Use a fresh output file and frozen `decoder-32.leaf` / `tokens.json` workdir.
All samples, failures and interrupted records must remain retained.

Trace a retained quality block with a `LEAF_DIAGNOSTIC_TRACE=1` build and
`tools/trace_decoder_logits.py --quality-record ... --tokens ... --block 1
--executable ... --baseline ... --output ...`. Use the quality record's original
cache root (`LEAF_CACHE_DIR`) and baseline binary; never mix trace and timing
executables or change frozen fixtures to obtain a pass.
