# Windows W8A8 vector SiLU/gating release

Status: integrated into the Windows W8A8 default after stable incremental
acceptance, a fresh PyTorch comparison and installed-wheel correctness checks.
**FP32 SiLU remains experimental; its incremental timing remains unqualified.**

## Release integration verified — 2026-10-10

The default path requires Windows, AVX2/FMA, a SiLU-gated FFN, eight-bit
activations, at least one INT8 tensor, no INT4 tensors, and at least eight tokens
in the forward call. FP32-protected tensors are permitted within W8A8. Forced
scalar execution, single-token decode, other activations and other operating
systems retain their previous paths. This does not authorize automatic W8A8
precision selection; that still requires a matching validation profile.

`LEAF_OPTIMIZED_SILU_W8A8` controls this policy. Use `-ConservativeSiluGate` in
the PowerShell builder or `LEAF_CONSERVATIVE_SILU_GATE=ON` in CMake to disable it.
`LEAF_EXPERIMENTAL_SILU_GATE` retains the broader opt-in precision coverage.

Release checks completed on October 9 and their saved records were verified
on October 10, without repeating completed benchmarks:

- **870 Python tests passed**, with four existing ONNX deprecation warnings.
- **Nine reduced architecture cases passed**, including gated ReLU, scalar and
  INT4 exclusions, threading, chunked cache, reset and generation checks.
- **Ten trained TinyLlama parity checks passed**: full held-out logits, chunk16,
  tokenwise cache, generation and forced scalar at each precision. Installed W8A8
  outputs exactly match the accepted candidate; installed FP32 outputs exactly
  match the previous baseline. This is not exact W8A8 parity with FP32 PyTorch.
- The locally built wheel selected its bundled decoder with no compiler on
  PATH; offline GPT-2 first preparation and cached reuse matched reference tokens.
  No heavy model frameworks were imported, and a stale precision profile fell
  back to FP32.

Installed decoder SHA256:
`e6b247c6597db56082557a657b295d530ff0a165239bbb3e53f0f7f2f34fe711`.
This is a local wheel validation, not a PyPI publication. The installed wheel
was not separately latency-benchmarked: the timings below belong to experimental
decoder `cdf897c563a53295ae208a1c7fa69886d5e686b32c10e9914dbf6001dee0f15d`.

Evidence: [release summary](../benchmark/results/silu-gate/release-validation.json),
[trained parity](../benchmark/results/silu-gate/release-parity.json),
[architecture audit](../benchmark/results/silu-gate/release-architectures.json),
and [installed package](../benchmark/results/silu-gate/release-package.json).

## Accepted W8A8 incremental result — 2026-10-09

The user completed all six alternating AB/BA pairs: 31 measured samples and
ten warmups per pass, one thread on CPU 2, Windows Above Normal priority,
63-token prefill plus one cached decode token. This gives 186 samples per
build per phase. The assistant verified the complete record, passing gate
and every recorded input hash against the local files.

| Phase | Previous Leaf W8A8 | SiLU candidate W8A8 | Change |
|---|---:|---:|---:|
| Prefill | 1320.3094 ms | **819.3776 ms** | **37.94% lower; 1.611× speedup** |
| Decode | 88.1954 ms | 88.3723 ms | +0.20%; passes non-regression |

All 12 individual passes and both builds' pooled phase samples pass stability.
Pooled p90/p10 ratios: prefill 1.0219/1.0246 and decode 1.0440/1.0437
(before/after). Median paired prefill ratio is 0.61997; pooled ratio is
0.62060. No outliers were removed and no gate thresholds were changed.

This qualifies the incremental SiLU change for the tested W8A8 workload.
It is not a fresh PyTorch comparison, a decode speedup, or FP32 acceptance.
The existing W8A8 quality tradeoff remains: 95.0787% next-token agreement,
perplexity ratio 1.0123987 and non-identical generation versus FP32 PyTorch.
Full samples and provenance: [W8A8 acceptance record](../benchmark/results/silu-gate/timing-8.json).

## Fresh PyTorch comparison — 2026-10-09

The accepted experimental executable was measured against fresh PyTorch eager
and SDPA FP32 processes, in forward and reverse order. Same cached model
snapshot and tokens, CPU 2, one thread, 31 samples and ten warmups per pass;
63-token prefill plus one cached decode token, last-token logits only.
Each median below pools 62 samples. All four runtimes pass individual, pooled
and between-pass stability checks.

| Runtime | Prefill | Decode | Stability |
|---|---:|---:|---|
| Leaf SiLU W8A8 candidate | **861.59695 ms** | **91.46975 ms** | Pass |
| Leaf experimental FP32 candidate | 1428.40295 ms | 226.55550 ms | Pass |
| PyTorch eager FP32 | 1804.02280 ms | 212.68855 ms | Pass |
| PyTorch SDPA FP32 | 1767.87970 ms | 211.42825 ms | Pass |

Against the faster PyTorch baseline (SDPA in both phases), W8A8 has **51.26%
lower prefill time (2.052× speedup)** and **56.74% lower decode time (2.311×)**.
This compares calibrated W8A8 with FP32, with the quality tradeoff stated above;
it is not an equal-precision claim or a comparison with llama.cpp/OpenVINO.
It does not measure model loading or complete-command latency.

The experimental FP32 prefill is faster, but its decode is slower than both
PyTorch baselines, and the earlier native incremental run remains failed and
incomplete. FP32 SiLU is therefore not part of the default release integration.
The timing record identifies the experimental binary; timings must not be
silently relabeled as measurements of the separately built installed wheel.

[Full framework comparison](../benchmark/results/silu-gate/pytorch-comparison.json).
Its generic harness records `quality_validated: false`; trained quality is
established separately by the linked quality record for the same binary and
artifacts. Release parity binds the integrated kernel to that tested path.

## Build and correctness

Current run: `build/silu-gate/20261009-063233`. Build passed with 195 Python tests
and 4,907,400 native comparisons, maximum observed difference zero ULP,
`avx2_exercised=1`. These are sampled numerical checks, not a speed measurement.
The first architecture run was blocked by sandbox temporary-directory permissions;
its log is preserved as `quality-sandbox-failure.log`. The same quality stage
was restarted outside the sandbox; no source or executable changed.

Quality then passed all eight reduced-model configurations and trained
TinyLlama checks across 1,016 scored targets. FP32 next-token agreement is
100%, maximum absolute logit error 0.0004682541, and perplexity ratio
0.9999998867. W8A8 agreement is 95.0787% with perplexity ratio 1.0123987262;
its generated text remains different from PyTorch. Both precisions pass
16-token and single-token cache parity. Metrics confirm vector SiLU executes
on eligible prefill and is absent from single-token evaluation. These quality
metrics match the previous release evaluation.

Partial timing, not a qualified result: the first AB pair measured baseline
1956.3019 ms versus candidate 1400.3824 ms prefill; the reversed pair measured
baseline 1920.7667 ms versus candidate 1403.4545 ms. All four passes passed
individual stability. Later passes failed: pair 3 candidate prefill/decode
p90/p10 ratios were 2.543/1.830; pair 4 candidate medians rose to
3064.1343/387.5177 ms; pair 5 baseline prefill spread was 1.251, above 1.25.
The process ended before the final baseline pass, leaving 11 of 12 passes.
No process remained on inspection. The original record is preserved unchanged;
neither its missing pass nor its failed stability checks can be ignored.
This is not a fresh PyTorch comparison.

## Follow-up phase measurements

Two fresh instrumented processes per build and precision, each five measured
iterations plus two warmups, CPU 2 and one thread. These means include warmup
and first use; they attribute work and are not acceptance timings.

| Prefill phase (ms per forward) | Baseline FP32 | Candidate FP32 | Baseline W8A8 | Candidate W8A8 |
|---|---:|---:|---:|---:|
| SiLU/gating | 462.7–538.7 | **20.70–20.70** | 543.5–544.9 | **20.37–20.74** |
| Linear | 1383.3–1592.7 | 1598.2–1655.3 | 687.3–690.1 | 671.6–673.2 |
| Attention | 23.6–27.5 | 27.6–28.7 | 137.8–140.9 | 136.3–136.4 |
| Inclusive forward | 1881.0–2172.2 | 1659.3–1717.6 | 1382.5–1389.4 | 841.4–842.9 |

The activation reduction is about 96% in these profiles. Candidate metrics
report 154 vector calls per process, while baseline reports zero. The gain is
localized to the intended operation; unrelated FP32 linear times also drifted,
so the whole-model percentage must come from separate matched acceptance.
Candidate linear work occupies about 96% of FP32 and 80% of W8A8 prefill;
W8A8 attention is about 16%, supporting the previously planned separate target.

The slow whole-model pass also slowed single-token decode, whose SiLU dispatch
was unchanged. This suggests a broader performance-state or scheduling effect;
wall timings alone do not identify its cause. A short instrumented Windows
diagnostic follows. Its first attempt exposed a harness bug: the diagnostic
runner called a validator that intentionally rejects diagnostic builds from
acceptance. The runner now separately validates diagnostic samples and keeps
the instrumentation marker; acceptance still rejects them. The fix passed
140 focused tests. It also checkpoints each completed diagnostic pass.

Four monitored passes completed after the fix. The first failed decode spread
(1.309): a 303.0158 ms wall sample recorded 203.125 ms thread CPU time, with
cycle counts close to the other decode samples. This is evidence of time not
accounted as executing the inference thread, subject to the Windows CPU-time
counter's coarse granularity; it does not identify the descheduling cause.
Only 1.65–1.73 GB of memory was available, but system paging alone cannot be
attributed to Leaf. No thermal or power-limit sensors were collected.

Added diagnostic-only per-sample process page-fault counts and ran another
four short passes (11 measured samples, three warmups each). All four passed
individual stability. All 44 prefills and 44 decodes reported **zero process
page faults**, even when wall/CPU differences reached about 225 ms for prefill
and 61 ms for decode. Faults therefore do not explain those measured gaps;
this does not rule out faults during the earlier interrupted acceptance run.
These are instrumented observations, not a qualified speedup. The process
counter includes both soft and hard faults. Its schema/guard tests passed
(16 tests); the production baseline/candidate binaries remain unchanged.

Sanitized telemetry: [CPU-time diagnostic](../benchmark/results/silu-gate/environment-summary.json)
and [page-fault diagnostic](../benchmark/results/silu-gate/environment-faults-summary.json).
Raw process inventories remain local under `build/silu-gate/20261009-063233/`.
W8A8 acceptance, the fresh comparison and release integration are complete.
FP32 SiLU remains experimental. Do not reinterpret the interrupted FP32 run as
a passing result or repeat completed correctness checks merely to resume work.

Evidence: [quality](../benchmark/results/silu-gate/quality.json),
[reduced architectures](../benchmark/results/silu-gate/architectures.json),
[interrupted timing](../benchmark/results/silu-gate/timing-32-interrupted.json),
[all four profile records and build manifest](../benchmark/results/silu-gate).

## Why this operation

The [current TinyLlama profiles](windows-tinyllama-current.md) attribute about
25% of FP32 prefill and 39% of W8A8 prefill to activation/gating. Those are
profiled costs, not predicted speedups. Both weight formats produce FP32
intermediate activations, so the same kernel can serve both paths.

The existing gated FFN computes `up *= gate / (1 + exp(-gate))`. The candidate
vectorizes that existing loop; it does not add a new model operation or alter
the projections. Four lanes use double-precision range reduction and a
degree-13 exponential polynomial, then round to float before the existing
float addition, division and multiplication. Groups with extreme or nonfinite
inputs use the original scalar expression. There is no approximate reciprocal
or global fast-math change.

`LEAF_EXPERIMENTAL_SILU_GATE` enables the broader candidate at build time. Runtime
dispatch requires the existing AVX2/FMA capability check, SiLU activation and
at least eight tokens. Single-token decode, non-gated FFNs, other activations
and forced-scalar execution retain their existing paths. The implementation
is GNU/Clang x86-64 specific, with a portable scalar fallback.

The native metrics identify the candidate build and count calls to the vector
gating function. Quality and timing checks require these markers, not just
the compiler flag. The experiment cannot qualify automatic precision selection.

## Reproduce the original experiment in separate stages

Use PowerShell from the repository with the existing development environment,
GNU C++ compiler and cached TinyLlama artifacts. Each stage preserves a log and
stops on failure. Do not overwrite failed or interrupted measurement records.
The manifest binds the builds to the sources; source changes require a new run.

### 1. Build and primitive correctness

```powershell
cd C:\Users\navaneeth\leaf
$run = "build/silu-gate/$(Get-Date -Format yyyyMMdd-HHmmss)"
.\scripts\test_silu_gate.ps1 -Stage Build -RunDirectory $run
```

This builds conservative-SiLU and candidate decoders from the same source, runs the native
SiLU test, and runs focused Python checks for quality binding, experimental
selection policy and packaging. Native tests cover dense/random inputs, extreme
values, signed zero, NaNs/infinities, tails, unaligned buffers, in-place aliasing,
guard regions and scalar fallback. The declared ordinary-value bound is three
ULPs against scalar math; special-value classification and signed zero must
match. This is a tested bound, not a universal correctly-rounded proof.

Share the `SiLU gating correctness passed` line (including `max_ulp` and
`avx2_exercised`), pytest summary and `PASS Build`, or the first failure.
Keep the printed run directory. A Windows AVX2 experiment requires
`avx2_exercised=1`; a scalar-only success cannot establish SIMD correctness.

### 2. Model quality

After reviewing stage 1, set the local model snapshot path and run:

```powershell
$model = 'PATH_TO_CACHED_TinyLlama-1.1B-Chat-v1.0'
.\scripts\test_silu_gate.ps1 -Stage Quality -RunDirectory $run -ModelPath $model
```

The reduced architecture suite checks reference/scalar parity, threading,
cache behavior and generation across supported families. Trained TinyLlama
checks reuse `build/tinyllama_windows_fair` without re-exporting weights.
They check all held-out logits, the existing FP32/W8A8 quality thresholds,
16-token chunks, single-token cache parity and generation. W8A8 generation
agreement is reported separately; it is not required to be identical to FP32.
Single-token evaluation must report zero vector SiLU calls. No tolerances are
relaxed for this candidate. Quality records bind the executable and inputs by hash.

Share `architectures.json`, `quality.json` and the final stage status. This stage
does not establish speed or stability.

### 3. Whole-model timing

Only after quality passes, leave the laptop plugged in and free of heavy work.
Run one command, share the result, then run the other precision separately:

```powershell
.\scripts\test_silu_gate.ps1 -Stage Timing32 -RunDirectory $run
# Run separately after reviewing the FP32 result:
.\scripts\test_silu_gate.ps1 -Stage Timing8 -RunDirectory $run
```

Each uses six alternating AB/BA pairs, 31 samples and ten warmups per process,
one thread pinned to logical CPU 2, Above Normal priority, and the same frozen
63-token prefill plus one-token decode workload. These runs can take many minutes.
Conservative and candidate processes run serially. No outliers are discarded.

The existing incremental gate requires pooled candidate/baseline prefill
ratio <= 0.98, decode ratio <= 1.02, and individual plus pooled p90/p10 spread
<= 1.25. Median paired ratios and every raw sample are also retained. A failed
gate remains a failed gate; the script must not print `PASS` merely because the
benchmark process completed. Share `timing-32.json` or `timing-8.json` and its log.
Interpret numerical quality, speed and measurement stability separately.

## After the results

Use a separate diagnostic profile to confirm activation cost changed; do not
mix profiling with acceptance timing. If the candidate qualifies, run a fresh
matched PyTorch comparison before claiming a framework speedup. FP32 and W8A8
need separate conclusions, and W8A8 retains its documented quality tradeoff.
For Windows W8A8, these steps and release checks are now complete; the README
and Mermaid diagram show the integrated default. FP32 remains opt-in.

Do not restart GPT-2 GEMM/packing experiments or widen this change to quantized
attention without evidence. That follow-up remains a separate experiment.
