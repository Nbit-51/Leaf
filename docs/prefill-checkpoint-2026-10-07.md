# Prefill investigation checkpoint — 2026-10-07 (Windows, GPT-2 FP32)

Durable handoff note. Baseline is the shipped Windows FP32 default
(`build/fp32-release/default/leaf_decoder.exe`, commit `5163075`).

## CI

`5163075` failed only on macOS-14: `weight-panel vector output is not bit-exact
against full-K`. Cause (confirmed via AArch64 assembly): Clang vectorizes the
contiguous weight-panel portable loop with strict in-order, *unfused*
reductions, while contracting the strided token-panel portable loop to
`fmadd`. Same order, different rounding. Fix `18b5bcc` compiles the
token-panel test with `-ffp-contract=off`; production numerics are unchanged.
CI is green on Linux, Windows and macOS for `18b5bcc`.

## Step 1–2: is there MLP GEMM headroom? (Hypothesis contradicted)

Hot kernel: `gemm_token_panels_f32_row_reuse` → `token_panel_detail::tile<6>`.
GCC 15.2 inner loop per K step: 2 contiguous input loads, 6 weight
broadcasts, 12 FMAs, 15 YMM registers, no spills.

Measured on CPU 2, Above Normal, one thread (records in
`benchmark/results/mlp-headroom/`):

| Shape (M=63) | Leaf row-reuse, 12 rotating weights | PyTorch 2.12 `F.linear`, 12 rotating |
|---|---:|---:|
| MLP up N3072 K768 | 3.38 / 2.98 ms (two runs) | 4.00 ms |
| MLP down N768 K3072 | 3.37 / 3.55 ms | 3.36 ms (passes 4.01, 3.24) |

`gemm_ceiling_benchmark.cpp` measures independent register-only 256-bit FMA
throughput adjacent to each case. The shipped kernel reaches 0.90–1.09 of
that measured ceiling on both MLP shapes, hot or rotating (the ceiling itself
varies about ±10% with frequency state). No nominal clock is assumed.

Conclusion: these measurements are consistent with little FMA-throughput
headroom at this instruction count; they do not prove a hardware ceiling for
every clock state or shape. Isolated MKL did not show a consistent advantage.
Persistent output-channel weight packing keeps
the same 2-load/6-broadcast/12-FMA mix and would add ~216 MiB, so it was
**not implemented**.

## Profile of the shipped default (`LEAF_DECODER_PROFILE=1`, 25 forwards)

Per prefill forward: linear 145.6 ms, attention 19.1 ms, embedding 3.3 ms,
layer norm 2.8 ms, activation 2.2 ms. Attention is ~12 MFLOP/layer but costs
~1.6 ms/layer — far below GEMM throughput, so it is the better-supported
next target.

## Predeclared candidate (written before measurement)

Candidate `attn-ilp`: changes only instruction scheduling, never FP order.

1. Hoist the position-embedding tensor lookup out of the per-element
   embedding loop (it currently builds a `std::string` and hashes it 63×768
   times per prefill).
2. Vector prefill attention: compute Q·K for four unmasked keys at once
   (each key keeps its own accumulator, chunk order and in-order horizontal
   sum), and keep P·V output accumulators in registers across keys in
   64-dimension blocks (same key order, separate multiply then add).

Gates:
- Native test: new kernel output `memcmp`-equal to the previous kernel
  (kept in the test as a reference) across dims, GQA, masks, polarity, NaN.
- Trained GPT-2: logits bit-identical to the shipped default on all held-out
  targets, chunked cache and greedy generation.
- Whole model, fresh serial balanced AB/BA, one thread on CPU 2, Above
  Normal, 10 warmups, 31 samples: prefill ratio ≤ 0.98, decode ratio ≤ 1.02,
  existing stability checks pass. Otherwise not shipped.

## Resumed validation and second candidate

The rebuilt ILP candidate is bit-identical on full logits, chunked cache and
generation at one and two threads. Six balanced pairs measured 159.7478 →
156.5862 ms prefill (1.979% lower) and 28.43845 → 28.54515 ms decode.
Three passes failed decode spread; the prefill ratio also narrowly missed
0.98. **Not qualified.** All samples and the candidate patch are retained in
`benchmark/results/attention_ilp_v1_windows.json` and
`benchmark/experiments/attention-ilp-v1.patch`.

Paired profiles isolate embedding at 3.59–3.90 → 0.11–0.12 ms and attention
at 18.39–19.43 → 17.17–17.42 ms. Instrumenting attention stages finds about
13.2 ms in exponentials/normalization, versus 2.9 ms in score calculation.
Only 66 probabilities are subnormal, all in one layer: this does not explain
the cost across all layers. MinGW disassembly shows `expf` calls double
`exp`, whose main path uses x87 and changes/restores its rounding control.

Next candidate, predeclared before measurement: vectorize four softmax
exponentials using double-precision range reduction and a degree-13 Taylor
polynomial on the reduced interval. Use scalar `std::exp` outside [-80, 0]
and for exceptional inputs, retaining underflow and NaN semantics. Preserve
score subtraction and sequential denominator accumulation. No global fast-math,
flush-to-zero, or artifact changes. Primitive gate: at most one FP32 ULP from
scalar exp on a broad deterministic sweep, plus exceptional/tail checks.
Trained-model gates remain FP32 reference allclose (atol/rtol 2e-3), >=99.9%
next-token agreement, perplexity ratio <=1.001, cached/generation parity.
Report bit identity when observed, but do not assume a polynomial is universally
bit-identical to every platform's libm. Whole-model timing gates stay unchanged.

## End-of-session handoff (local work, do not restart experiments)

User requested stopping after one final confirmation and resuming in the
morning. We agreed to keep work local tonight; do not push or merge this session.
Branch remains `diag/core-scaling`; last committed checkpoint is `249c353`.
The runtime/test/tool edits below are local and uncommitted. The final
confirmation outcome is recorded below when complete.

Completed for the softmax candidate:

- Normal build: `build/attn-softmax/candidate/leaf_decoder.exe`.
- Native attention test: 1,052,672 sampled exp arguments, maximum observed
  FP32 ULP error **0**. GQA, broadcast masks, mask polarity, key tails,
  vector output tails, scalar fallback and nonfinite cases pass.
- Trained GPT-2: bit-identical full logits and chunked logits against the
  shipped baseline at threads 1 and 2; greedy tokens match both baseline
  and saved PyTorch generation. Record: `build/attn-softmax/identity.json`.
  Exact equality is to Leaf's baseline, not to PyTorch floating-point logits.
- Eight reduced architecture cases, including scalar and quantized paths,
  pass exact paired regression: `build/attn-softmax/architectures.json`.
- Full suite: **831 passed, zero skipped, four existing ONNX warnings**,
  50.86 seconds. Log: `build/attn-softmax/pytest.log`.
- First six-pair softmax run: prefill 158.7654 → 144.0173 ms (**9.29% lower**),
  decode 27.98995 → 28.08685 ms (+0.35%). All six prefill pairs improve.
  Both pooled phases are stable, but candidate pair 3 fails decode spread.
  Strict gate **fails**; retain this run, do not replace it with confirmation.
  Record: `build/attn-softmax/acceptance.json`.

Final confirmation was declared before execution after the user confirmed
plugged-in/no-heavy-work conditions. It uses 101 samples rather than 31,
six balanced pairs, ten warmups, CPU 2, one thread and Above Normal priority.
No builds, correctness tests or PyTorch runs overlap its timing window.
Thresholds remain prefill <=0.98, decode <=1.02, and every pass plus both
pooled phases stable at p90/p10 <=1.25. Every sample is retained.

```powershell
python tools/benchmark_default_abba.py --before build/fp32-release/default/leaf_decoder.exe --after build/attn-softmax/candidate/leaf_decoder.exe --workdir build/gpt2_prefill_followup --output build/attn-softmax/confirmation-101.json --pairs 6 --runs 101 --warmup 10
```

Resume order:

1. Read this checkpoint, `git status`, and the final result. Preserve local
   changes and completed validation; do not repeat the old GEMM packing or
   96-row scheduling experiments.
2. Use the final outcome to decide qualification, retaining the earlier
   failure in any claim. If qualified, run the fresh balanced PyTorch eager/
   SDPA comparison on Windows with `--default-kernels`. No new stable PyTorch
   parity/speed claim has been established for this candidate yet.
3. Complete wheel build/install smoke checks for the changed runtime. The
   old installed wheel does not contain this candidate. Confirm primitive
   CI coverage on all three OSes after pushing; current local changes have
   not run on remote CI.
4. Update README results and the existing Mermaid diagram with the accepted
   execution path. Commit/push `diag/core-scaling`, review required CI, then
   merge into main as requested earlier, once the work is complete.

The diagnostic-only stage-counter override is preserved locally at
`build/attn-ilp/diagnostic-include/leaf/kernels/attention_vector.h`; outputs are
`build/attn-ilp/attention-stages.txt` and `probability-counts.txt`. It is not
compiled into the candidate. The original ILP-only patch and failed result
are retained under `benchmark/experiments/` and `benchmark/results/`.

## Final outcome — stop here tonight

The 101-sample confirmation PASSED the unchanged gate. All twelve passes and
both pooled phases are stable. Pooled medians across 606 samples per binary:

| Phase | Shipped baseline | Softmax candidate | Change |
|---|---:|---:|---:|
| Prefill | 161.1558 ms | 145.4869 ms | 9.72% lower |
| Decode | 28.64235 ms | 28.48115 ms | 0.56% lower |

Median paired ratios: prefill 0.90221171; decode 0.99025637.
Retain the earlier 31-sample run's failed individual decode spread alongside
this passing confirmation. This qualifies the local incremental native
comparison on this workload; it does not establish a fresh PyTorch speed win.

Durable records are in benchmark/results/attention_softmax_{initial_windows,
confirmation_windows,identity,architectures}.json. All code changes remain
local/uncommitted on diag/core-scaling; nothing was pushed or merged tonight.
No benchmark processes remain running. README/diagram updates, fresh PyTorch
comparison, changed-runtime wheel verification, remote CI and merge remain
for the morning. Do not repeat completed full tests or exact-parity checks
unless source changes warrant it.

## Resumed 2026-10-08

The user resumed work and confirmed plugged-in/no-heavy-work conditions.
The overnight pause is over. New wheel build/install and offline first/cached
generation checks pass, with the default optimized FP32 runtime active.
Installed/source full logits, chunked logits and generation are bit-identical
at one and two threads.

Fresh installed-wheel/PyTorch forward-reverse comparison (101 samples and ten
warmups per pass, CPU 2, one thread, Above Normal): Leaf 144.30735/28.11020 ms,
eager 153.51550/28.68440 ms, SDPA 151.66305/28.57335 ms (prefill/decode).
All three runtimes pass individual, pooled and between-pass stability.
The **4.85% prefill reduction versus the fastest measured PyTorch** qualifies;
decode is 1.62% lower, passing non-regression but not a separate 2% improvement
threshold. Evidence, exact hashes and scope are in
`benchmark/results/attention_softmax_validation.json` and the new report
`docs/windows-attention-softmax.md`. README and Mermaid now reflect the path.
Next delivery steps: commit/push the current branch, inspect CI, create/review
the PR and merge to main under the user's earlier instructions.

## TinyLlama follow-up — resumed after usage interruption, 2026-10-08

GPT-2 delivery is committed and pushed as `0d28693`; PR #2 is open and all
Windows/Linux/macOS checks passed. Merge was held for the user's requested
second-model validation. Do not repeat the completed GPT-2 acceptance runs.

The cached model is TinyLlama-1.1B-Chat-v1.0. Its original source snapshot is
`C:/Users/navaneeth/Documents/Vs code projketcs/Imporved_Hydra/Hydra_Engine/TinyLlama-1.1B-Chat-v1.0-git`;
the frozen validation artifacts are in `build/tinyllama_windows_fair`.
The installed executable is `build/attn-softmax/installed/leaf/bin/leaf_decoder.exe`.

`build/attn-softmax/tinyllama-quality.json` completed successfully before the
interruption. Both FP32 and smoothed W8A8 pass their existing quality gates
over 1,016 scored targets and chunked-cache parity. FP32 generation matches
PyTorch exactly. W8A8 has 95.0787% next-token agreement, perplexity ratio
1.0123987, and different generated tokens; it is a separate quality tradeoff.

On resume, no benchmark processes were running and fresh timing had not begun.
The declared timing run is `build/attn-softmax/tinyllama-framework.json`:
31 measured iterations plus ten warmups per pass; CPU 2, one thread, Above
Normal priority; shipped defaults; Leaf FP32, Leaf W8A8, eager, SDPA, then
reverse order. Preserve every sample and the unchanged stability gates.
The harness checkpoints each completed pass. Do not overwrite partial records.

After timing, profile the shipped default of GPT-2 FP32, TinyLlama FP32 and
TinyLlama W8A8 separately, without overlapping performance measurements.
`tools/profile_decoder.py --default-only --activation-bits {32,8}` now supports
that diagnostic workload. Update results/README, push on `diag/core-scaling`,
and merge PR #2 only after final-revision CI succeeds.

### Completed resumed tests — do not repeat

All eight TinyLlama passes completed and are preserved in
`benchmark/results/tinyllama_current_pytorch_windows.json`. Pooled medians
(prefill/decode): Leaf FP32 1944.3069/225.5490 ms; W8A8 1381.9081/92.70555 ms;
PyTorch eager 1800.6946/213.2119 ms; SDPA 1881.60295/243.57595 ms.
Leaf FP32 and eager pass stability. W8A8 fails individual and pooled spread;
SDPA fails pooled decode. This is not a new qualified TinyLlama speedup.
All trained quality checks pass, with W8A8's non-identical generation retained.

Two separate post-timing profiles per configuration are complete. TinyLlama
SiLU/gating consumes 25% of FP32 prefill and 39% of W8A8 prefill; W8A8 attention
adds 10%. Next kernel experiment: vector SiLU/gating, then separately consider
vector attention dispatch for quantized artifacts. Do not rerun the previous
GPT-2 packing experiments without evidence from the relevant projection sizes.
If timing spread persists, measure per-sample CPU/wall time and fault deltas;
these wall-only samples do not establish the cause.

Evidence, exact hashes, all failed gates and reproduction commands are linked
from `docs/windows-tinyllama-current.md`. Focused regression suite: 251 passed.
No runtime kernel changed in this second-model follow-up. The accepted Windows
default stays enabled; no automatic precision profile was promoted. Only Git
delivery and final-revision CI remain after this checkpoint.

## Delivery and working arrangement — 2026-10-09

PR #2 merged to main at `f08746f` after all six Windows/Linux/macOS checks
passed. The README cleanup and organic Leaf logo follow on a separate
documentation branch; the detailed README is preserved as
`docs/project-reference.md` with a documentation index in `docs/README.md`.

For future work, the user prefers to run long benchmarks/tests locally.
Focus assistant effort on implementation, architecture, documentation and
interpreting results. Provide copy-paste commands with a fresh named output
file, the relevant environment assumptions, and exactly which outputs to share.
Reuse completed evidence. Take over execution if the user asks or cannot run it.
Do not start another long benchmark merely to fill a waiting interval.

## SiLU/gating implementation — 2026-10-09

Documentation/logo PR #3 merged as `8260fc7`; the completed diagnostic and
Codex branches were removed after merging. Current local work continues on main.

The next candidate is implemented under `LEAF_EXPERIMENTAL_SILU_GATE`, off by
default: vectorize the existing gated-SiLU loop for AVX2/FMA prefill of at least
eight tokens, across FP32 and quantized weights. Single-token decode and other
activations retain their existing dispatch. Added primitive tests, dispatch
metrics, trained quality checks with tokenwise cache parity, quality-bound
FP32/W8A8 AB/BA timing, packaging/CI wiring and a staged PowerShell runner.

The user subsequently authorized assistant execution this turn. Build passed
in `build/silu-gate/20261009-063233`: 195 focused Python tests; 4,907,400 native
comparisons with maximum observed difference zero ULP and AVX2 exercised.
The initial Quality stage failed on sandbox temporary-directory permissions
before model evaluation. Its log is preserved as `quality-sandbox-failure.log`;
the same stage was restarted outside the sandbox without changing sources or
executables. Quality then passed all eight reduced configurations and trained
TinyLlama FP32/W8A8 checks across 1,016 scored targets, including chunk16 and
tokenwise cache parity. Next-token agreement and perplexity ratio match the
previous release. FP32 AB/BA timing is running with output `timing-32.json`;
do not start another timing process concurrently. W8A8 timing and diagnostic
phase attribution remain after this run.
Use [the runbook](silu-gate-experiment.md) and `scripts/test_silu_gate.ps1`.
Do not repeat Build or enable the default before reviewing actual results.

### Usage-limit handoff during FP32 timing

User reports only 10% usage remaining and asks to hurry. Do not start W8A8
timing, profiling or another long experiment this turn. The existing six-pair
FP32 run saves each completed pass and writes its final gate automatically to
`build/silu-gate/20261009-063233/timing-32.json`, with transcript `timing32.log`.
It is still running at this checkpoint (exec session 43919). Read its record
before starting or repeating anything; `complete: false` is not acceptance.

First two complete pairs, all individual passes stable:
- AB: baseline 1956.3019/227.6221 ms, candidate 1400.3824/224.8902 ms.
- BA: candidate 1403.4545/223.4061 ms, baseline 1920.7667/224.7545 ms.

These suggest roughly 27-28% lower prefill, with unchanged decode. Full pooled
and per-pass gates remain pending; no fresh PyTorch comparison has been run.
Next turn, inspect the final gate, then do separate phase attribution and W8A8
timing as warranted. Do not rebuild the matching binaries or repeat the passed
195 tests, primitive checks or trained quality unless sources change.

## Continued SiLU diagnosis — 2026-10-09

The old FP32 process is gone. Its record contains 11/12 passes, with candidate
pair 3 failing prefill/decode spread, candidate pair 4 at 3064/388 ms, and
baseline pair 5 narrowly failing prefill spread. Do not describe it as still
running or accepted. Preserved verbatim under
`benchmark/results/silu-gate/timing-32-interrupted.json`.

All four phase profiles completed (two fresh processes per build/precision).
SiLU/gating drops from 463–539 ms to 20.70 ms in FP32 and from 543–545 ms to
20.37–20.74 ms in W8A8: about 96% lower activation cost in these profiles.
Candidate linear costs now occupy about 96%/80% of FP32/W8A8 prefill.
Quality, architecture checks, four profiles and build manifest are copied to
`benchmark/results/silu-gate/`; production executables are unchanged.

User confirmed plugged in/no heavy work. An instrumented diagnostic exposed
a harness bug: its validator rejected diagnostic builds. Fixed only the
diagnostic tool to validate samples separately, preserve the instrumentation
marker, and checkpoint each completed pass. Acceptance tools still reject
instrumented builds. Focused diagnostics/comparison tests: 140 passed.

Four diagnostic passes then completed; the first failed decode spread at
1.309. One decode was 303.0158 ms wall vs 203.125 ms thread CPU time. Available
memory was 1.65–1.73 GB, with some system-wide paging; this does not attribute
faults to Leaf. Sanitized record: `environment-summary.json` in the evidence
directory. Raw process telemetry stays under ignored build/.

Added per-sample process page-fault deltas to diagnostic builds only (includes
soft and hard faults; not a hard-fault counter). Counter schema version 2;
diagnostic contract tests now 16 passed. New diagnostic binary lives under
`build/silu-gate/20261009-063233/diagnostic-faults/`. Four short monitored passes
are running, output `diagnostic-environment-faults.json` (exec session 82031).
Wait for completion; no concurrent benchmark. W8A8 acceptance and fresh
PyTorch comparison have not been started. Keep the kernel opt-in until the
remaining performance decision is supported by evidence.

### Completed diagnostic follow-up

The page-fault diagnostic is complete; no test/benchmark remains running.
All four short passes stable, with zero process faults across all 44 measured
prefills and 44 decodes. Maximum observed wall-minus-thread-CPU gaps still
reached about 225/61 ms. Thus faults do not explain those particular gaps;
the earlier 3-second acceptance pass is not explained by this later run.
Windows thread CPU-time accounting is coarse; do not label an exact scheduler
or thermal cause without additional evidence. Public aggregate telemetry is
`benchmark/results/silu-gate/environment-faults-summary.json`; raw process
inventories stay local in the ignored build directory.

Hand the long W8A8 acceptance run to the user's terminal per their preference:
`./scripts/test_silu_gate.ps1 -Stage Timing8 -RunDirectory build/silu-gate/20261009-063233`.
It uses existing validated binaries and quality, no rebuild. Preserve the
original FP32 failure. Kernel remains opt-in; no new default or Git push yet.

## W8A8 acceptance completed by user — 2026-10-09

`Timing8` completed all six AB/BA pairs and passed every gate. Verified the
saved record and all input hashes. Evidence copied without alteration to
`benchmark/results/silu-gate/timing-8.json`. Pooled prefill: baseline
1320.3094 ms, candidate 819.3776 ms (37.9405% lower; 1.61136×). Decode:
88.1954 → 88.3723 ms (+0.2006%, passing non-regression). All 12 individual
passes and pooled phase stability checks pass. 186 samples/build/phase.

Do not rerun this acceptance or completed correctness checks without a
relevant source/workload change. This is a qualified incremental W8A8 win,
not yet a fresh PyTorch comparison or FP32 acceptance. README and experiment
report now record it. Default remains unchanged. Next: choose the qualified
release configuration, obtain its fresh framework comparison, then integrate
and validate the release; preserve the interrupted FP32 failure throughout.

## SiLU release complete locally — 2026-10-10

The interrupted chat did not interrupt the release pipeline: all stages finished
on October 9. Saved results and 21 input hashes were checked on resumption.
Do not rerun completed acceptance or correctness checks just to resume work.

- Fresh matched candidate comparison, one thread/CPU 2, 63-token prefill and one
  cached decode: Leaf W8A8 861.59695/91.46975 ms; PyTorch SDPA FP32
  1767.87970/211.42825 ms; eager FP32 1804.02280/212.68855 ms. All stability
  checks pass. W8A8 versus SDPA is 2.052×/2.311×, with the existing 95.08%
  agreement, +1.24% perplexity and non-identical-generation tradeoff.
- Windows default enables vector SiLU for eligible W8A8 prefills only. FP32
  SiLU stays experimental; its failed/incomplete incremental record is retained.
- 870 Python tests, nine reduced architecture cases, ten trained TinyLlama exact
  native parity checks and installed-package offline checks passed. Installed
  W8A8 matches the accepted candidate; FP32 matches the old baseline.
- All 14 packaged native source files match the workspace. The installed wheel
  was not separately timed or published to PyPI. Exact parity is correctness
  evidence, not a new latency measurement.

Local run: `build/silu-gate/release-20261009`. Public records:
`benchmark/results/silu-gate/{pytorch-comparison,release-validation,release-parity,release-architectures,release-package}.json`.
Experimental source/evidence checkpoint: `5e5246f`. Release implementation,
README and Mermaid updates are ready for Git delivery on main; verify Git/CI
state rather than assuming publication from this checkpoint.

Next performance experiment: separately investigate quantized-path attention
(about 16% of measured W8A8 prefill); linear work remains about 80%. Preserve
the shipped SiLU default and frozen baseline, profile the specific path, then
apply the same quality and incremental acceptance gates. Avoid reopening old
GPT-2 GEMM packing hypotheses without new evidence. CPU-runtime competitor
comparisons and Linux performance remain separate outstanding work.
