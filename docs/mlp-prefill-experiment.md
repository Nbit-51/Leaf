# MLP prefill experiment — 2026-10-07

**Current verdict: below the whole-model acceptance target.** Vector packing
passes correctness and improves the two isolated MLP shapes. Normal-priority
ABBA is inconclusive because stability fails; a separately scoped Above Normal
follow-up passes stability and observes a 1.46% reduction, below the unchanged
2% target. Keep the preceding paired-row GEMV build as the accepted baseline.
The new candidates do not enable a default.

## Predeclared candidate

Baseline: commit `c002555`, packed FP32 GEMM with row reuse and paired-row GEMV.
Windows only, one thread pinned to CPU 2. Preserve all preceding experiments.

Hypothesis: for long MLP projections, keeping six output rows but visiting
128-column segments across up to four token panels can reuse weight segments
and bound the active input working set. The current row-reuse kernel finishes
all columns for one panel before moving to the next. This differs from the
earlier blocked-K candidate, which reused inputs across up to 96 output rows
inside one panel. No hardware cache-miss cause is established.

The candidate reuses the existing accumulation primitive, retains increasing-K
FMA order and applies bias once. Scratch is bounded at 1.5 KiB per worker call.
No new persistent weight layout or model-specific dispatch is introduced.

First test parity, tails, strides, scalar fallback and row slices. Then compare
packing-inclusive shape timings against **row reuse**, using 12 rotating weight
replicas, serial ABBA, 31 samples and ten warmups per pass. Record all shapes;
MLP up/down are the target. Do not claim an isolated gain from unstable samples.
Only a promising candidate proceeds to opt-in integration and trained quality,
cache/generation checks and whole-model ABBA. The objective is prefill: ratio
<= 0.98, both phases <= 1.02, p90/p10 and between-pass drift <= 1.25.
Keep paired-row decode enabled in both builds. A native gain requires a separate
fresh matched framework comparison before a PyTorch superiority claim.

## First result and separate packing candidate

The panel-blocking shape attempt is retained as
`gpt2_mlp_panel_rotating_windows.json`. All shapes fail combined stability;
the MLP-up baseline pass medians shift from 4.8004 to 2.6307 ms. Its pooled
16.80% apparent reduction is not accepted. MLP down observes a 5.68% increase.
This candidate remains outside decoder dispatch.

The second candidate changes only input packing: use an AVX2 8x8 transpose
for complete tiles, retaining scalar tails and exactly the existing packed
layout. MLP GEMM remains the current row-reuse kernel, with unchanged floating
point arithmetic. Compare packing-inclusive timings to row reuse with scalar
packing before deciding on integration. Apply the same quality/speed/stability
gates; no panel-blocking change is included in this candidate.

## Vector-packing results

Packing-inclusive synthetic timings, M=63, twelve rotating resident weight
replicas, one thread on Windows CPU 2, ten warmups and 31 samples per ABBA pass:

| Projection N x K | Row reuse, ms | Vector packing, ms | Change | Stable? |
|---|---:|---:|---:|---|
| Square 768 x 768 | 0.85345 | 0.86205 | +1.01% | No |
| MLP up 3072 x 768 | 3.49748 | 3.45923 | -1.09% | Yes |
| MLP down 768 x 3072 | 3.41800 | 3.32063 | -2.85% | Yes |
| Prospective fused QKV 2304 x 768 | 2.57738 | 2.55833 | -0.74% | Yes |

[Raw shape record](../benchmark/results/gpt2_mlp_pack_rotating_windows.json).
These are per-matrix means across twelve calls, not whole-model latency. For
compatibility the harness retains the `full_k` stage label; the explicit
`baseline_kernel: row_reuse` field identifies the actual baseline for both new
candidates. Earlier records without that field use the original full-K kernel.

The integration selects vector packing only for MLP up, gate and down
projections in the existing packed FP32 path. It preserves the packed layout,
GEMM arithmetic, activation, output storage and one-token decode dispatch.
It requires `LEAF_EXPERIMENTAL_MLP_PACK` at build time and the existing runtime
float-tiles opt-in; the build emits an experimental marker and is excluded
from automatic selection. Dispatch describes the operation, not a model name.

## Whole-model ABBA and failed stability

The [native ABBA](../benchmark/results/gpt2_mlp_pack_windows_abba.json) compares
the frozen paired-row GEMV binary to the same experimental stack plus MLP
packing. It reuses the preceding immutable quality record and freshly checks
candidate held-out quality, chunked cache and exact generation. Both builds
use one thread, CPU 2, 31 measured runs and ten warmups per pass. Timing runs
are serial and do not overlap builds or tests. The user confirmed mains power
and no deliberate heavy workload.

| Phase | Before median, ms | After median, ms | Observed change |
|---|---:|---:|---:|
| Prefill | 167.3237 | 163.20865 | -2.46% |
| Cached one-token decode | 28.1297 | 27.9255 | -0.73% |

The improvement and regression limits pass, but stability fails:

- First AFTER prefill p90/p10 is **1.250534**, exceeding 1.25.
- Final BEFORE decode p90/p10 is **1.431357**, exceeding 1.25.
- Pooled BEFORE decode p90/p10 is **1.305585**, also exceeding 1.25.

The first failure is small, but the limit is unchanged; the baseline decode
failure is substantial. Preserve all samples and the failed verdict. There is
no repeat selected to replace it. The observed 4.12 ms prefill reduction is
**not an accepted whole-model speedup**. No fresh PyTorch comparison or gap
claim is made for this candidate. The earlier 154.59 ms PyTorch measurement
cannot be subtracted from this session's result to claim a new gap.

## Validation and checkpoint

- 802 Python tests pass, six skip, with four existing ONNX warnings.
  After the diagnostic and priority controls, the final suite is **813 passed,
  six skipped**, with the same four warnings.
- Native token-panel checks pass, including bit-exact packed layouts,
  vector/scalar paths, padding, strides, dimensions crossing panel/K boundaries,
  independent row slices, output guards, input preservation and invalid extents.
- [Default regression](../benchmark/results/mlp_pack_default_regression.json):
  all eight architecture cases are bit-exact against the previous default.
- [Experimental architectures](../benchmark/results/mlp_pack_architectures.json):
  all eight cases pass, including scalar, two threads, quantization and cache.
- Trained GPT-2: allclose passes, 100% agreement on 1,016 next-token targets,
  perplexity 62.58004233 versus reference 62.58010265; chunked parity and exact
  generation against both PyTorch and the before build pass.
- [Validation record](../benchmark/results/mlp_prefill_validation_windows.json)
  binds source/binary hashes and replays the failed gate from raw samples.
  Builds used Windows MinGW g++; CMake was unavailable and not exercised.
  No Linux/WSL evaluation was performed.

Reproduce the candidate with `scripts/build_decoder.ps1 -ExperimentalVectorGelu
-ExperimentalAttentionAvx2 -ExperimentalRowReuse -ExperimentalGemvPair
-ExperimentalMlpPack`. Set `LEAF_EXPERIMENTAL_FLOAT_TILES=1`. Compare against
`build/decode-gemv/candidate/leaf_decoder.exe` using
`tools/benchmark_decoder_comparison.py --objective prefill`, the immutable
`gpt2_gemv_pair_matched_windows.json` record, `build/gpt2_prefill_followup`,
`--keys 32 --cpu 2 --runs 31 --warmup 10`, and a new output path.

The follow-up below holds this candidate fixed while separating packing cost
and timing disturbances. No second kernel change is combined with that run.

## Follow-up attribution and next kernel

The instrumented before/after ABBA separates MLP packing from the enclosing
linear call and records thread CPU time/cycles for each inference. Mean MLP
packing falls from 2.0930 to 1.2434 ms per prefill, a 0.8496 ms saving; enclosing
MLP linear time is about 89 ms. These profiled values include warmups and are
diagnostic, not acceptance timings. Large wall-time outliers occur with near
normal cycle counts in both builds. Thread accounting is coarse (15.625 ms
increments here); cycles are not converted to elapsed time. This supports
substantial off-CPU delay, without identifying the specific scheduler/wait cause.
Windows refused the ETW CPU trace for lack of system-profiling privilege.

[Per-inference and component data](../benchmark/results/mlp_cost_diagnostic_windows.json)
includes a BEFORE decode at 158.4276 ms wall / 31.25 ms accounted CPU time /
67,063,910 cycles, versus that pass's median 28.2696 ms / 62,857,265 cycles.
An AFTER decode takes 118.4263 ms / 15.625 ms CPU time / 64,487,574 cycles,
versus its pass median 29.3883 ms / 62,851,739 cycles. This separates major
off-CPU delays from extra arithmetic; it does not identify the process or
kernel wait responsible, or explain every fluctuation. CPU-time accounting
is visibly quantized and can exceed a short wall interval; do not treat its
per-sample subtraction as precise wait duration. Microsoft likewise cautions
against [converting thread cycles to elapsed time](https://learn.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-querythreadcycletime).

The [host telemetry summary](../benchmark/results/mlp_pack_environment_windows.json)
records four stable passes, no sampled activity on sibling CPU 3, processor
performance about 145–150% while the child is busy, and no meaningful HighQoS
advantage. It includes load/warmup and cannot locate short stalls. Raw process
inventories stay local under `build/mlp-panel/environment-raw/`.

A separately recorded uninstrumented Above Normal priority ABBA gives both
timed child processes the same verified priority. All stability gates pass;
prefill is 1.46% lower, below the unchanged 2% target. This is a scoped follow-up,
not a replacement for the failed normal-priority run or a global system change.

[Controlled raw comparison](../benchmark/results/gpt2_mlp_pack_above_normal_windows_abba.json):
prefill 160.72605 → 158.3804 ms; decode 27.3521 → 26.9472 ms. All quality and
stability checks pass; `prefill_improvement_passed` is false. This makes the
decision more specific than the initial failure: a small packing optimization,
not the full gain suggested by the noisy run. The priority follow-up supports
using a controlled scheduling condition for further experiments but is not an
order-balanced proof that priority alone removed the earlier stalls. No
ordinary-priority or PyTorch superiority claim follows from it.

Next isolated kernel: combine the existing 3x32 wide-token microtile with row
reuse. The earlier wide candidate traversed all weights per token panel; the
new schedule reuses three weight rows across panels. At K=3072, that row group
is 36 KiB versus 72 KiB for the current six-row group. The hypothesis concerns
weight reuse, not a proven cache-miss bottleneck. Keep existing wide input
packing, increasing-K arithmetic, bias and scalar fallback. Compare to the
accepted 6x16 row-reuse kernel with packing included, twelve weight replicas,
31 samples/ten warmups, CPU 2 and Above Normal priority. Do not combine the
vector-packing candidate. Require native parity and promising stable MLP-shape
results before whole-model integration.

The [wide row-reuse shape result](../benchmark/results/gpt2_mlp_wide_reuse_rotating_windows.json)
does not justify integration. Up observes a ratio of 0.98185 and down 1.48931;
both fail stability. Only the prospective fused-QKV shape is stable, with a
neutral 0.99852 ratio. Native bit-exact, scalar and tail checks pass. This
rejects integrating that candidate on the available evidence; no qualified
49% slowdown claim is made from unstable timings.

The remaining target is MLP GEMM execution rather than input packing. Preserve
the established six-row kernel; the next change needs evidence about its
actual inner-loop cost, not another loop-order assumption. Diagnostic builds
are explicitly marked and rejected by performance qualification. They are
compiled with `LEAF_DIAGNOSTIC_TIMING` and run using
`tools/diagnose_mlp_costs.py`; the normal build contains none of these counters.

[Final verification](../benchmark/results/mlp_prefill_final_validation_windows.json)
replays both whole-model gate verdicts and binds the final source/test hashes.
Rebuilding normal candidate and default executables after adding diagnostic
guards produces byte-identical contents in every PE section compared with the
previously measured/architecture-tested binaries. This checks that diagnostic
instrumentation does not enter either normal executable; file hashes may still
differ because PE headers contain linker timestamps.
