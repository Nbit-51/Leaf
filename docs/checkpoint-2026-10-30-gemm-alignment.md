# October 30: MLP GEMM alignment diagnostic

Continue on `codex/mlp-schedule`; do not merge or delete the branch. Preserve the
unrelated `docs/embedding.md` edit and October 8 image. The user's efficiency mode
is required; no system power setting is changed.

## Question and declared experiment

October 29 profiles placed 94-95% of prefill in linear projections and 66-74%
in MLP projections. Input packing itself was only 1-1.5%. The shared packing work
is complete. This experiment measures the actual compute path before selecting
another change.

`benchmark/gemm_alignment_benchmark.cpp` exercises the shipped full-K 6x16 tile
through one noinline raw-panel entry. Smol hidden/intermediate widths are 960/2560;
Qwen widths are 896/4864. Test both up/down directions, 63 and 31 tokens, and one
or four rotating weight matrices: sixteen cases. Packing is outside the timer.
Four byte-identical packed copies have pointer offsets 0, 16, 32 and 48 modulo64.
Every live output is checked against the existing runtime kernel. Scalar fused
checks independently traverse direct token-major inputs, including the tails.

Each round uses symmetric forward/reverse phase order, reversed again on alternate
rounds. Retain every sample. Adjacent independent register-only FMA loops estimate
local throughput without inferring GHz. This estimate is not a hardware ceiling:
cache state and frequency can differ. Separate phase allocations also change cache
exposure; this is a controlled pointer-phase diagnostic, not unique attribution
to one CPU event. A large effect requires a same-workload runtime confirmation.

Local root: `build/gemm-alignment-20261030/`. `plan.json` records the declaration;
`baseline-source/` preserves the production headers. Runtime source is unchanged
at the start of the experiment. The wrapper rejects partial/malformed native
records and retains unstable distributions rather than filtering them.

## Preliminary checks

The sandbox could not register Windows effective-mode notifications; the failed
setup is retained under `phase-screen/`. A subsequent permitted run reached the
native correctness check and stopped before collecting any case timing because
the baseline MinGW `std::fma` library reference disagreed with the existing SIMD
kernel. `phase-screen-ac/` retains that failure with an empty raw file; it is not
a performance result. A target-AVX2/FMA scalar check matches the stock kernel on
the same sampled positions. The initial executed C++ source is preserved as
`initial-gemm_alignment_benchmark.cpp`; the scalar-reference diagnosis is kept
locally. No production arithmetic was changed to resolve this harness issue.

## Resume

Read final results below when available. Do not repeat failed setup or use its
empty file as evidence. Build commands, source/binary hashes and raw output must
bind the actual executed benchmark revision. No release/model/competitor speed
claim is authorized merely by completion of a microbenchmark.


## Completed alignment screen

`phase-screen-corrected/alignment.json` completes all sixteen cases, 17,446,400
exact output comparisons and 1,440 scalar hardware-FMA oracle positions. AC and
effective `better-battery` audits pass; every sample is retained. Ten cases have
stable distributions at all four phases, six do not. Aligned phases show no
consistent advantage across both models/directions. One stable rotating Smol-up
case improves, but that does not justify a general allocator change. No production
alignment policy was changed. The adjacent FMA controls also drift, especially
in the first case; their fractions must not be advertised as a hardware bound.

A read-only review found retaining 8-23 final MLP rows instead of 16-31 would
remove one packed panel at a 63-token prefix, with a likely whole-prefill effect
under 1%. It was not implemented or timed in this session. The October 15 final
query/attention extension was not repeated.

## Broader original BF16 matrices

Implemented explicit `leaf run --source-bf16`, separate from the old head-only
scope and FP32 cache. New version 6 extends original BF16 storage to canonical
attention/MLP/embedding matrices, while norms/biases/F32/F16 sources remain FP32.
No additional quantization, dependency or accumulation change. The source-value
and quality gates are exact, not a relaxed PyTorch tolerance. CLI checks capability
before weight downloads; ordinary defaults remain unchanged. `leaf evaluate`
does not yet prepare this storage policy; use the dedicated developer qualifier.

Local root is `build/source-bf16-matrices-20261030/`:

- Initial candidate `candidate-initial.txt` (also initially `candidate.txt`):
  `cache/native/ac350a1af3587177/leaf_decoder.exe`.
- `trained/quality.json` is complete/passing: every payload bit, 16 full blocks
  per model across fast/accurate policies, eight continuations and eight controls
  per model. References are the preceding qualified Leaf, not a fresh oracle.
- `all-units.xml`: 1,513 passed, two optional legacy skips. `focused.xml`:
  285 passed, two skips. Installed wheel check and code/data-section binding pass.
- Initial short screens compare existing head-only BF16 to broader matrices,
  accurate/accurate, CPU2, one thread, two balanced pairs, 7 samples +3 warmups,
  profiling/counters off. All passes, pooled phases and AC/effective-mode audits pass.
  Smol prefill 490.59235 ->533.57530 ms (+8.76%), decode 76.01540 ->48.46885 ms
  (-36.24%). Qwen prefill 556.91155 ->623.58215 ms (+11.97%), decode97.35350
  ->65.25210 ms (-32.97%). These short diagnostics do not qualify release latency.

The prefill penalty motivated one bounded change: reduce converted scratch from
96 rows to the kernel's six-row tile. Largest Qwen down buffer drops from
1,867,776 to116,736 bytes. Arithmetic/rowtile boundaries are unchanged; scalar and
single-token paths are unaffected. `candidate-tile6.txt` points to cache84ef3bff71eeb111.
`tile6.xml`: 176 focused checks pass, one optional legacy skip. Full trained audit
is running under `trained-tile6/`, with all initial results retained. Do not time
concurrently with correctness/builds. Do not assume the smaller scratch is faster.

Update this checkpoint with the final selected source, qualification, wheel and
screens before any push. Keep this branch; do not merge/delete it.


## Final checkpoint before branch push

The user requested stopping new experiments and pushing with about 5% usage left.
The six-row audit finished **complete/passing** before that push. Select the
six-row implementation currently in the working engine; do not revert it or
merge this branch. `final-runtime.txt` and `candidate-tile6.txt` identify
`cache/native/84ef3bff71eeb111/leaf_decoder.exe`. The original `candidate.txt`
continues to identify the initial96-row binary for provenance.

- `trained-tile6/quality.json`: complete/passing.32full128-token blocks across
  two models/two policies, **411,828,224 logit values**.16greedycontinuations and
  16cache/scalar/two-thread controls also pass. Every logical tensor value is
  independently audited against the original FP32 artifact. No new oracle run.
- `tile6.xml`:176focusedpasses, oneoptionallegacy skip. The preceding full suite
  is1,513passes/twooptionallegacy skips on the96-row version. Do not mislabel
  that full suite as a six-row run.
- `installed-check-tile6.json` and `installed-code-binding-tile6.json` pass:
  actual wheel installation, isolated/offline doctor+generation, accurateFP32,
  broad originalBF16storage, assigned2workers, no compiler/override/Torch.
  Installed .text/.rdata/.data match the fully trained-qualified six-row binary.
- Final wheel: `wheels-tile6/leaf_cpu-0.2.0-py3-none-win_amd64.whl`.

**Next action:** run the two short source-storage comparisons using the existing
`timing-plan.json`, final six-row binary/proof and already prepared artifacts.
Compare against the qualified head-only artifact/runtime as before. Timing for
six rows is stillpending; do not borrow the initial96-row latency numbers.
If prefill remains slower, profile conversion/allocations and read the optional
CPU-work/page-fault counters before choosing another change. Do not repeat full
correctness or rebuild unchanged sources. Use the six-row installed scripts and
proof paths; original scripts remain for the original candidate.

Ordinary FP32 arithmetic/default storage are unchanged. Broader storage is an
explicit option with validated fidelity and artifact-size evidence. Its full
latency qualification is open. Keep efficiency mode and the existing branch.
No benchmark process is left running. No main merge or branch deletion.
