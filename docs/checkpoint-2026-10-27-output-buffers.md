# October 27: benchmark accounting and reusable output storage

Continue on `codex/mlp-schedule`. Main has not been merged. Preserve the unrelated
`docs/embedding.md` edit and untracked October 8 image. No power setting changed.
No model download, competitor benchmark or full latency acceptance was attempted.

## Implemented and checked

- Optional `--benchmark-counters` in native bench mode; ordinary inference reads
  no counters. Version-one process/caller accounting includes raw cycles, CPU
  time, soft+hard faults and separate monotonic read brackets. No Administrator
  trace or profiling privilege is required. CPU/group endpoints describe stamp
  reads, not thread migration inside forward. No Hz or offCPU inference.
- `tools/benchmark_fp32_math.py`: local prepared inputs, exact quality-report
  binding, balanced AB/BA, power-mode/AC guards, retained samples, atomic pass
  checkpoints and incomplete-attempt/error retention. Supports same-binary math
  comparisons, A/A controls, prior binaries with their own bound quality, and
  uninstrumented diagnostics. A diagnostic screen cannot qualify acceptance.
- `Decoder::forward_into` writes caller-owned/resizable logits and retains
  capacity. The original `forward()` remains compatible. The native command
  reuses output storage for all modes without skipping projection work or
  changing numerical operations. Runtime/metrics expose the reusable path.
- Native API probe checks bit-exact old/new API outputs, full/last row shapes,
  cache/reset, capacities/pointers, independent owned returns, destination
  switching, exception recovery and caller-affinity restoration.
- README memory row/Mermaid and focused API/accounting documentation updated.

## Final qualification

Local outputs: `build/benchmark-counters-20261027/`.
Final runtime: path in `runtime-output-reuse.txt`, cache key `006ea80c451ea8fe`.
Pre-reuse counter runtime: `runtime.txt`, cache key `cfe245a348278f1a`.
`source-counter-baseline/` reconstructs that exact earlier native cache key.

- `output-reuse-trained/quality.json`: complete/passing, 16 complete 128-token
  blocks, 12 reference continuations, ten controls and eight default regressions.
  Exact rebinding to the prior qualified accurate binary; no new PyTorch oracle.
- `output-reuse-units.xml`: **1,350 passed, two optional skips**. Earlier counter
  stage: 1,348 passed plus the later two checkpoint-retention cases.
- `output-buffer-api.json`: **130 comparisons**, eight reduced configuration
  combinations plus two explicit two-worker trained-model combinations.
- `native-counter-check.json`: native delta units/wrap/zeros/failures and live
  ordinary Windows counter APIs pass.
- `installed-check.json`: compiler-free isolated offline wheel doctor/generation
  with accurate math, original BF16 head storage and explicit two-worker CPUs.
- `installed-code-binding.json`: installed `.text/.rdata/.data` match the final
  trained-qualified binary. Wheel: `output-reuse-wheels/leaf_cpu-0.2.0-py3-none-win_amd64.whl`,
  SHA256 `eac5f6736aba680ef12e5415b0dd1a81b9daffd1e6ccf37c68f5a385832cc385`.

The sandbox denied the initial read-only effective-mode API preflight; the same
preflight succeeded outside the sandbox. All measured windows then retained and
passed the `better-battery` mode audit. No settings were changed.

## Measurements and inference

`smol-policy-cost/` and `qwen-policy-cost/` compare accurate versus fast math on
the same counter-enabled binary. Both short screens stable: pooled accurate
prefill costs +0.77% Smol/+2.15% Qwen; decode nearly unchanged. These are not
promotion or nonregression acceptance results, and the ordinary fast policy's
older strict tolerance failures are not declared fixed.

`smol-output-reuse/` and `qwen-output-reuse/` isolate the output-ownership change
with accurate math on both sides. Every individual/pooled spread passes in these
two-pair/seven-sample screens. Qwen's per-phase per-sample fault count drops
**148 -> 0**; Smol stays zero. Pooled latency changes stay within 0.34%:
Qwen prefill 557.8043 -> 556.22805 ms, decode 107.261 -> 107.23915;
Smol prefill 483.40155 -> 485.0331, decode 78.5667 -> 78.48435.

The output-allocation churn is removed; no material latency win is established.
It was not the primary prefill bottleneck. These short diagnostic passes do not
resolve the earlier failed/interrupted ordinary acceptance or internal PPM
transitions. `acceptance_timing` and `timing_qualified` remain false in all
records. Full acceptance still uses the original sample counts/thresholds.
Raw records, historical harness source, code snapshots and test evidence are in
`benchmark/validation/benchmark-counters/20261027/` (weights/binaries omitted).

## Next implementation

Return to the actual shared linear cost, using the now-usable accurate policy,
reusable outputs and the existing native operation profile. Do not repeat quiet
A/B runs unchanged or chase the eliminated 148-fault pattern. Explicit Q/K/V and
up/gate packing reuse is currently enabled for accurate math; audit its cost
and whether the same unchanged-input guarantees can safely serve the ordinary
packed path, without promising a whole-model gain. MLP projections remain the
prior dominant operation; use a fresh scoped profile to select a change.

Default accurate-math promotion and latency acceptance remain open. Original
BF16 storage still covers only head/tied embeddings; interior source-BF16 is a
separate future format/reader/export change. Dataset/network expansion and new
model downloads remain deferred until this runtime milestone is settled.
