# October 20: prefill scheduling and worker placement

Branch `codex/mlp-schedule`, based on main `880139d` (merged PR #10).
**Default engine unchanged; no new kernel release or competitor win.** Dataset
work is deferred. Preserve the separate pre-existing `docs/embedding.md` edit
and unrelated untracked image; neither belongs in this checkpoint's commit.

## Candidate and checks

The six-row FP32 GEMM currently gets 32-row worker jobs. This candidate uses
96 rows when rows/worker >=384, 48 when >=192, and 32 otherwise, retaining at
least four jobs/worker. One-thread, scalar, small-matrix, GEMV and quantized
policies are unchanged. No arithmetic/artifact-format change. Accurate K=256
math remains experimental.

- **240** native exactly-once/range/reuse cases pass, 1/2/4 threads, with invalid
  grain rejection/recovery. The final pool retains the old 128-byte size.
- **1,125 Python units**, ten reduced architecture cases and eight trained
  ordinary-build thread/cache/generation regression cases pass.
- Final accurate build: **16 trained full-logit blocks**, **12 reference
  continuations** and **8 scalar/thread/cache controls** pass bit-identically.
  This rebinds the prior full qualification without relaxing tolerances.
- Candidate wheel: isolated installed `doctor` and fresh offline source-BF16
  generation pass without compiler/binary setup or Torch/Transformers imports.
- Two K-loop unroll variants each pass 454 exactness cases but have no qualified
  gain. Stop them. The refined job probe passes 324 exactness cases; mixed
  timing stability prevents a blanket operator gain claim.

## Whole-model acceptance fails

Both sides: accurate K=256 FP32 math, source-BF16 vocabulary storage, four threads
on distinct physical P-cores 2/4/6/8, identical work/artifacts. Six balanced pairs
per model, 31 samples and ten warmups/pass; no samples removed. Native/parent AC
checks, scoped awake requests and sleep audits. No logged sleep overlap.
Gate: both pooled and paired prefill <=0.98, decode <=1.02; all individual and
pooled p90/p10 <=1.25. Retain both attempts; do not retry unchanged until one passes.

| Attempt/model | Pooled prefill before / after ms | Paired ratio | Failure |
|---|---:|---:|---|
| First / Smol | 173.85885 / 169.51545 | 0.97150 | Pooled decode ratio 1.02321 |
| First / Qwen | 233.31760 / 225.20420 | 0.95999 | Spread |
| Final / Smol | 176.51610 / 202.84480 | 1.05970 | Ratios and spread |
| Final / Qwen | 202.91610 / 184.45270 | 0.98804 | Paired prefill threshold and spread |

The final attempt removes an avoidable pool-layout expansion. It still fails.
No release-qualified prefill improvement was established.

## Diagnostic: intermittent effective worker placement

An unchanged four-thread Smol binary reproduces fast ~167–169 ms and slow
~202–203 ms prefill bands. In the slow passes CPU 2 is repeatedly parked and
~1% busy despite the four-core process mask. All measured Smol forward phases
have zero process fault deltas. Prefill cycle totals are similar across bands.
Coarse CPU accounting is not exact per-phase dispatch attribution.

The follow-up uses the same instrumented binary in off/on/on/off order. It
registers actual inference worker IDs, gates startup and pins each owned thread
to one allowed core for ON. Windows background threads are excluded. The monitor
is on CPU 27; no global settings change. Each model's full 33-token logits
exactly match the uninstrumented candidate. No AC failure or logged sleep.

| Model | Placement | Prefill pass medians ms | Decode pass medians ms |
|---|---|---:|---:|
| Smol | Normal | 203.6318, 169.2985 | 51.0818, 44.5364 |
| Smol | One owned worker/core | 168.3159, 168.7956 | 45.7024, 46.2562 |
| Qwen | Normal | 222.9989, 181.8958 | 58.8495, 52.2084 |
| Qwen | One owned worker/core | 181.8285, 182.3621 | 53.2067, 53.2002 |

Smol's slow normal pass has CPU 4 parked in all measured telemetry observations
and ~1% busy; all four are active in assigned-worker passes. Qwen's slow normal
pass has CPU 8 parked in most observations. Qwen also records **3,997** total
process faults in that pass, **297** in each assigned-worker pass, and one in
the final normal pass. Soft/hard faults are combined, so disk paging is not
established; that additional memory activity is a Qwen confound.

This supports worker placement/core parking as a contributor on this host.
It does not explain every historical spike, prove thermals, or qualify a universal
affinity speedup. Normal placement can reach the fast band. Two ON passes/model
are a diagnostic control, not full acceptance.

## Isolation and resume

Unqualified engine edits were removed after preserving the patch/raw source.
[`mlp-schedule`](../benchmark/experiments/mlp-schedule) now reconstructs the
candidate only in a fresh private `build/` tree. Reconstruction passes the
240-case worker test, reproduces both generated translation units byte-for-byte,
and matches all 33-token logits on Smol/Qwen (1,622,016 / 5,013,888 values).
This is a reconstruction smoke check, not another full trained qualification.

Compact raw reports and source/runner snapshots are retained in
[`results/20261020`](../benchmark/experiments/mlp-schedule/results/20261020).
Large weights, binaries and logits remain in ignored `build/`.

Local base: `build/mlp-unroll-20261020/`:

- `full-model-v2/`: first complete quality/timing; `full-model-final/`: final.
- `process-diagnostic/`: unchanged-candidate trace that catches both bands.
- `placement-v2/`: completed placement controls for Smol/Qwen.
- `placement-diagnostic/`: harness abort before timing; explicit worker IDs fix
  ambiguity caused by Windows background threads in process enumeration.
- `unit-final-results.xml`, `architectures.json`, `default-parity-final.json`,
  `layout.json`, `installed-check.json`: completed non-timing checks.
- `build/mlp-schedule-reconstruction-20261020/`: private rebuild and smoke proof.

## Next implementation

Add an explicit validated worker-placement policy to the native pool, including
caller-affinity restoration and thread-lifetime cleanup. Respect allowed masks;
do not hardcode laptop CPU numbers or model identities. Qualify any default
change with correctness and matched acceptance. Then compare job sizes using
the same placement policy on both sides. Do not repeat inner-FMA/OpenBLAS trials.

MLP projections remain the compute target (~67% Smol / ~77% Qwen in the retained
profile). Placement addresses execution variability; it does not remove the
accumulation cost or establish a lead over llama.cpp. Finish inference before
expanding dataset work.
