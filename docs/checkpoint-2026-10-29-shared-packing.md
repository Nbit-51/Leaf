# October 29: shared FP32 input packing

Stay on `codex/mlp-schedule`; the user explicitly wants to keep the branch and
not merge it yet. Preserve the unrelated `docs/embedding.md` edit and October 8
image. No new model download or competitor benchmark was performed. Keep the
user's efficiency mode; no power setting was changed.

## Implemented

- Ordinary optimized Windows FP32 now enables the existing SIMD input packer
  and the existing explicit Q/K/V and up/gate reuse promises, as accurate math
  already did. No new FMA/reduction order, weight representation or tile shape.
- Q/up always refresh; K/V/gate may reuse only with matching source/shape and
  no scaling. Mixed quantized artifacts remain excluded. Any nonpacked call
  clears the reusable source. Scalar/nonqualified platform behavior is retained.
- Native profiling adds separate nested `linear_details` (MLP/other totals,
  packing time and fresh/reuse counts); existing phase sums remain nonoverlapping.
  Disabled profiling reads no clocks. `tools/profile_decoder.py` now forwards
  explicit `--fp32-math` and records it.
- Trained qualifier optionally audits all fast blocks/continuations against a
  bound prior fast runtime, in addition to its existing accurate-reference check.

## Validated

Everything is under `build/shared-packing-20261029/`:

- `candidate.txt`: final runtime `cache/native/e08b482db58ebd47/leaf_decoder.exe`.
- `baseline.txt`: profiling-only baseline `cache/native/7b1d69b4898f5cef/leaf_decoder.exe`;
  `baseline-source/` retains its engine sources. Timed old/new comparison uses
  the previous qualified runtime in `build/benchmark-counters-20261027/runtime-output-reuse.txt`
  (cache 006ea80c451ea8fe), with profiling disabled on both sides.
- `all-units.xml`: **1,372 passed, two optional skips**; `focused.xml`: 33 passed.
- `native-check.json`: **1,462,208 values** pass raw-bit packing checks, including
  IEEE edge patterns, tails/strides, source mutation and workspace growth/shrink.
- `trained/quality.json`: complete/passing. **16 full 128-token blocks per policy**
  (205,914,112 logit values each), 12 accurate-reference continuations, 12 ordinary
  old/new continuations, ten accurate cache/scalar/thread/storage controls and
  eight ordinary regressions. Full fast checks establish no behavioral change;
  they do not fix the prior strict PyTorch tolerance misses. No new oracle run.
- `executed-qualifier.py` preserves the exact executed script hash. The tracked
  qualifier differs only by removed trailing whitespace after qualification.
- `installed-check.json` + `installed-code-binding.json`: isolated compiler-free
  offline doctor/generation with accurate FP32 + source BF16 head + two assigned
  workers; installed code/constants match the trained-qualified executable.
- `wheels/leaf_cpu-0.2.0-py3-none-win_amd64.whl`, SHA256
  `90a4ad223a74e378e97821ef992aaf2670d9ff50a5416e8edd00209a1f6492a5`.

## Measured inference

Fresh baseline profiles: linear 94-95% of prefill, MLP 66-74%; packing only 1-1.5%.
Profile pack time falls 9.212 -> 3.179 ms for Smol and 8.694 -> 3.298 ms for Qwen.
Fresh packs fall 224 -> 128 and 168 -> 96, with 96/72 explicit reuses. These profile
means include first-use/warmup and are diagnostic, not kernel acceptance.

`smol-unprofiled/` and `qwen-unprofiled/`: two balanced AB/BA pairs, seven samples
and three warmups per process; 63-token prefill plus one fixed cached decode,
fast FP32 both sides, identical artifact, assigned CPU2, AboveNormal child
priority, profiling/counters off. Every individual/pooled spread check and the
AC/effective-mode audit passes. Effective mode remains `better-battery`.

Prefill pooled medians: Smol 491.07455 -> 481.8984 ms (1.87% lower; paired 1.72%);
Qwen 557.25615 -> 553.8977 ms (0.60% lower; paired 0.92%). Decode is not changed by
this packing path; small decode movements are not claimed as a gain.

These screens do not fulfill full six-pair/31-sample acceptance and do not
establish a 2% prefill improvement. All `timing_qualified`/`acceptance_timing`
fields remain false. Main remains unchanged. README/Mermaid and the focused
report are updated; raw results/source snapshots/tests are in
`benchmark/validation/shared-input-packing/20261029/` with a hashed archive.

## Next

The measured packing cleanup is finished. Do not rerun it unchanged hoping for
a larger headline number. The next experiment should measure actual MLP GEMM
shapes against a current pinned hardware-throughput control, including load/cache
behavior and packed-buffer alignment, before selecting a compute-kernel change.
The earlier headroom finding already warned against assuming weight packing
would provide a large gain. Do not infer frequency from raw Windows cycles.

Accurate-math promotion/full latency qualification remain open. Original BF16
storage still covers head/tied embeddings only. Broad interior source-BF16
storage needs a separate format/export/runtime validation. Model/dataset/network
scope expansion remains deferred; no universal runtime win is claimed.
