# October 18 accumulation continuation

Branch: `codex/linear-accumulation`, original checkpoint `08bc3b6`, release base
`107cb98`. Preserve the unrelated root PNG. User now authorizes full quality
and controlled timing, confirms plugged in / no other work, and requests
publication after verification. Do not reuse the earlier small-check-only
restriction. No agent compute is active.

## Completed

- Original K=128 / FP64-outer-sum candidate (`20261017-v2`) passes full trained
  **eight-block quality and six continuations on both Smol and Qwen**, unchanged
  strict tolerance and perplexity gates. Each model scores 1,016 next-token
  targets with 100% agreement and zero all-logit tolerance violations.
- Inlining removes repeated saves/restores of nine SIMD registers and
  `vzeroupper` from the partial-call boundary. The inline and 96-row cache-blocked
  variants preserve the two retained block outputs exactly but show **no
  demonstrated micro speed win**. Do not choose one based on a best sample.
- Original candidate cache checks pass normal full/chunked/threaded paths,
  trained GPT-2/TinyLlama prefixes and exact W8A8 exclusions. Forced scalar
  fails: Smol 121 / Qwen 3 violations on the tested prefix. Released scalar
  controls also fail (147 / 481), narrowing the remaining issue to its long
  FP32 dot reduction. These failures remain saved, not ignored.
- Added pure-FP32-only scalar FP64 dot reduction. Final candidate:
  `build/linear-accumulation-20261018-scalar/leaf_decoder.exe`, SHA256
  `d725b5e54780907225d3cdaf85749bbc6d67ecca3e3ddeac2873de99c584102d`.
  It uses the original noinline GEMM policy, not the slower cache variant.
- Final candidate passes all **12 cache/scalar/thread/earlier-model/quantized
  exclusion checks**, with zero violations in reference checks.
- Final candidate is rebound through **all 16 full blocks and 12 continuations**
  to the complete strict quality passes: `quality-binding.json` complete/pass.
  No reference metrics are reused based on a subset; every block is compared
  byte-for-byte, and every generation matches saved PyTorch tokens. Derived
  quality records identify this provenance explicitly.
- Final candidate passes ten reduced architecture configurations and
  **1,048 unit tests**. Native multi-EOS tests use this candidate; the old-reader
  tests use `build/fp32-silu-20261015/release/leaf_decoder.exe`.

## Serial acceptance

Before binary: `build/numerics-20261017/release/leaf_decoder.exe` (released
Windows default). Candidate: final scalar-corrected binary above. The new
`benchmark/experiments/linear-accumulation/acceptance.py` requires complete,
hash-bound eight-block quality and exact continuations, then calls the existing
ABBA runner without relaxing any gate: prefill <=0.98, decode <=1.02, six pairs,
31 measured samples / ten warmups, CPU 2, all individual and pooled stability.

First attempt could not write the sandbox system-temp request file and has
**zero measured passes** (`acceptance-smol.json`). Retain it as interrupted.
Retry sets workspace TEMP/TMP and uses `acceptance-smol-v2.json`.
Do not overwrite or restart an ongoing acceptance process. Do not run any
other model load, build, microbenchmark or test concurrently with acceptance.

Smol is complete: all twelve passes and pooled distributions stable, but
prefill **492.9177→514.36175 ms (+4.35%)** fails the unchanged speed gate.
Decode **80.5039→80.10565 ms** passes non-regression. Qwen is also complete:
all twelve passes and pooled distributions stable, prefill
**565.62355→589.518 ms (+4.22%)** fails the unchanged speed gate; decode
**110.15875→109.54915 ms** passes non-regression. No acceptance process is
active. **Do not repeat these completed comparisons.**
The following command records the invocation, not a rerun request:

```powershell
$env:TEMP = "$PWD\build\linear-accumulation-20261018-scalar\tmp"
$env:TMP = $env:TEMP
python benchmark/experiments/linear-accumulation/acceptance.py --before build/numerics-20261017/release/leaf_decoder.exe --after build/linear-accumulation-20261018-scalar/leaf_decoder.exe --quality build/linear-accumulation-20261018-scalar/full-quality-qwen/quality.json --workdir build/numerics-20261017/decode-silu/qwen-work --output build/linear-accumulation-20261018-scalar/acceptance-qwen.json
```

All current reports live in the final build directory. Original full quality
is under `build/linear-accumulation-20261017-v2/full-quality-{smol,qwen}`.
Inline/cache candidate reports are in `build/linear-accumulation-20261018-{inline,cached}`.

Initial cache checker incorrectly treated saved `[1,tokens,vocab]` arrays as
`[tokens,vocab]`; no model failure followed from that wrapper bug. It is fixed,
and the failed report is retained as `20261017-v2/cache-quality.json`.
`cache-quality-v2.json` is the completed original-candidate run (with genuine
scalar failures). Final candidate `cache-quality.json` passes all cases.

**Publication is stopped at the user's latest direction:** fix the overhead
before opening a PR. The branch was pushed through `cbc9953` as a checkpoint;
no PR was opened or merged. No production default is changed. Do not turn the
completed failed speed gate into a pass, or repeat that same comparison.
Independent-CPU timing, larger contexts and full trained GPT-2/TinyLlama
reevaluation remain outside the completed Smol/Qwen qualification.

## Follow-up: cheaper reduction and shared packing

The user's library policy permits OpenBLAS when a fair comparison demonstrates
a useful win. Leaf owns its import, runtime, cache, execution and custom kernel
design; selected numerical dependencies are allowed. OpenBLAS is **not**
integrated. This is not a blanket ban on libraries.

New code remains under `benchmark/experiments/linear-accumulation/`:

- `tree_sum.h` combines short FP32 FMA partials with a balanced FP32 tree,
  removing the FP64 outer conversions. It retains the six-row / sixteen-token
  tile and validates the same strides, extents and scalar fallback.
- `--reuse-packing` enables the existing exact vector transpose and explicitly
  reuses unchanged input panels within Q/K/V and up/gate groups. Every group
  begins with a fresh pack; scaled inputs and non-FP32 artifacts cannot reuse.
- Tree collapse visits only populated levels instead of scanning all 64
  possible levels. Retained Smol/Qwen outputs are byte-identical before/after
  this traversal-only change. It did not materially improve measured timing.
- `--tree-chunk-size 256` halves partial-combination frequency relative to
  128, using one global width rather than per-model rules. Arithmetic changes,
  so it requires fresh complete reference checks.

Build directories `20261018-tree`, `-reuse`, `-depth`, and `-k256` have separate
binaries/manifests and retained results. The latest candidate is
`build/linear-accumulation-20261018-k256/leaf_decoder.exe`.
Historical tree manifests incorrectly record `inline_partial=false` while
their recorded compiler commands explicitly enable it; the metadata field is
fixed for future builds. Do not rewrite historical manifests or input hashes.

The K=256 candidate passes thirteen synthetic cases, both retained length-128
blocks (zero strict logit violations), and all twelve retained-reference
cache/scalar/thread/earlier-model/W8A8 checks. Its **diagnostic-only** two-pair
screen reports prefill release/candidate medians of 490.15335/492.7776 ms on
Smol and 561.59515/570.31875 ms on Qwen: ratios 1.00535 and 1.01553. Decode
ratios are 0.99721 and 1.00183. These seven-sample passes do not meet the
acceptance sample count and establish neither a release nor a speed claim.
The previous K=128 tree/reuse screens were roughly 2% above release.

Four short existing-profiler runs (`profile-{smol,qwen}-{before,after}.json`)
retain phase evidence: linear FP32 is over 93% of profiled prefill, RMSNorm
under 1%. These profiles include first use and warmup, so their means must
not replace unprofiled steady-state acceptance measurements.

`run_blas_probe.py` builds a separate native comparison executable against the
existing NumPy OpenBLAS ILP64 DLL. Same inputs/outputs, one thread, CPU 2,
above-normal priority, packing included, reversed run order, 31 samples per
pass. All four synthetic shapes pass the same FP64-reference tolerance.
`build/linear-accumulation-openblas-20261018/results.json` retains all samples,
DLL/source/executable hashes and the actual OpenBLAS configuration
(`0.3.31.188.0`, Haswell, ILP64). Projection medians:

| Tokens / K / outputs | Leaf release | Accurate tree K=128 | OpenBLAS |
|---|---:|---:|---:|
| 63 / 960 / 2560 | 3.24220 ms | 3.35075 ms | 4.82520 ms |
| 63 / 896 / 4864 | 5.83540 ms | 6.04035 ms | 8.52435 ms |
| 63 / 2560 / 960 | 3.30240 ms | 3.35670 ms | 4.81860 ms |
| 63 / 4864 / 896 | 5.90005 ms | 5.99160 ms | 8.53160 ms |

This particular OpenBLAS build is slower in this single-core microbenchmark;
it is not a whole-model, multithreaded or universal library comparison. No
OpenBLAS dependency/backend has been added to Leaf.

Full eight-block, six-continuation reference checks for K=256 **complete and
pass on both models**, serially via `leaf evaluate`. Each scores 1,016 targets
with 100% next-token agreement and zero strict logit violations; all twelve
greedy continuations match exactly. Perplexity ratios are 1.00000005629
(Smol) and 1.00000020417 (Qwen). Reports are under
`full-quality-{smol,qwen}` in the K=256 build. No long acceptance run is
queued: the short screens still do not suggest the unchanged >=2% prefill
improvement gate would pass.

An additional `--tree-cache` candidate reuses each K chunk across twelve
output rows with independent trees. At K=256 its active input/weight working
set is 28 KiB plus a few KiB of live tree state, unlike the older 96-row /
FP64-total experiment. The intended arithmetic is bit-identical to K=256;
the probe explicitly compares every synthetic output against the original
tree traversal. This is an L1 working-set hypothesis, not a speed claim.
Build: `build/linear-accumulation-20261018-l1/`.
The build and thirteen synthetic correctness cases pass, including exact
comparison with the non-cached tree. The short packing-inclusive micro probe
is still slower than release: 3.24670/3.34545 ms for 63x960x2560 and
5.90475/6.19295 ms for 63x896x4864 (release/cache-tree). **Stop this variant
here; no model acceptance or larger quality rerun is queued.** It has not
demonstrated the hypothesized cache gain. Keep K=256 without `--tree-cache`
as the fully quality-checked follow-up candidate; it remains an experiment.

All processes from this follow-up finished. No timing, build, model evaluation
or profiler is left active. Source and this checkpoint are saved locally;
publication remains stopped. The next change must target measured linear
cost and clear a short balanced screen before spending time on another full
acceptance run. OpenBLAS can be reconsidered for other builds, shapes or
thread counts, but this measured single-core DLL is not an optimization here.
