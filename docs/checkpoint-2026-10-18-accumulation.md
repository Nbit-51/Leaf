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
Decode **80.5039→80.10565 ms** passes non-regression. Qwen is now running
serially in tool session **76390**, output `acceptance-qwen.json`; do not start
another Qwen run. Its first pair is stable at 565.53→587.98 ms prefill.
The following command records its current invocation, not a rerun request:

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

Do not promote or advertise a speed win unless final gates pass. If performance
misses its gate, preserve the demonstrated accuracy fix and report its cost;
any release-criterion change requires the user's explicit decision. Publish
reviewable work through a PR, never bypass protected main. Merge/delete the
temporary branch only when the actual release decision and all CI checks permit.
