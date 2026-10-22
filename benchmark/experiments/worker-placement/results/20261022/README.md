# October 22 worker evidence

[`evidence.zip`](evidence.zip) contains 411 byte-preserved raw records and source
snapshots. [`manifest.json`](manifest.json) lists each entry's original path,
size and SHA-256, plus the archive digest. Every archived entry was checked after
writing. Weights, logits, executables, installed packages, temporary unit-test
fixtures and unrelated process inventories are omitted. The private environment
trace is represented only by an aggregate summary and its raw-record hash.

| Comparison | Verdict |
|---|---|
| Four-core placement off/on | Assigned side stable; joint speed gate fails on unassigned spread |
| Accurate arithmetic, one-core old/new default placement | Non-regression passes |
| Matched four-core job sizes | Prefill passes; Smol baseline decode spread fails joint gate |
| Matched four-core bounded waiting | **Passes both models and every spread check** |
| Final ordinary arithmetic, one-core old/new | Ratios within 2%; individual spread fails joint gate |

The bounded-wait comparison uses accurate K=256 FP32 math and the same assigned
physical cores on both sides. Smol decode is 45.3642 → 42.8258 ms; Qwen is
53.9541 → 50.78745 ms. Prefill passes non-regression. These are local Leaf policy
comparisons, not competitor results or ordinary-default strict-quality passes.

The corresponding policy is integrated locally for explicit multi-thread CPU
placement. Final checks include 1,155 units, 130 native worker cases, ten reduced
architectures, twelve ordinary trained prefix/cache/generation regressions and
an isolated installed-wheel check. The accurate candidate matches all sixteen
trained blocks, twelve reference continuations and ten controls exactly.

One-core speed-band variation remains unattributed: monitored and native-loop
diagnostics did not reproduce the fast burst. Repeated logits and linear work
stay identical in the recorded native audit. Failed/partial experiments remain
failed; the manifest marks release readiness false. See the
[checkpoint](../../../../../docs/checkpoint-2026-10-22-worker-placement.md)
for the publication hold and exact next diagnostic, followed by the ordinary
Smol/Qwen numerical-quality work.
