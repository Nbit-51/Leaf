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

Initial monitored and native-loop diagnostics did not reproduce the one-core
fast burst. Repeated logits and linear work
stay identical in the recorded native audit. Failed/partial experiments remain
failed; the manifest marks release readiness false. See the
[checkpoint](../../../../../docs/checkpoint-2026-10-22-worker-placement.md)
for the publication hold and exact next diagnostic, followed by the ordinary
Smol/Qwen numerical-quality work.

[`minimal-evidence.zip`](minimal-evidence.zip) and
[`minimal-manifest.json`](minimal-manifest.json) append eighteen source/record
entries without modifying the first archive. The twelve-pass minimal replay
reproduces prefill spread failure in its final **unchanged baseline** pass.
Instrumented passes retain exact cold-reference outputs and matching packed
tile coverage. This still does not attribute the fast band. The administrator
trace wrapper's syntax and artifact preflight pass preceded the recording below.

[`trace-evidence.zip`](trace-evidence.zip), [`trace-manifest.json`](trace-manifest.json)
and [`trace-summary.json`](trace-summary.json) retain the completed administrator
replay, selected aggregates and diagnostic-reader sources. Microsoft TraceEvent
decodes 17,431,184 events with zero reported loss; CPU 2/3 switch chains have no
discontinuities. The first unchanged baseline fails prefill spread (1.26984);
the other eleven passes have passing spreads. No samples are removed.

The first process receives 97.10% of its elapsed window as scheduled CPU 2 time.
Its fast interval overlaps a `Constrained` → `LowLatency` PPM transition and
delivered-performance event medians of 248–254; later steady passes report 154.
These are decoded event distributions, not normalized latency scores or proof
of a thermal cause. A post-trace API check reports `better-battery` despite
verified AC and a false legacy saver flag. The effective mode during the earlier
trace was not continuously measured. The raw 995-MiB ETL and other-process
inventories stay local; the summary binds the ETL by SHA-256.

The archive contains the initial 94-test mode-guard snapshot. Later explicit
matched-efficiency support is separate: the user chose to retain their power
settings, so saving modes can be selected explicitly while all transitions and
the original statistical gates still fail closed. Fresh acceptance remains
required; no earlier failed gate becomes a pass.

[`efficiency-evidence.zip`](efficiency-evidence.zip) and its
[`manifest`](efficiency-manifest.json) append the final 97-test guard sources,
real matched-mode preflight and the interrupted efficiency-mode attempt. All
twelve Smol passes remain: pooled/paired latency ratios pass, but stability
fails. Seven Qwen passes are retained without a complete verdict. The user
requested stopping and pushing a checkpoint; the raw incomplete record is
unchanged. Final mode-window/sleep audits did not finish, so this is not a
qualified comparison. No system power setting was changed.
