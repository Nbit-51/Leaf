# October 22: explicit worker placement

Continue on `codex/mlp-schedule`, starting from `c88595c`. Dataset work is deferred.
Keep the pre-existing `docs/embedding.md` edit and unrelated image unstaged.

**Latest:** the administrator trace is complete. It captures a PPM profile/
delivered-performance change during the first unchanged-baseline speed burst.
The new effective-mode guard passes 97 focused tests. The user explicitly chose
to retain their Energy saver/Best Power Efficiency settings; do not change them.
The fresh `efficiency-default-model/ordinary.json` comparison uses the same
`better-battery` mode on both binaries, a 30-second quiet startup, no profiling,
and unchanged gates. Smol's ratio checks pass but its spreads fail. At the user's
request to push a checkpoint and take a break, testing stops with all twelve
Smol passes and seven of twelve Qwen passes retained. No complete Qwen verdict,
final mode-window verdict or sleep audit exists for this interrupted attempt.
Do not retry unchanged or publish this as a successful remedy.

## Implemented

- `worker_affinity.h`: validated Windows/Linux thread scopes, allowed-mask checks,
  nested binding reuse and caller restoration. No process mask/power setting changes.
- `Workers`: optional CPU list, verified startup, owned-worker lifetime binding,
  exception propagation and safe reuse after failed tasks.
- Decoder/CLI: optional `--worker-cpus` list, native capability and metrics;
  incompatible older overrides fail before weight downloads. Existing auto-speed
  profiles cannot select quantization for a different placement policy.
- Default placement and model arithmetic are unchanged. Earlier unqualified
  worker-grain code stays isolated. Its builder now reads historical engine
  revision `880139d`, preserving reproduction after current engine changes.

## Completed correctness

- The placement-only ordinary runtime passes **1,151 Python unit tests**;
  the final bounded-wait runtime passes **1,155**, as recorded below.
- Native worker test: **130** range, participation, binding/restoration, nested
  scope, task-exception/reuse and invalid-policy checks.
- All **16 full trained logit blocks**, **12 reference continuations** and
  **10 scalar/cache/thread/default controls** match the prior qualified accurate
  build exactly. Strict tolerance thresholds are unchanged.
- Six ordinary FP32 trained regression cases match the preceding default exactly;
  this is not new strict PyTorch qualification.
- Wheel builds; isolated installed startup and offline source-BF16 generation
  with two explicit worker CPUs pass without compiler/decoder overrides or
  Torch/Transformers imports.
- Historical GEMM-schedule reconstruction builds and passes its 240 native cases.

## Controlled timing and remaining spread

The placement stage is now complete: assigned workers pass every individual
and pooled spread check on both models. The unassigned side fails several
spread checks, so the joint percentage-speed gate remains failed. Pooled
assigned p90/p10: Smol prefill 1.04648 / decode 1.14235; Qwen prefill 1.04018 /
decode 1.10580. All samples remain retained. This is a repeatability result,
not a qualified claim that placement makes every workload faster.

The one-core **accurate-arithmetic/default-placement** non-regression stage
passes both phases/models. Smol prefill 490.37065 → 490.97995 ms, decode
76.5763 → 76.20775; Qwen prefill 557.97285 → 556.70695 ms, decode
98.2324 → 97.5628. This is not the final ordinary installed-arithmetic guard.

The new isolated job-size candidate retains a fixed-32 queue specialization for
GEMV and changes only multi-token packed FP32 work. All sixteen full trained
blocks, twelve reference continuations and ten controls match exactly. Twelve
ordinary prefix/cache/generation regression cases also pass across Smol/Qwen
FP32, GPT-2 FP32 and TinyLlama W8A8; these are not strict PyTorch qualification.

Matched four-core placement on **both** binaries gives:

| Model | Before / after prefill ms | Before / after decode ms | Verdict |
|---|---:|---:|---|
| Smol | 174.85225 / 168.91345 | 45.62010 / 45.72155 | Prefill passes; decode spread fails |
| Qwen | 187.20755 / 183.06035 | 54.29260 / 54.14110 | Both phases pass |

The joint gate remains **failed**. One Smol baseline decode pass has p90/p10
1.25687626 against 1.25. All other individual and pooled checks pass. No sleep
overlap was logged. Do not remove that pass, relax the limit, enable the job
candidate, or retry unchanged to obtain a pass.

At the user's request, hold publication and prioritize the remaining comparison
spread. A new unchanged-binary A/A diagnostic uses the native explicit policy,
process CPU time/cycles/faults and an external core monitor on CPU 27. It passes
four Smol runs, with decode spreads 1.0706–1.1432, zero process faults in measured
forwards and no sleep. Full 33-token logits match the uninstrumented binary.
It does not reproduce or explain the failed burst. Decode cycle spreads are
1.029–1.042; wall time varies more. Parking flags can coexist with substantial
busy time here, so a flag alone does not establish that a core stopped executing.

The caller-barrier/dispatch diagnostic in `queue-jitter/` reproduces a failed
decode pass: p90/p10 **1.285**. Its fourth pass's fast/slow quartiles have median
wall times **44.463 / 54.149 ms** and worker-barrier waits **3.531 / 10.357 ms**.
Wall/wait Pearson correlation is 0.977 for that pass (0.797–0.977 across four).
There are 225 pool calls/decode, typically 124–127 blocking completions. Fault
deltas remain zero. This localizes much of the variability to synchronization
waiting; it does not establish the responsible OS process or distinguish late
worker dispatch from late caller wakeup/worker imbalance.

`build_spin.py` constructs a private 50-us bounded spin-before-sleep candidate.
Only explicitly assigned multi-thread pools use it; unassigned and one-thread
execution remain blocking. Both the caller's completion wait and owned workers'
next-job wait use release/acquire epoch notifications before falling back to the
existing mutex/condition-variable protocol. A deadline check every 64 pauses
bounds idle spinning, including shutdown. It changes no arithmetic. The native
130-case affinity/range/exception test passes. Full trained checks are running
under `spin-model/`; do not call the candidate a timing or stability fix yet.

Its predeclared comparison is `synchronization`: same assigned cores and accurate
math on both sides, prefill paired/pooled <=1.02, decode <=0.98, every individual
and pooled p90/p10 <=1.25, unchanged AC/awake/sleep guards and no filtering.
The bounded-spin comparison is now **complete/passing**: six balanced pairs per
model, 31 samples and ten warmups/pass, with all individual and pooled spreads
passing and no logged sleep overlap.

| Model | Before / after prefill ms | Before / after decode ms | Paired decode ratio |
|---|---:|---:|---:|
| Smol | 175.4384 / 173.3760 | 45.3642 / 42.8258 | 0.94116106 |
| Qwen | 186.54535 / 185.0512 | 53.9541 / 50.78745 | 0.93451649 |

Pooled decode reductions are **5.60% / 5.87%**; prefill passes the declared
non-regression criterion. This is accurate K=256 arithmetic with four explicit
physical cores on this Windows host, not a competitor benchmark or an ordinary
default strict-quality pass. Do not reclassify the failed job-size experiment.

The diagnostic mechanism also changes as intended: blocking completions fall
from approximately 125 to 2–3 per Smol decode; caller context switches fall from
approximately 140 to 22–24. Four instrumented new-policy passes have decode
spreads 1.094–1.128. Process cycles increase, consistent with additional active
polling. These sequential diagnostic runs describe a mechanism and CPU tradeoff;
the separate uninstrumented balanced comparison supplies the timing verdict.

The policy is integrated into the public worker pool, restricted to explicit
multi-thread placement. Actual metrics report `worker_spin_us` (50 or zero).
Every engine source/header matches the qualified private tree byte-for-byte,
except `main_decoder.cpp`'s outside-timing output/capability metadata.

The final ordinary binary passes **1,155 units**, ten reduced architecture
configurations and twelve trained prefix/cache/generation regressions across
Smol/Qwen FP32, GPT-2 FP32 and TinyLlama W8A8. The new wheel passes isolated
installed `doctor` and offline source-BF16 generation with two assigned workers,
without compiler/binary overrides or Torch/Transformers imports. The one-core
ordinary non-regression guard is complete under `final-default-model/ordinary.json`
and **fails stability**, despite all pooled/paired median ratios being within
the 1.02 non-regression limits. Smol final-runtime pass 4 fails prefill/decode
spread (1.56863 / 1.30003); Qwen final-runtime pass 4 fails decode (1.52601).
The corresponding pooled ratios are Smol 1.005915 / 1.000541 and Qwen
0.999308 / 0.995908. No sleep overlap. Do not mark this guard passed.
Keep the default's strict numerical failures separate from these regressions.

The unstable Smol pass changes from approximately 500-ms prefill / 82-ms decode
to 307–332 / 59–65 ms near its end; the next unchanged baseline also has a faster
band before returning to approximately 490 ms. This is not established as a
new-code slowdown or as thermal throttling. Worker spinning is inactive in this
one-thread mode (`worker_spin_us=0`). Hold publication while diagnosing the
separate execution-state change.

`one-core-env/` contains a private version-4 dispatch diagnostic. Full 33-token
logits match the final ordinary runtime. The completed four-pass Normal/HighQoS/
HighQoS/Normal trace records actual CPU boundaries, dispatch counters, processor
performance, SMT-sibling activity and faults. It uses the same Above Normal,
AC and display/system-awake policy as acceptance. HighQoS is an experiment on
owned child processes, not a proven remedy or a global power-plan change.
Raw `environment-private.json` contains unrelated process telemetry: keep it
local; publish only a privacy-preserving summary and its raw-record hash.

All four monitored passes are stable, at approximately 493–498 ms prefill and
81 ms decode. CPU boundaries match CPU 2, its SMT sibling is idle in all external
samples, and busy-core performance readings stay approximately 149% of nominal.
HighQoS does not establish a remedy; the faster band was not captured.

`native-clock-v4/audit.json` also completes four passes without an external
sampler. Every warmup/measured prefix and decode has bit-identical repeated
logits. Every measured prefill records 225 linear calls and 19,629,342,720 scalar
multiply terms; per-phase call/work counts are invariant. Native rate counters
use >=1.1-second intervals and record cached-reading age, rather than pretending
they refresh per operation. All four passes are stable around 491–495 ms /
80.6–81.3 ms; the faster band remains uncaptured. No sleep overlap. This verifies
repeated-work integrity in the observed state, not a cause for the earlier burst.

Earlier private audit attempts are retained: a guarded source-anchor abort,
a missing SDK-header compile failure, and `native-clock-v3/audit.json` rejecting
invalid over-frequent rate-counter readings. That failed runner did not persist
its first metrics before validation; v4 fixes this by saving raw metrics first.
None of these attempts is acceptance evidence. The display-awake diagnostic CLI
addition passes 68 focused diagnostics/power tests.

`build_spin.py` now defaults to historical source `31fff42`, so the isolated
experiment remains reconstructible after integration. Its originally executed
builder is retained at `bounded-spin/build_spin-original.py`.
The revised builder reconstructs both generated translation units and the
worker header byte-for-byte and passes the 130 native cases again under
`spin-reconstruction/`.

`benchmark/experiments/worker-placement/validate.py` owns the predeclared gates.
Outputs are under `build/worker-placement-20261022/model/`:

- `quality.json`: complete/passing, bound to the new accurate binary.
- `placement.json`: same new binary, four allowed P-cores 2/4/6/8, off/on ABBA.
- `default.json`: old/new unbound, one P-core 2; run after placement finishes.

Additional records: `jobs-model/{quality,jobs}.json`,
`jobs-ordinary-regression.json`, `remaining-jitter/smol-diagnostic.json` and
`remaining-jitter/diagnose-runner.py` (exact earlier runner bytes). The next
private diagnostic is `queue-jitter/`; keep every attempt, including failures.

Each stage uses six balanced pairs/model, 31 samples and ten warmups/pass. No
filtering. All individual and pooled p90/p10 <=1.25. Placement requires paired
and pooled prefill <=0.98, decode <=1.02 for a speed claim. Default requires both
phases <=1.02 for non-regression. Every sample has native AC checks; parent AC
monitoring, scoped display/system-awake requests and sleep audits are required.
No builds, conversions or other tests may overlap either timing stage.

Do not treat an ON-only stability pass as a joint performance pass. If the OFF
stage is unstable, report that separately and do not claim a percentage speedup.
Placement remains optional regardless of the result. Read the complete JSON
verdict before publication; preserve partial/failed attempts rather than overwriting.

## Local resume paths

- `build/worker-placement-20261022/accurate-final/leaf_decoder.exe` and `build.json`.
- `default-runtime.txt`: final ordinary binary path.
- `units-final.xml`, `ordinary-parity.json`, `worker_affinity_test_final.exe`.
- `wheels/`, `installed/`, `installed-check.json`, `verify_installed.py`.
- `historical-schedule/`: historical private rebuild; no current engine edits.

Retain compact evidence and commit locally. Publication is held because the
ordinary one-core guard remains unqualified. Do not rerun unchanged A/B until it
passes, waive the failed guard, or claim a root cause from stable diagnostics.
The less intrusive probe below is complete. Keep all failed records. After the
release decision, use a PR and merge only after every CI check passes; remove
only the confirmed merged branch.

User steering: immediately after this worker/job work, fix the ordinary FP32
Smol/Qwen strict logit-tolerance failures. The accurate K=256 candidate already
passes full quality; its shared runtime policy and performance cost need a
product solution. Do not describe placement as fixing those numerical errors.
Keep dataset expansion deferred.

Compact public-safe evidence is retained under
[`worker-placement/results/20261022`](../benchmark/experiments/worker-placement/results/20261022).
Its 411-entry archive is verified against the accompanying manifest. Original
process inventories and temporary unit fixtures stay local. No push, PR, merge
or branch deletion has occurred for this work.

## Minimal replay and required next action

`minimal-work-v2/replay.json` completes the original twelve-pass ABBA pattern,
with the unmodified old binary BEFORE and a minimal instrumented build AFTER.
The latter has no PDH, EnableThreadProfiling or ReadThreadProfilingData imports
(checked in its import table), and operator-timing scopes compile to no-ops.
Thread CPU time/cycles, coarse GetTickCount64 wall time and actual CPUs remain.
Two distinct 64-token prompts first match independent cold decoder references.
All warmup/measured outputs subsequently match cold references exactly.

Each measured AFTER prefill checks **actual packed-kernel tile coverage** against
requested work: 19,896,729,600 padded multiply terms, in addition to 225 linear
calls and 19,629,342,720 nominal linear terms. Pack token/column/panel shapes
are checked before dispatch. This distinguishes requested work from visited
tiles; it is not a hardware retired-instruction counter. Every coverage check
passes and all AFTER spreads pass.

The final **unchanged BEFORE** pass nevertheless reproduces instability:
prefill p90/p10 **1.30303842**, including a 331.461-ms sample among roughly
490–510-ms samples. It has no native CPU/work instrumentation. The minimal AFTER
passes remain around 491–506 ms. The fast band was therefore reproduced in the
original binary, not attributed by the added counters. This diagnostic does not
supersede the failed ordinary release guard. No sleep overlap.

No processor firmware-limit Event 37 is logged during the checked window; that
does not rule out throttling or supply a temperature measurement. The previously
requested administrator ETL is absent. This process is not elevated and lacks
`SeSystemProfilePrivilege`, so kernel tracing requires the user's Administrator
PowerShell. No recording is active.

The concrete next probe is ready:

```powershell
cd C:\Users\navaneeth\leaf
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\trace_worker_spread.ps1
```

The script replays the **unmodified** binaries and same cached Smol workload,
retaining owned PIDs/UTC windows for event attribution. Installed CPU.Light,
Power.Light and Thermal.Light profiles are verified. It uses a random named WPR
instance and only stops/cancels that instance; ownership is persisted before
the replay. Existing recording/results are preserved. It checks AC, prevents
display/system sleep, keeps all samples and saves the ETL locally in ignored
`build/worker-placement-20261022/administrator-trace/`. Kernel tracing may alter
timing; this is attribution, not promotion. Actual privileged start/stop remains
untested here. Syntax and `-CheckOnly` preflight pass, with about 196 GiB free.
While the user runs it, do not compile, test, load models, or run another benchmark.
Keep the Administrator window open until it finishes. Do not publish its ETL or
other-process inventory; publish an aggregate and raw-record hash instead.

## Completed administrator trace and explicit efficiency comparison

The named recording starts/stops successfully and all twelve original Smol
passes complete. Microsoft TraceEvent 4.0.0 decodes 17,431,184 events with zero
reported loss. Context-switch chains on logical CPUs 2 and 3 have no mismatches.
The first unchanged baseline fails prefill spread at 1.26983587. The other eleven
passes have passing spreads; the diagnostic still is not acceptance timing.

During first-pass seconds 7–9, delivered-performance event medians rise to
248–254, coincident with `Constrained` → `LowLatency` → `Constrained` PPM
transitions. Later steady passes report medians of 154. The first decoder gets
97.10% scheduled CPU 2 time over its elapsed process window; the later decoders
get 98.57–99.24%. This directly corroborates a CPU power-management contribution
to the captured swing. It does not prove thermal throttling, identify the app
that requested the profile, or explain every historical outlier. Values are
decoded event distributions, not normalized acceptance scores.

Post-trace, the documented effective-mode callback reports `better-battery`,
while AC is online and the legacy `GetSystemPowerStatus` saver flag is false.
The user's Settings screenshot independently shows Energy saver on and Best
Power Efficiency locked. That mode was not continuously recorded in the earlier
trace; do not retroactively assert its identity for every trace sample.

`leaf.power.benchmark_power_mode_guard` now registers one read-only V2 callback,
waits for the initial state, retains notifications, and unregisters synchronously.
Unknown modes or any transition invalidate the window. Saving modes require an
explicit exact `required_mode`, so intentional efficiency-mode comparisons are
allowed without changing the user's configuration. An unconfirmed unregister
retains callback ownership instead of freeing an address Windows might call.
The guard does not lock frequency or replace the p90/p10 gate. Focused tests:
`power-modes-explicit-units.xml`, 97 passed. Real unprivileged API preflight:
`power-mode-preflight-matched.json`, passed with `better-battery` explicitly
selected. The default preflight correctly rejects that mode when unselected.

The fresh ordinary acceptance command is:

```powershell
python benchmark/experiments/worker-placement/validate.py ordinary --directory build/worker-placement-20261022/efficiency-default-model --binary build/worker-placement-20261022/final-cache/native/6ad3e278b798b3ae/leaf_decoder.exe --baseline build/source-bf16-20261020/default/leaf_decoder.exe --baseline-quality build/worker-placement-20261022/final-ordinary-regression.json --required-power-mode better-battery --settle-seconds 30
```

Do not rerun into that directory or overwrite its records. No builds, conversions,
other tests or monitoring shells overlap this run; progress is read through the
existing exec session. Smol's pooled/paired ratios pass non-regression, but two
baseline passes slow to about 1,067 ms prefill, so its spread gates fail. Retain
this failed attempt; settling plus effective-mode monitoring is not yet a proven
stability fix. Testing was then stopped at the user's request, after seven Qwen
passes. The raw record remains `complete:false, passed:false`; it is not edited
to invent a completed run. No native decoder remains running. The operating
system releases process-owned execution requests/notifications when the stopped
process exits. The separate stop record retains the reason and counts. Keep all
nineteen completed passes; five Qwen passes and the final sleep/mode audits were
not completed. No power setting was changed.

Curated `trace-evidence.zip` has sixteen entries, SHA-256
`8bf0d22db3342295c63f25d9d39b040ff7bb42bec83987a1cd6b83b45d0630ee`.
Its manifest and summary append earlier evidence. Raw ETL, decoded catalogues,
other-process metadata and .NET packages are excluded. Reader/analysis sources
and pinned parser dependency identities are retained for reconstruction. The
initial archive records the 94-test guard snapshot; later explicit-mode changes
and their 97-test proof are retained separately. The user now authorizes pushing
the branch as a checkpoint; no merge is justified by these timings. Preserve
the unrelated embedding edit and image. Resume with shared ordinary FP32
Smol/Qwen accumulation-quality work; before another timing comparison, add
clock/profile attribution for the retained efficiency mode instead of repeating
an unchanged quiet-run hypothesis.
