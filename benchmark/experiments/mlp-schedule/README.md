# Parallel prefill job sizing

**Not released.** Output checks pass, whole-model acceptance fails. The default
engine is unchanged. `candidate.patch` and `build_candidate.py` reconstruct this
version only under ignored `build/`. The follow-up caught an intermittent
worker-placement problem on this Windows host; see the
[checkpoint](../../../docs/checkpoint-2026-10-20-prefill-jobs.md) and
[retained evidence](results/20261020). Neither is a new competitor win.

Reconstruction reads engine revision `880139d` from Git history, so subsequent
runtime changes cannot silently alter this historical experiment. A full clone
containing that commit is required; the current engine stays untouched.

The decoder's existing workers dispatch 32 output rows at a time. Its FP32 GEMM
uses six-row tiles, leaving a two-row tail in every full job. The candidate uses
96-row jobs when there are at least four jobs/worker, 48-row jobs for smaller
projections, and the existing 32-row jobs otherwise. Single-thread work, GEMV,
scalar execution and quantized linear paths retain their scheduling policy.
The same rows are computed with the same per-output FMA and reduction order.

`model.py` checks all eight prior fully qualified trained blocks/model, six
reference continuations/model, and scalar/cache/thread exclusions. It then
compares the previous and candidate accurate K=256 binaries on the **same four
physical P-cores**. This is an incremental job-scheduling comparison, not a
one-core/four-core speed claim or a fresh competitor comparison.

The predeclared gate requires both pooled and median-paired prefill ratios <=0.98,
decode ratios <=1.02, and all pass/pooled p90/p10 <=1.25. Six balanced pairs/model,
31 samples and ten warmups/pass. Native per-sample AC checks, a monitoring thread,
a scoped display/system request and Windows sleep auditing are required. Windows
event-log access can require an unrestricted process; a failed audit does not pass.

## Resume the current candidate

From the repository root, with pinned cached models from the source-storage
checkpoint and a fresh directory:

```powershell
python benchmark/experiments/mlp-schedule/build_candidate.py --output build/prefill-jobs-fresh
python benchmark/experiments/mlp-schedule/model.py check --directory build/prefill-jobs-fresh/model --binary build/prefill-jobs-fresh/accurate/leaf_decoder.exe
```

Do not retry acceptance unchanged. Address worker placement first. `model.py
timing` retains the original gate; never overwrite or filter its failed attempts.

The placement control uses the same instrumented binary in off/on/on/off order.
Actual Leaf worker IDs are registered and startup is gated, excluding Windows
loader/service threads. ON pins the four owned threads individually to 2/4/6/8;
the monitor stays on CPU 27. No global power settings change. CPU time/cycles,
faults, per-core performance/busy counters and parked flags include load/warmup.
This is diagnostic instrumentation, not acceptance timing.

```powershell
python benchmark/experiments/mlp-schedule/diagnose.py build --source build/prefill-jobs-fresh/accurate --directory build/prefill-jobs-fresh/placement
python benchmark/experiments/mlp-schedule/diagnose.py run --directory build/prefill-jobs-fresh/placement --model smol --strict-affinity-abba
python benchmark/experiments/mlp-schedule/diagnose.py run --directory build/prefill-jobs-fresh/placement --model qwen --strict-affinity-abba
```

`GetProcessTimes` sums all thread CPU time; accounting is coarse on this host.
Cycle counts are not an independent frequency measurement. The Windows
[parked flag](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_cpu_set_information)
describes current state; startup topology alone cannot establish what happened
during inference. [Thread affinity](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadaffinitymask)
is restricted to the owned child's allowed processors.

The completed first Smol proof was retained after a harness argument error:
`verify` still requires a positive chunk argument. `--resume-quality` accepts only
unchanged bound binaries/fixtures and the retained original runner snapshot. It
reuses exact cases, never treats the incomplete report itself as a passing gate.
The corrected report must finish every remaining case before timing.

## Earlier probes, not released

`build.py`, `probe.cpp` and `run.py` retain a separate two-step K-loop experiment
using GCC's documented [loop-specific pragma](https://gcc.gnu.org/onlinedocs/gcc/Loop-Specific-Pragmas.html).
Six-row and five-row versions each pass 454 scalar/vector exactness cases. Neither
showed an eligible operator speed gain, so neither was integrated into the decoder.
The first six-row timing record also has a rejected event-log audit and remains
incomplete. Its samples and source snapshots remain in the local checkpoint.

`build_queue.py`/`queue_probe.cpp` isolate worker partitioning with unchanged GEMM
math; they are diagnostic probes, not release timing gates. The refined probe
passes 324 exactness cases and has mixed stability. Whole-model acceptance is the
decision point. Large binaries and weights stay in ignored `build/`.
