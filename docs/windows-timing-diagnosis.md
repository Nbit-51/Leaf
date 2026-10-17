# Windows timing diagnosis — 2026-10-04

**The historical 2× timing jump still has no proven root cause.** It was not
reproduced with telemetry attached. Current evidence does not justify blaming
VS Code, declaring thermal throttling, or changing the stability threshold.
The user reports browsing ChatGPT/LeetCode during the earlier fluctuations,
and compiling C++ during the beginning of this investigation. Compilation
therefore cannot explain all earlier noise.

## What was measured

Host: Intel Core i7-14700HX, 20 physical cores / 28 logical processors,
approximately 16 GiB RAM, connected to AC, Acer power plan. The plan exposes
5% minimum and 100% maximum processor state for both AC and battery. Those
settings are configuration, not evidence of actual clock speed or throttling.

Windows CPU-set information places logical CPU 2 on the higher-performance
class; CPU 3 shares its physical core. The diagnostic pins the native child
to CPU 2, verifies its affinity, and pins telemetry to a different physical
core (logical CPU 27). Ordinary timing already pins CPU 2; this rules out an
accidental efficiency-core selection, not contention or power management.

The [published environment summary](../benchmark/results/gpt2_windows_environment_summary.json)
retains timing samples and aggregate telemetry. Raw process inventories remain
local in `build/gemm-followup/environment-raw/`; their SHA-256 hashes identify
them without publishing unrelated application names or process IDs.

Two diagnostic runs are retained, both with the same scalar-GELU executable,
packed GEMM, 63-token prefill, one-token cached decode, one thread, ten warmups
and 31 samples per pass. Their order is normal / HighQoS / HighQoS / normal.
HighQoS affects only the owned child process. No global power plan or process
priority was changed, and no other application was stopped by the tool.

| Diagnostic | Normal prefill median ms | HighQoS prefill median ms | Combined stage stability |
|---|---:|---:|---|
| Initial probe, apps open | 212.8985 | 213.6807 | Both pass; probe has high overhead |
| Revised probe, user closed apps | 216.0566 | 214.0451 | Both pass |

The first probe used repeated process enumeration and consumed several seconds
of Python CPU time per pass. It is not clean acceptance timing. The revised
probe uses bulk Windows PDH counters: collection takes roughly 15–18 ms per
one-second sample; Python CPU consumption is about 0.17–0.22 seconds per pass.
The instrumentation change and the user closing apps are confounded, so these
two runs do **not** establish the causal effect of closing applications.

During the revised quiet diagnostic:

- All four individual passes pass stability, with prefill medians 213.9215,
  214.0782, 214.0120 and 218.9331 ms.
- CPU 3 is idle in all samples. CPU 2's processor-performance counter stays
  near 150% while the child is busy (observed range 148.70–151.54%).
- Available memory is at least 6.65 GiB, versus about 2.70 GiB in the initial
  open-app diagnostic. System page-read rates are mostly zero; sampled spikes
  are not evidence that the decoder is paging.
- Both normal and HighQoS stages pass. Their prefill difference is under 1%,
  so there is no demonstrated large HighQoS remedy in this state.

The monitor includes loading/warmups and uses one-second intervals; it cannot
attribute a 30 ms decode outlier to a specific context switch. Process CPU
totals cannot locate a competitor on CPU 2, and short-lived processes can be
missed. No temperature, package-power, or thermal/power-limit sensor was read.
The reported 2100 MHz nominal frequency is not treated as a live-clock trace.

## Unmonitored qualification still fails

One [quiet scalar/vector GELU ABBA](../benchmark/results/gpt2_gelu_quiet_windows_abba.json)
was run after the monitored control, with telemetry disabled and the existing
quality and timing gates unchanged. Quality passes and observed prefill falls
18.56%, but the first BEFORE decode p90/p10 ratio is **1.26092**, exceeding
**1.25**. All other phase/pass checks and between-pass drift checks pass.
The full experiment is correctly rejected; no further A/B retries were made.

Closing applications is thus useful for establishing a quieter measurement
state, but it has not eliminated all noise or validated the previous speedup.
The first open-app diagnostic was also stable, so having VS Code open alone
is not established as sufficient to produce the historical slowdown.

## Fresh PyTorch comparison and optimization target

A [fresh matched comparison](../benchmark/results/gpt2_gelu_quiet_matched_windows.json)
retimes PyTorch eager/SDPA and experimental Leaf in separate processes on CPU
2, one thread, 31 samples / ten warmups, identical prefix/cache semantics and
last-token logits. Frozen quality caches and model/data identities are checked.
This uses **packed GEMM plus experimental vector GELU**, not Leaf's default.

| Implementation | Prefill median ms | Decode median ms | Timing stability |
|---|---:|---:|---|
| PyTorch eager | 169.0865 | 30.0117 | Both fail |
| PyTorch SDPA | 169.0666 | 29.9585 | Decode fails |
| Experimental Leaf FP32 | 183.7884 | 31.9989 | Both pass |

Leaf's observed gap to the fastest PyTorch median is **8.71% prefill and 6.81%
decode**. These are provisional descriptive ratios: SDPA decode p90/p10 is
1.27864 and fails. There is no qualified parity/lead claim and no automatic
promotion. This snapshot measures implementations sequentially rather than
framework-level ABBA, another reason not to interpret a small ratio causally.

The useful engineering target is now explicit: approximately 15 ms less
prefill and 2 ms less decode in this workload. The previous experimental
profile puts linear operations around 81%, attention around 14%, and
activation around 1%. Further GELU tuning is unlikely to close that gap alone.
The next optimization should address linear/attention cost and separately
protect one-token GEMV, with shape measurements before whole-model integration.
Those phase shares are earlier diagnostics, not a decomposition of this new
timing run, and the default build's performance must be reported separately.

## What would establish a root cause

The large slowdown needs to be captured with simultaneous telemetry. A fall
in processor performance suggests frequency/power behavior; lost scheduled
CPU time suggests competition; activity on CPU 3 suggests shared-core pressure.
These are hypotheses to test, not findings from the current quiet trace.
For the remaining short decode outliers, an ETW scheduler trace is more
appropriate than increasing the performance-counter sampling rate. A
temperature/power-limit trace is needed before attributing a slowdown to
thermal throttling. No global system tuning is justified by the evidence yet.

## October 15, 2026: diagnostic startup race

A reproducible runner defect contaminated some monitored samples. Python was
pinned to monitor CPU 27, spawned the decoder (which inherited CPU 27), and
only then changed the child's affinity to CPU 2. The native warmup and sample
loop could already be running. Checking the later process affinity did not
verify where earlier work actually executed.

Counter version 3 records the actual logical CPU at both boundaries of each
prefill and decode. Replaying the saved old launch sequence with this build
captured the defect in pass 1 (zero based):

| Phase / measured sample | Start CPU | End CPU | Wall time | Process page faults |
|---|---:|---:|---:|---:|
| Prefill 0 | 27 | 27 | 135.239 ms | 0 |
| Decode 0 | 27 | 27 | 30.0789 ms | 0 |
| Prefill 1 | 27 | 2 | 69.2652 ms | 0 |

CPU 27 is an E-core; CPU 2 is a P-core. Ordinary P-core medians in this replay
are about 52 ms prefill and 10 ms decode. All four replay passes nevertheless
pass their p90/p10 checks: a small number of contaminated samples can lie outside
those percentiles. No samples were removed. The replay is evidence of a
measurement defect, not performance qualification.

The corrected runner temporarily pins its parent to CPU 2 before creation,
then restores the monitor to CPU 27. A diagnostic-only start gate holds the
child until priority, optional HighQoS and PDH initialization finish. Each
sample's recorded CPU boundaries must match the requested CPU, or the run is
retained as a failure. Runtime capability checks reject older diagnostic
binaries. Ordinary decoder builds do not read or wait on the gate.

One corrected off/on/on/off HighQoS run used 61 samples and ten warmups per
pass, Above Normal priority, Pythia-70M FP32, 63-token prefill and one cached
decode. All **976 CPU boundary observations** were CPU 2; all four passes
passed spread checks. Prefill medians were 52.45, 52.32, 53.46 and 52.06 ms;
decode medians were 10.26, 10.22, 10.74 and 10.13 ms. The earlier threefold
startup spikes did not recur. This does not demonstrate a HighQoS benefit.

The changed diagnostic build passed start-gate wait/release checks and
**3,219,456 byte-identical trained-model logits** against the preceding
diagnostic build. The 88 focused tests cover CPU mismatch rejection, counter shapes,
affinity inheritance/restoration on spawn failure, and sanitized reporting.

The separate acceptance A/B runner already pins before spawning. This defect
therefore **does not explain every historical acceptance failure**, and failed
kernel experiments remain failed. Calling-thread CPU time is coarse on this
machine; short per-phase CPU/wall differences alone cannot prove preemption.
Cycle counts also are not a direct live-frequency measurement.

Windows Kernel-Power events record Modern Standby from **15:34:41 to 15:42:43
IST**. The Ollama comparison finished at 15:27:39; the preceding Pythia
diagnostic finished at 15:31:44. Neither overlaps that interval. The corrected
run and replay occurred after resume. This sleep observation cannot explain
those spikes or older runs the user kept awake.
[Modern Standby](https://learn.microsoft.com/en-us/windows-hardware/design/device-experiences/modern-standby)
includes screen-off and low-power segments; a working application view alone
does not establish continuous benchmark execution. The scoped
[execution request](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate)
does not prevent user-initiated sleep.

Retained [sanitized summaries](../benchmark/results/timing-spread-20261015/environment-summaries.json)
include every native sample, CPU boundaries, gates and raw-record hashes,
without publishing the process inventory. The local raw traces remain under
`build/timing-spread-20261015/`. Sleep/run bounds are retained in the
[correlation record](../benchmark/results/timing-spread-20261015/power-correlation.json).
These are instrumented diagnostics, not a newly qualified speedup.

## Acceptance-runner follow-up — October 15, 2026

The separate acceptance launch path still reproduces scattered variability.
Version-4 diagnostic builds add Windows thread-dispatch context-switch counts
and wait-reason masks around every forward, with no system-wide trace or PDH
monitor. A six-pair replay uses the same `run_native` launch, inherited CPU 2
affinity and Above Normal creation priority as the acceptance runner. All raw
samples and actual CPU boundaries are retained.

In pair 2 AFTER, decode sample 11 takes **16.6991 ms** versus a pass median
**10.5587 ms**, while its thread-cycle count is only **1.04473 times** the
median. It records **54 context switches** versus median 12 and **zero page
faults**. Prefill sample 11 takes 68.4729 ms versus median 51.2091 ms, cycle
ratio 1.03216, with 287 switches versus median 57 and zero faults. Wait masks
include dispatch interrupts. This supports a scheduling/interruption component
in the reproduced spread. Some other slow samples also use substantially more
cycles; these counters do not identify the interrupting driver/process or
exclude cache, shared-core or clock effects. They cannot assign a cause to
every historical outlier.

A balanced six-pair placement control, using the identical binary on P-cores
2 and 14, passes every individual spread check on both cores. It therefore
does not establish changing cores as a cure. A subsequent uninstrumented A/A
control of the identical baseline binary still fails pair 3 BEFORE: prefill
p90/p10 **1.31855**, decode **1.36044**. Both pooled phase distributions pass.
The A/A run also fails the 2% improvement criterion, as expected for identical
binaries; it is a stability control, not an optimization trial. No threshold
was relaxed and no sample was discarded.

[Dispatch replay](../benchmark/results/acceptance-spread-20261015/dispatch-abba.json),
[core placement control](../benchmark/results/acceptance-spread-20261015/core2-core14.json),
[uninstrumented A/A](../benchmark/results/acceptance-spread-20261015/uninstrumented-aa.json).

WPR cannot enable the system-performance profiling policy in this non-elevated
session (`0xc5585011`); it left no recording active. A guarded, administrator-only
`scripts/trace_acceptance_spread.ps1` is prepared for attribution if needed. It
preserves an existing recording, saves its own trace locally and stops it in
`finally`. ETL/process inventories must not be committed. An administrator trace
was not completed in this investigation. Thread-dispatch counters are available
without elevation and are explicitly diagnostic, not promotion evidence.

The later TinyLlama/Ollama rerun was stopped after the user reported mains
failure. Windows records a power-source change at **16:18:17 IST**, overlapping
that rerun. Its partial record remains unqualified. Fresh comparisons now
require known AC state at launch/request boundaries; the native generation
adapter also checks before and after each sample, outside its timers, and aborts
on offline/unknown state. Native A/B checks pass boundaries; neither scheme
claims to detect every possible transient between checks. The new adapter is
capability-checked, so an older binary cannot silently omit the power guard.

The useful compute target is now [FP32 SiLU](silu-gate-experiment.md#fp32-follow-up--october-15-2026),
which remains scalar in the release default. Correctness can be checked on
battery; acceptance and matched-runtime timing wait for mains power. This work
improves diagnosis and prevents power-mixed runs from qualifying; it does not
claim Windows scheduling variability is completely eliminated.

References: [thread dispatch profiling](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-enablethreadprofiling),
[counter meanings](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-performance_data),
[Windows power status](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-system_power_status).

## Reproduction

Validation: 165 focused tests pass, covering the new topology parser and
privacy-preserving summary, existing power/affinity controls, and benchmark
gates. The quiet ABBA verdict replays exactly from raw samples; published
telemetry summaries match their local raw-record hashes. No decoder kernel
or default execution policy changed during this investigation.

Run the diagnostic separately from builds/tests; use a new output filename.
The tool prevents automatic sleep during its lifetime. User-initiated sleep
and lid/power actions remain unaffected.

```powershell
New-Item -ItemType Directory -Force build/prefill-diagnostics | Out-Null
g++ -std=c++17 -O3 -DNDEBUG -DLEAF_DIAGNOSTIC_TIMING=1 -pthread -static -static-libgcc -static-libstdc++ -I engine/include engine/src/decoder.cpp engine/src/main_decoder.cpp engine/src/kv_cache.cpp engine/src/kernels/transformer.cpp -o build/prefill-diagnostics/leaf_decoder.exe -lpsapi
$env:LEAF_EXPERIMENTAL_FLOAT_TILES = '1'
python tools/diagnose_windows_benchmark.py --executable build/prefill-diagnostics/leaf_decoder.exe --artifact build/gpt2_prefill_followup/decoder-32.leaf --tokens build/gpt2_prefill_followup/tokens.json --cpu 2 --monitor-cpu 27 --runs 31 --warmup 10 --high-qos-abba --output build/gemm-followup/environment-raw/new-diagnostic.json
python tools/summarize_windows_diagnostics.py --inputs build/gemm-followup/environment-raw/new-diagnostic.json --output build/gemm-followup/new-diagnostic-summary.json
Remove-Item Env:LEAF_EXPERIMENTAL_FLOAT_TILES
```

Windows API references: [CPU-set topology and efficiency class](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_cpu_set_information),
[formatted wildcard performance counters](https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw),
[counter collection limits](https://learn.microsoft.com/en-us/windows/win32/perfctrs/about-performance-counters),
and [scoped process power policy](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation).
