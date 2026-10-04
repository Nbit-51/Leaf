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
$env:LEAF_EXPERIMENTAL_FLOAT_TILES = '1'
python tools/diagnose_windows_benchmark.py --executable build/prefill-diagnostics/leaf_decoder.exe --artifact build/gpt2_prefill_followup/decoder-32.leaf --tokens build/gpt2_prefill_followup/tokens.json --cpu 2 --monitor-cpu 27 --runs 31 --warmup 10 --high-qos-abba --output build/gemm-followup/environment-raw/new-diagnostic.json
python tools/summarize_windows_diagnostics.py --inputs build/gemm-followup/environment-raw/new-diagnostic.json --output build/gemm-followup/new-diagnostic-summary.json
Remove-Item Env:LEAF_EXPERIMENTAL_FLOAT_TILES
```

Windows API references: [CPU-set topology and efficiency class](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_cpu_set_information),
[formatted wildcard performance counters](https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw),
[counter collection limits](https://learn.microsoft.com/en-us/windows/win32/perfctrs/about-performance-counters),
and [scoped process power policy](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation).
