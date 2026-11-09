# Optional Windows benchmark accounting

The native decoder accepts `--benchmark-counters` in **bench mode only**.
Ordinary inference and ordinary benchmarks do not read these counters. The flag
requires Windows and uses ordinary process APIs; no Administrator trace,
profiling driver, database, or new numerical kernel is involved.

`--runtime-info` advertises `benchmark_counters_supported` and
`benchmark_counter_version`. The Python developer helper accepts
`run_native(..., mode="bench", benchmark_counters=True)` and rejects absent or
malformed requested counters.

## What is measured

The version-one `benchmark_counters` metrics object contains separate `prefill`
and `decode` arrays, aligned with their existing forward wall-time samples.
Warmups are excluded. The process measurements include **all workers**, including
active spinning. Caller measurements cover the benchmark thread.

- `process_cpu_ms`, `caller_cpu_ms`: kernel plus user CPU accounting. Windows
  accounting can be coarse; a short sample can legitimately have a zero delta.
- `process_cycles`, `caller_cycles`: raw integer cycle counts. They are not CPU
  frequency measurements, retired-instruction counts, or a measure of waiting.
- `process_page_faults`: soft plus hard faults, with DWORD counter wrap handled.
  These are not specifically disk faults.
- `interval_ms`: a monotonic bracket including complete counter reads. It is
  distinct from the existing forward timer. Adjacent phase brackets overlap at
  the middle read and must not be added to infer a combined duration.
- `caller_cpu_start/end`, `caller_group_start/end`: processor endpoints of the
  counter reads. Caller affinity is restored after forward, so these endpoints
  do not prove placement or migration during inference.

Process CPU time can exceed wall time with multiple workers. Subtracting it from
wall time does not establish how long a thread was descheduled. Neither these
counters nor the effective-mode guard observe every internal Windows processor
power-management transition. The earlier ETW evidence remains distinct.

## Paired FP32 policy diagnostic

This reusable developer command accepts prepared local artifacts and tokens;
it does not download models. Supply a completed trained quality report bound to
the exact binary and artifact:

```powershell
python tools/benchmark_fp32_math.py --binary PATH_TO_DECODER --artifact PATH_TO_MODEL --tokens PATH_TO_TOKENS_JSON --quality-report PATH_TO_QUALITY_JSON --directory build/math-cost-run --worker-cpus 2 --required-power-mode better-battery
```

Choose an allowed CPU for your host. `tokens.json` must contain `benchmark_ids`.
Each sample resets the session, prefills `ids[:-1]` and decodes the fixed last
token. The default is two balanced AB/BA pairs, seven measured samples and three
warmups per process. `--threads` must match the number of explicit worker CPUs.
Both math modes use the same artifact, workload, process priority and placement.
The command audits AC and effective mode and keeps the display/system awake
without changing persistent power settings. Explicit user-initiated sleep is
still possible; this short diagnostic does not perform a post-run event-log
sleep audit and cannot qualify a release.

`--before-math fast --after-math fast` is an unchanged-policy A/A control.
For an isolated implementation comparison, supply `--before-binary` and
`--before-quality-report`, and select the same math policy on both sides. Each
quality report must bind its own executable and the shared artifact.
`--without-counters` runs the same diagnostic workload without instrumentation.
Use a fresh directory for each run. Completed passes are checkpointed before
post-run power validation; interrupted attempts remain marked incomplete.

The report retains every sample, per-pass and pooled p90/p10 spread checks, and
paired ratios. It always records `acceptance_timing: false` and
`timing_qualified: false`. A diagnostic result cannot promote the default or
create a speedup claim. Release acceptance still requires the existing full
sample counts, numerical qualification and unchanged latency/spread thresholds.

The API semantics follow Microsoft's [process CPU accounting](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes)
and [thread cycle counter](https://learn.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-querythreadcycletime)
contracts. Tiling, weight formats and FP32 math policies are unchanged by this
measurement feature.
