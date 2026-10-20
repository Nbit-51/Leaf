# Source-preserving BF16 storage

The native decoder now supports optional original BF16 output-head/tied-embedding
storage through `leaf run --source-bf16-head`. Values expand exactly for FP32
computation. FP32 source weights stay FP32. See the
[implementation, quality and timing scope](../../../docs/source-bf16.md).

## Original isolated probe

The real Smol/Qwen vocabulary projections are bit-identical to FP32 storage
and about 38% faster in this probe, streaming half the weight bytes. Both arrays
coexist in the probe, so this is not a process-memory reduction.
[Full scope and next checks](../../../docs/checkpoint-2026-10-19-linear-and-storage.md).

Model paths come from `build/numerics-20261017/model-paths.json`. Run serially
from the repository root with fresh output folders:

```powershell
# Correctness only; can run on battery.
python benchmark/experiments/source-bf16/run_probe.py --check-only --output build/source-bf16-check-fresh
# Timing: AC connected, no heavy background work.
python benchmark/experiments/source-bf16/run_probe.py --output build/source-bf16-timing-fresh
```

Native timing is pinned to CPU 2 at above-normal priority. Every sample checks
AC before and after. Source hashes, exact-output checks and all samples are kept.

## Native integration checks

`integration.py` audits every exported tensor, rebinds complete trained quality
by bit-exact full-logit comparison, checks generation/cache/scalar/thread paths,
then performs serial matched timing. It requires the pinned snapshots, FP32
artifacts and complete prior accurate-candidate quality records described in the
October 19 checkpoint; it is not a model download utility.

```powershell
python benchmark/experiments/linear-accumulation/build.py --output build/source-bf16-fresh/accurate-final --precise-scalar --tree-outer --tree-chunk-size 256 --reuse-packing
python benchmark/experiments/source-bf16/integration.py prepare --directory build/source-bf16-fresh
python benchmark/experiments/source-bf16/integration.py check --directory build/source-bf16-fresh
python benchmark/experiments/source-bf16/integration.py screen --directory build/source-bf16-fresh
python benchmark/experiments/source-bf16/integration.py acceptance --directory build/source-bf16-fresh
```

The predeclared decode gate is paired ratio <=0.98, prefill <=1.02, and every
pass/pooled p90/p10 <=1.25. Six balanced pairs/model, 31 measurements and ten
warmups/pass; one thread, CPU 2, above-normal priority. All samples are retained.
The temporary system/display request ends with the benchmark. An AC monitor
checks both policies; the new binary additionally checks AC per native sample.
Windows sleep-event auditing rejects an attempt that overlaps standby.
`--attempt NAME` writes a distinct timing report without replacing a failed run.

The first October 20 acceptance attempt predates the display/sleep-event guard.
Its exact runner is preserved alongside the results; the companion sleep audit
invalidates it after a recorded Modern Standby interval. Do not treat the passing
Smol summary in that attempt as a separate qualified release.
