# Source-preserving BF16 storage probe

An isolated vocabulary GEMV experiment, **not native decoder support**. Only
already BF16 source weights are accepted. They expand exactly into FP32 registers
and retain existing paired-GEMV arithmetic. No arbitrary FP32 values are rounded.

The real Smol/Qwen vocabulary projections are bit-identical to FP32 storage
and about 38% faster in this probe, streaming half the weight bytes. Both arrays
coexist, so this is not a process-memory reduction.
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
