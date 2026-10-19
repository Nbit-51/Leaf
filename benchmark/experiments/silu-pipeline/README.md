# Exact SiLU pipeline

Four independent AVX2 FP64 Horner chains replace one chain at a time. Per-lane
arithmetic and four-lane exceptional fallbacks are unchanged. The production
header uses this implementation without a new runtime switch.

The native contract test passes 5,257,011 checks, maximum ULP error zero.
The retained reversed-order probe measures 17–19% lower operator latency,
with exact old/new output and alias checks. Short model screens do not qualify
an end-to-end gain. [Scope and evidence](../../../docs/checkpoint-2026-10-19-linear-and-storage.md).

From the repository root, with AC connected:

```powershell
python benchmark/experiments/silu-pipeline/run_probe.py --output build/silu-pipeline-fresh
```

The default baseline is immutable commit `f814501`, not moving HEAD. Override
with `--baseline-ref` explicitly. Existing output directories are refused.
