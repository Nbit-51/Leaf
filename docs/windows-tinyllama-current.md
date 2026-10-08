# TinyLlama validation and next prefill target — 2026-10-08

The installed Windows runtime from `0d28693` passes trained TinyLlama-1.1B-Chat-v1.0
FP32 and calibrated W8A8 quality checks. FP32 remains slower than PyTorch on this
workload. W8A8 has substantially lower measured medians, but fails timing stability;
this run does **not** qualify a new stable TinyLlama speedup or automatic precision
selection. The independently qualified GPT-2 default remains enabled.

## Completed comparison

One thread on logical CPU 2 (a physical P-core), Above Normal priority, Windows,
batch one, 63-token prefill and one cached decode token, last-token-only logits.
Each runtime runs twice in fresh serial processes, forward then reverse order:
Leaf FP32, Leaf W8A8, PyTorch eager, SDPA, SDPA, eager, W8A8, FP32.
Each pass has ten warmups and 31 retained measurements; medians below pool 62
samples. No correctness tests, builds or profiling overlapped these measurements.
PyTorch is `2.12.0+cpu`, with one compute and one interop thread. Model weights,
tokens, artifact provenance and executable hashes are checked at both boundaries.

| Runtime | Prefill median | Decode median | All stability checks |
|---|---:|---:|---|
| Leaf FP32 | 1,944.3069 ms | 225.5490 ms | Pass |
| Leaf calibrated W8A8 | 1,381.9081 ms | 92.70555 ms | **Fail** |
| PyTorch eager FP32 | 1,800.6946 ms | 213.2119 ms | Pass |
| PyTorch SDPA FP32 | 1,881.60295 ms | 243.57595 ms | **Fail** |

Leaf FP32 is 7.98% slower on prefill and 5.79% slower on decode than the stable
eager baseline. W8A8's medians are 23.26% and 56.52% lower respectively, but those
are descriptive results, **not a qualified stable performance claim**.

Why stability fails, against the unchanged p90/p10 limit of 1.25:

- W8A8 first-pass decode: **1.2680**; second-pass prefill: **1.3203**.
- W8A8 pooled prefill/decode: **1.3283 / 1.3676**.
- SDPA passes individually, but pooled decode is **1.3021**. Its two prefill
  medians move from 1,752.34 to 2,042.75 ms; decode moves from 219.79 to 254.84 ms.
- All between-pass median ratios are below 1.25. That alone does not excuse
  failing individual or pooled sample spread.

The second W8A8 pass includes three consecutive prefill measurements of roughly
2.38, 3.04 and 2.74 seconds, alongside much faster intervals. Samples are retained.
This run does not record per-inference CPU time, page faults or clock telemetry,
so it cannot identify preemption, paging, clock changes or another cause. Do not
assign a cause from wall-clock samples alone or repeatedly rerun until a pass.

## Quality and memory

The completed quality check reuses the frozen source-matched PyTorch reference,
eight 128-token sequences, 1,016 scored targets, chunked-cache checks and a
16-token generation. Timing uses the same installed executable and artifacts.

| Precision | Next-token agreement | Perplexity / reference | Chunked parity | Generation exact |
|---|---:|---:|---|---|
| FP32 | 100% | 0.99999989 | Pass | Yes |
| Calibrated W8A8 | 95.0787% | 1.01239873 | Pass | No |

FP32 reference logits pass atol/rtol 2e-3, maximum observed absolute error
0.0004683. W8A8 passes the existing >=95% agreement and <=1.02 perplexity-ratio
limits. This is limited held-out coverage, not universal output equivalence.

Artifact sizes are 4,400,212,544 bytes (FP32) and 1,103,777,152 bytes (W8A8), a
74.92% reduction. This comes from quantization, not pruning. These sizes are not
whole-process memory or a measured PyTorch memory comparison. The raw native
timing passes also retain peak working-set measurements.

## Measured hot paths and next experiment

After acceptance timing ended, each shipped Leaf configuration was profiled in
two fresh processes with five measurements and two warmups. Instrumented means
include warmup and first use. They locate work; they do not replace uninstrumented
latency or establish incremental speedups.

| Prefill phase | GPT-2 FP32 | TinyLlama FP32 | TinyLlama W8A8 |
|---|---:|---:|---:|
| Linear operations | 92.7–92.8% | 73.1–73.2% | 50.0–50.1% |
| Activation / gating | 1.1–1.2% | **24.8–25.0%** | **38.8–39.1%** |
| Attention | 3.2–3.3% | 1.3% | 9.9–10.1% |

TinyLlama activation/gating costs 546.7–551.1 ms in the FP32 profiles and
459.6–517.2 ms in W8A8. The gated decoder loop still calls scalar SiLU for each
element; the vector GELU-new path serves non-gated GPT-2. This is a concrete
new target, not evidence that all GPT-2 kernels failed to transfer. Decode is
still dominated by linear operations: about 93% for FP32 and 87% for W8A8.

Next implementation order:

1. Add and test a vector SiLU-plus-gating primitive with exceptional-value/scalar
   handling. Measure the actual gated MLP sizes; then verify trained logits,
   chunked cache and generation for both TinyLlama precisions, followed by a
   serial whole-model comparison. Do not infer the gain from the profile alone.
2. Separately evaluate vector prefill attention for quantized weights. Attention
   tensors are FP32, but current optimized dispatch requires an entirely FP32
   artifact; W8A8 therefore uses the older attention path. Enabling it needs its
   own quality and performance checks, not a blind dispatch change.
3. For persistent timing spread, collect per-sample wall/CPU time and fault
   deltas in a diagnostic run before another acceptance rerun. Existing samples
   cannot establish the cause. Do not relax the thresholds.

GPT-2 remains linear-dominated after its accepted attention improvement. The
earlier GPT-2 GEMM-headroom result does not settle TinyLlama's larger projection
shapes; revisit those shapes if linear work remains the next measured target.
No runtime kernel or architecture path changed during this follow-up, so the
existing Mermaid execution diagrams remain accurate.

## Evidence and reproduction

- [Raw timings](../benchmark/results/tinyllama_current_pytorch_windows.json)
- [Trained quality](../benchmark/results/tinyllama_current_quality_windows.json)
- [Combined gates and evidence hashes](../benchmark/results/tinyllama_current_validation_windows.json)
- Profiles: [GPT-2](../benchmark/results/gpt2_current_profile_windows.json),
  [TinyLlama FP32](../benchmark/results/tinyllama_current_fp32_profile_windows.json),
  [TinyLlama W8A8](../benchmark/results/tinyllama_current_w8a8_profile_windows.json)

The tools checkpoint completed passes and require fresh output paths. The
resumed quality run was already complete and was not repeated. Focused harness,
provenance, profiling and comparison regression tests: **251 passed**.

```powershell
python tools/verify_cached_decoder.py --executable <installed-leaf-decoder> --model <local-TinyLlama-snapshot> --workdir build/tinyllama_windows_fair --output <fresh-quality.json>
python tools/benchmark_core_scaling.py --executable <installed-leaf-decoder> --model <local-TinyLlama-snapshot> --workdir build/tinyllama_windows_fair --output <fresh-timing.json> --cores 2 --counts 1 --default-kernels --include-smoothed-w8a8 --runs 31 --warmup 10
python tools/profile_decoder.py --executable <installed-leaf-decoder> --artifact build/tinyllama_windows_fair/decoder-8-smooth.leaf --tokens build/tinyllama_windows_fair/tokens.json --output <fresh-profile.json> --cpu 2 --threads 1 --runs 5 --warmup 2 --default-only --activation-bits 8
```
