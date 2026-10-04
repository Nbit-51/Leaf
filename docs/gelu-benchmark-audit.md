# GELU inference-error and benchmark audit — 2026-10-04

The observed 21–26% reduction concerns **whole-model prefill latency** with
vector GELU enabled. It is not an accepted speedup, training improvement, or
21% reduction in numerical error. All four previous comparisons failed the
combined timing-stability gate. This audit keeps that verdict unchanged.

## Numerical accuracy

These are inference checks on a frozen trained GPT-2 checkpoint. No weights
are trained, no optimizer runs, and no gradients or backward GELU are tested.
The held-out subset is eight contiguous 128-token WikiText-2 test blocks:
1,016 next-token targets, excluding the final unscored row in each block.
This is a small implementation-quality subset, not full-dataset perplexity.

The [fresh paired audit](../benchmark/results/gpt2_gelu_error_benchmark_audit.json)
uses the same Windows binaries, FP32 artifact, token cache, reference hashes,
and packed-GEMM policy as the saved final comparison. An independent PyTorch
float64 cross-entropy calculation reproduces our NumPy perplexities within
`rtol=1e-12, atol=1e-12`. A separate regression test checks block boundaries,
next-token shifting and large logits against analytically known losses.

| Metric | Scalar GELU | Vector GELU |
|---|---:|---:|
| Perplexity | 62.5800533447 | 62.5800533635 |
| Next-token agreement with PyTorch | 100% | 100% |
| Maximum absolute logit error vs PyTorch | 0.00465393 | 0.00366211 |
| Logit RMSE vs PyTorch | 0.000158671 | 0.000135661 |
| Largest fraction of existing allclose tolerance | 25.79% | 26.35% |

PyTorch reference perplexity is 62.5801026507. The gate is elementwise
`abs(actual-reference) <= 0.002 + 0.002*abs(reference)`; an absolute logit error
above 0.002 alone does not violate it. The largest absolute error and the
largest normalized error can occur at different logits.

Direct AFTER versus BEFORE isolates the incremental numerical change more
clearly than comparing each with PyTorch:

- Maximum absolute logit change: **0.00126648**; RMSE **0.0000775630**.
- Maximum fraction of the same allclose tolerance: **19.97%**, all values pass.
- Mean negative log-likelihood change: **+2.99416e-10 nats per target**.
- Perplexity ratio AFTER/BEFORE: **1.0000000002994**.
- Per-block mean loss changes range from **−1.56343e-6 to +1.29066e-6**;
  the aggregate alone hides these small positive and negative changes.

The small decrease in error versus PyTorch does not establish improved model
accuracy: floating-point changes can cancel other rounding differences. The
earlier native elementwise test, cache parity, and exact greedy-generation
checks remain complementary evidence. Broader models, prefixes and prompts
are still needed before generalizing these results.

## Benchmark setup

Source inspection confirms that native timing uses a monotonic clock, resets
the request outside timing, prefills 63 tokens, then decodes one token using
that prefix's KV cache. Both forwards compute last-token logits. Loading,
artifact preparation, process startup and file I/O are excluded. Warmups run
the same work and are excluded from timing samples. This measures resident,
batch-one forward latency, not user-visible request latency or cold start.

Both stages use the same artifact, tokens, thread count and CPU affinity
request. The comparison removes phase profiling for child runs, measures
serial A/B/B/A passes, and retains all samples. The saved PyTorch timings
are historical context and do not qualify a fresh performance claim.

All four saved GELU records replay exactly: raw pass stability, pooled stage
statistics, between-pass drift and final verdict. The longer Linux run fails
because the first BEFORE decode p90/p10 ratio is **1.30958**, above **1.25**,
even though its prefill and AFTER stage pass. The 1.25 spread rule is a noise
screen, not a confidence interval proving a 2% improvement. ABBA reduces some
order effects but cannot remove arbitrary system load or power-state changes.
Windows and WSL here are the same physical host, not independent hardware.

Two harness checks were missing and are now enforced:

1. Unscoped comparisons must match an explicit frozen BEFORE kernel policy;
   changed or absent policy cannot silently reuse its quality provenance.
2. Each timed child must report the requested thread count, activation
   precision and exact sample count for both phases.

The historical GELU records pass both checks. These gaps therefore do not
explain their observed reductions. Regression tests deliberately inject wrong
policies, thread/precision settings and sample counts and require rejection.

## Identical-executable A/A control

One predeclared [Windows A/A control](../benchmark/results/gpt2_gelu_benchmark_aa_windows.json)
uses the original scalar-GELU executable on **both** sides, identical packed
GEMM settings, CPU 2, one thread, ten warmups and 21 samples per pass. No build,
test suite or other benchmark was launched alongside its timing. Background
host activity and power state were not controlled or independently logged.
There was one control run, with no retries or outlier filtering.

| Pass | Label | Prefill median ms | Decode median ms |
|---|---|---:|---:|
| 1 | BEFORE | 238.2496 | 34.3959 |
| 2 | AFTER (same binary) | 244.6780 | 33.9161 |
| 3 | AFTER (same binary) | 557.1127 | 51.9167 |
| 4 | BEFORE | 558.5845 | 49.1165 |

Pooled BEFORE/AFTER prefill medians are 390.9582 / 331.0984 ms: an apparent
**15.31% reduction without a code change**. Decode increases 6.65%. Pooling
samples across substantially different measurement states can produce a
misleading ratio even in ABBA order. The existing within-pass and between-pass
checks reject the run, as does the decode slowdown condition. Its verdict
also reproduces exactly from the raw samples.

This control demonstrates noise in this session; it does not quantify the
noise in prior sessions or identify its cause. Do not subtract 15.31% from
the GELU result. The setup can measure the intended workload and reject bad
runs, but this host session cannot support a reliable speedup claim.

## Next qualification steps

Keep vector GELU disabled by default. First establish a repeatable A/A control
under a recorded, quiet host state, with the same CPU and thread settings.
Predeclare the subsequent paired A/B protocol and preserve every attempt;
passing a spread screen alone is not statistical evidence for a small gain.
Then measure fresh matched PyTorch baselines and expand trained-quality checks
to longer prefixes, more held-out text and additional relevant model families.

Numerical correctness and loss arithmetic pass this audit. Whole-model speed
qualification remains unresolved. No timing threshold has been relaxed.

Validation after the harness changes: **756 tests passed, 6 skipped**, with
four existing ONNX deprecation warnings. The fresh paired model audit and
all five saved-record replays (four A/B attempts plus the A/A control) pass.

## Reproduce the numerical and saved-record audit

Use a **new output filename** for each audit; original evidence is immutable.
Run native timing separately from tests and model-quality audits.

```powershell
$env:LEAF_EXPERIMENTAL_FLOAT_TILES = '1'
python tools/audit_decoder_experiment.py --comparisons benchmark/results/gpt2_gelu_packed_windows_abba.json benchmark/results/gpt2_gelu_packed_windows_final_abba.json benchmark/results/gpt2_gelu_packed_linux_abba.json benchmark/results/gpt2_gelu_packed_linux_confirmation_abba.json --quality-comparison benchmark/results/gpt2_gelu_packed_windows_final_abba.json --frozen-record benchmark/results/gpt2_gelu_before_packed_windows.json --before build/prefill-diagnostics/leaf_decoder.exe --after build/gemm-followup/gelu-final/leaf_decoder.exe --workdir build/gpt2_prefill_followup --output build/gemm-followup/error-audit-repeat.json
Remove-Item Env:LEAF_EXPERIMENTAL_FLOAT_TILES
```
