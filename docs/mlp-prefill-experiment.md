# MLP prefill experiment

## Predeclared candidate

Baseline: commit `c002555`, packed FP32 GEMM with row reuse and paired-row GEMV.
Windows only, one thread pinned to CPU 2. Preserve all preceding experiments.

Hypothesis: for long MLP projections, keeping six output rows but visiting
128-column segments across up to four token panels can reuse weight segments
and bound the active input working set. The current row-reuse kernel finishes
all columns for one panel before moving to the next. This differs from the
earlier blocked-K candidate, which reused inputs across up to 96 output rows
inside one panel. No hardware cache-miss cause is established.

The candidate reuses the existing accumulation primitive, retains increasing-K
FMA order and applies bias once. Scratch is bounded at 6 KiB per worker call.
No new persistent weight layout or model-specific dispatch is introduced.

First test parity, tails, strides, scalar fallback and row slices. Then compare
packing-inclusive shape timings against **row reuse**, using 12 rotating weight
replicas, serial ABBA, 31 samples and ten warmups per pass. Record all shapes;
MLP up/down are the target. Do not claim an isolated gain from unstable samples.
Only a promising candidate proceeds to opt-in integration and trained quality,
cache/generation checks and whole-model ABBA. The objective is prefill: ratio
<= 0.98, both phases <= 1.02, p90/p10 and between-pass drift <= 1.25.
Keep paired-row decode enabled in both builds. A native gain requires a separate
fresh matched framework comparison before a PyTorch superiority claim.
