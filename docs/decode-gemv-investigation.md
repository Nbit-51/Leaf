# Decode GEMV investigation — path trace, 2026-10-05

This applies the supplied investigation guidance to the current implementation.
It completes the source-path trace and declares the next experiment; it contains
no new kernel implementation, latency measurements or speedup claim. The
[Windows-only development policy](linear-row-reuse.md#platform-acceptance-policy)
remains in force. Further WSL/Linux evaluation is deferred.

## Actual execution path

`main_decoder.cpp` benchmark mode resets the decoder, times a 63-token prefix,
then separately times one cached token. Loading and warmup samples are outside
the reported latency samples. `Decoder::forward` enters `Impl::forward`, which
calls `linear` for Q/K/V, attention output, MLP up/down and finally `lm_head`.

The current [Windows reference](../benchmark/results/gpt2_row_reuse_matched_windows.json)
records FP32 weights/activations, AVX2 enabled, one thread, packed prefill enabled,
and `LEAF_EXPERIMENTAL_FLOAT_GEMV` disabled. Therefore:

1. Constructor dispatch selects `dot_avx`, not `dot_specialized_avx`.
2. One-token calls fail the packed GEMM condition `tokens >= 8`.
3. FP32 skips activation quantization and integer kernels.
4. `Workers::run` executes the row range directly for one thread; there are no
   worker wakeups or barriers in this configuration.
5. The row loop invokes the selected dot function once per output row. Its
   `tokens == 1` branch already uses four independent AVX2/FMA accumulators,
   consuming 32 input/weight elements per iteration, followed by a fixed
   reduction and SIMD/scalar tails. Bias is added in `linear` afterwards.

The opt-in `dot_one_avx<32>` already specializes the weight datatype and retains
the four-accumulator arrangement. It must be included as an existing comparator,
not rediscovered as a new optimization. Its historical
[GEMV experiment](../benchmark/results/gpt2_float_gemv_abba.json) was unstable;
that result neither qualifies it nor proves it is slower.

Weights are row-major `[output_channels, input_channels]` in mapped artifact
storage. Inputs/outputs are token-major. Normal FP32 decode does not pack or
convert those weights or its input. Optional calibrated input scaling is a
separate branch. Intermediate vectors resize with reusable capacity; final
logits use a local vector each forward. Allocation/dispatch costs have not yet
been isolated and must not be declared dominant from source inspection alone.

Weight-only INT8/INT4 with FP32 activations use dot/dequantization paths with
per-group scales. With 8-bit activations, decode quantizes input and uses the
integer dot path; the multi-token VNNI tile is not selected for `tokens == 1`.
The first new candidate should be FP32-only, leaving these behaviors intact.

## Shapes that must be measured

The cached GPT-2 config has hidden width 768, 12 layers and vocabulary 50,257.
Its default MLP expansion is 3,072. For one-token decode:

| Projection | Weight N×K | Calls per forward | Share of logical FP32 linear weight elements |
|---|---:|---:|---:|
| Q/K/V/attention output | 768×768 | 48 | 22.92% |
| MLP up | 3072×768 | 12 | 22.92% |
| MLP down | 768×3072 | 12 | 22.92% |
| Vocabulary head | 50257×768 | 1 | 31.24% |

These are dimension-derived element counts, not measured time or DRAM traffic.
The large vocabulary head must not be omitted from a GEMV benchmark. With
last-token-only logits, it also receives one token during **prefill**. A GEMV
change can therefore affect both reported phases even though transformer-block
prefill still uses packed GEMM. Full-logit quality evaluation uses a different
head shape; cached/chunked and generation checks remain necessary.

## Nuances to the proposed optimization

- The current path already has one-token specialization and independent sums.
  The new hypothesis is sharing input loads across output rows while preserving
  each row's existing accumulation/reduction order.
- Input vectors are only 3 KiB or 12 KiB for these shapes. Repeated input loads
  are not automatically repeated DRAM reads. Multiple rows may reduce load
  instructions or dispatch overhead, but may also increase register pressure
  or complicate weight streaming. Assembly and measured throughput must decide.
- Preserve four chains per row initially. Changing reduction order and row
  grouping together would confound numerical and performance analysis.
- Include resident single-matrix and rotating-weight conditions. Label any
  averaging explicitly: twelve-call sample means are not individual decode
  latencies. Effective weight-byte throughput is not a hardware bandwidth
  counter and cannot by itself establish a memory-bandwidth bottleneck.
- Inspect emitted assembly before claiming that runtime datatype branches,
  unused reductions or function calls actually survive compiler optimization.
- Keep MLP prefill tuning as a separate candidate. Combining it with decode
  row grouping would obscure which change helped or regressed each phase.

## Predeclared evaluation sequence

1. Add an isolated Windows harness exercising the actual existing GEMV code,
   not a loosely equivalent reimplementation. Measure all four shapes above
   and obtain a fresh model baseline using the frozen current row-reuse build.
2. Compare the established `dot_avx` and existing datatype-specialized path.
   Inspect assembly and record dimensions, ISA/compiler, precision, threads,
   weight working set, warmups, raw samples, p10/p90, medians and throughput.
3. If justified, implement one opt-in multi-row FP32 candidate with unchanged
   per-row arithmetic, bounded storage, scalar fallback and no weight duplication.
4. Test row/column tails, unaligned inputs, bias, row slices, guards and numerical
   parity; then validate architectures, trained logits/loss, cache and generation.
5. Run one predeclared serial Windows whole-model ABBA with 31 samples and ten
   warmups per pass, one thread on CPU 2, no profiler or concurrent build/test.
6. Retain every attempt. Classify the result as ACCEPT, REJECT or INCONCLUSIVE;
   unstable timings remain inconclusive and do not authorize promotion.
7. After a passing native comparison, obtain fresh matched Windows PyTorch
   timings and report its gap separately from the Leaf before/after gain.

### Required gate adjustment before candidate timing

The existing `native_experiment_gate` is explicitly prefill-oriented: it requires
`prefill_after / prefill_before <= 0.98`. Reusing it unchanged for a decode target
would incorrectly reject a stable decode improvement with unchanged prefill.

Add an explicit, recorded **decode objective** before timing the candidate,
while preserving the existing default and historical record replay. Its required
improvement is `decode_after / decode_before <= 0.98`; both phases must remain
within the existing 1.02 no-slowdown allowance. Preserve all quality checks,
p90/p10 <= 1.25, between-pass drift <= 1.25, sample requirements and raw data.
Test both objective modes and historical replay. This changes the declared
optimization target, not numerical thresholds or the objective after seeing data.

The implementation and measurements are the next stage. This trace alone does
not establish which hardware resource limits GEMV or justify a speed claim.
