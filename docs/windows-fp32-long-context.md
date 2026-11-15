# Four-core FP32 inference on Windows

Leaf can now split long FP32 prefill projections by token rows and divide attention
by head across its existing persistent workers. Leaf assigns complete outputs to
workers, preserving the per-output floating-point accumulation order. The shared
AVX2 attention kernel also skips whole groups of masked positions during softmax.
The path uses the converted FP32 weights directly and leaves the model's math
policy in place. Select four threads with `leaf run ... --threads 4` on a CPU with
AVX2 and FMA. The projection path starts at 768 prompt tokens. Parallel attention
also runs for single-token decode with at least 512 cached positions including
the new token.

## Decoder validation

Windows, Core i7-14700HX, four physical P-cores (`0, 2, 4, 6`), plugged in under
the existing Better Battery policy. Each comparison used the same full FP32
artifact on both sides and measured six order-balanced pairs with five samples
and two warmups per pass. The workload prefilled 768 tokens, then timed a cached
single-token forward pass. The reductions use the median of the paired ratios.
Every per-pass and pooled p90/p10 spread stayed within 1.25× for the table below.

| Model and math policy | Prefill p50, old → current | Paired reduction | Cached decode p50, old → current | Paired reduction |
| --- | ---: | ---: | ---: | ---: |
| OPT-125M, accurate FP32 | 1,302.81 → 616.82 ms | 52.80% | 26.88 → 18.07 ms | 31.32% |
| SmolLM2-360M, fast FP32 | 4,558.64 → 2,140.51 ms | 53.19% | 77.76 → 54.00 ms | 30.51% |

Both models' prefill and next-step cached logits matched the previous Leaf binary
bit-for-bit at 64 and 768 tokens. Each tested 16-token greedy continuation also
matched. In a separate held-out OPT evaluation at four threads, Leaf and PyTorch
agreed on all 1,016 next-token predictions and six greedy continuations. Maximum
absolute logit error was `7.82e-5`, and every logit passed the existing tolerance.
SmolLM2's full-FP32 checks here measure parity with the prior Leaf build.

In the direct four-core comparison against PyTorch 2.12 CPU SDPA, Leaf's pooled
medians were 637.62 vs 969.42 ms for prefill and 18.82 vs 29.42 ms for cached
decode. The median paired ratios were 0.877 and 0.605. All boundary logits passed
tolerance, with maximum absolute error `1.29e-5`. Both runtimes had passes that
exceeded the spread limit, so this comparison remains diagnostic for both phases.

The masked-softmax change alone had standalone paired prefill estimates of 6.52%
on one-core OPT and 8.08% on source-BF16 SmolLM2. Those two timing runs missed the
spread gate and remain diagnostic. The qualified percentages above apply to the
combined release with four cores on the tested FP32 workloads.
[Raw paired samples and parity records](../benchmark/validation/windows-fp32-20261115/decoder/manifest.json)

## CPUs with fewer than four physical cores

Four workers are a performance setting. `leaf run` defaults to one worker.
Use `--threads 1` on a single-core system or start with `--threads 2` on a
dual-core system. The existing decoder kernels handle those settings and retain
the requested FP32 math policy. Runtime feature checks select eligible AVX2/FMA
kernels. A CPU without those instructions uses the portable path. Individual
model weights and the KV cache still have to fit in available memory.

An Intel Core i3-6100 exposes four hardware threads across two physical cores
and supports AVX2. Its threads in each core share execution resources. An Intel
Pentium G4400 or Celeron G3900E exposes two cores and two hardware threads, and
Intel lists SSE4.2 as their highest SIMD extension. CPU tier and model are more
useful for choosing the test matrix than model year alone.
[Core i3-6100](https://www.intel.com/content/www/us/en/products/sku/90729/intel-core-i36100-processor-3m-cache-3-70-ghz/specifications.html)
· [Pentium G4400](https://www.intel.com/content/www/us/en/products/sku/88179/intel-pentium-processor-g4400-3m-cache-3-30-ghz/specifications.html)
· [Celeron G3900E](https://www.intel.com/content/www/us/en/products/sku/90713/intel-celeron-processor-g3900e-2m-cache-2-40-ghz/specifications.html)

Valve's October 2026 survey reports 2.86% with fewer than four physical cores
among its Windows respondents, summing the one-, two- and three-core rows.
This is a sample of Steam users. For a university deployment, inventory the lab
machines directly before estimating how many use these settings.
[Steam Hardware Survey](https://store.steampowered.com/hwsurvey/cpus)
