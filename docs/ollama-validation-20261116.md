# Leaf and Ollama CPU validation

This comparison uses audited FP32 checkpoint values, a CPU-only Ollama
backend, FP32 KV cache, matched prompt tokens and cache work, and greedy
generation. Leaf's source-BF16 SmolLM2 values expand to the same FP32 weights
as the Ollama checkpoint. The report measures native prefill and cached
generation with HTTP and model loading outside those phases.

The numerical preflight requires matching generated text and a finite
selected-token log probability within `2e-4` for every generated position.
It checks the reported output count before comparing values. A missing
probability can occur anywhere in a response, so the checker cannot assume
the reported positions are a contiguous prefix.

Use `--quality-only` for a fresh preflight before a long timing run. The
runner records output equality, probability coverage, per-pass and pooled
timing stability, and the speed thresholds. A comparison can meet the
evidence requirements while measured latency favors Ollama.
[Benchmark runner](../tools/benchmark_ollama_decoder.py) ·
[Ollama generate API](https://docs.ollama.com/api/generate)

## Matched Windows runs

The Leaf engine came from the current main tree. Ollama 0.40.16 used the portable
CPU package. Each scenario ran on four physical P-cores, with six order-balanced
pairs, three warmups and eleven measured requests per pass. The AC and Windows
Better Battery mode checks passed. All 198 paired continuations matched.

| Model | Input → output tokens | Prefill p50, Leaf / Ollama | Cached generation p50, Leaf / Ollama | Timing spread pass | Quality prompts passed |
| --- | ---: | ---: | ---: | --- | ---: |
| SmolLM2-360M | 64 → 2 | 179 / 197 ms | 28.1 / 47.8 ms | Neither phase | 6 / 8 |
| Qwen2.5-0.5B | 64 → 2 | 233 / 193 ms | 66.8 / 57.5 ms | Cached generation | 0 / 8 |
| SmolLM2-360M | 256 → 16 | 1,018 / 625 ms | 671 / 714 ms | Neither phase | 6 / 8 |

The cached-generation column measures one cached forward for the two-token
requests and fifteen cached forwards for the sixteen-token requests. Qwen's
cached-generation phase passed the timing spread checks on both sides and had
a median paired Leaf/Ollama ratio of 1.122. The full comparison still failed
the numerical preflight.

The longer SmolLM2 run gives a specific workload to profile next. Leaf's
prefill median exceeded Ollama's by about 393 ms on 256 input tokens, while
its cached generation median fell by about 43 ms. The Leaf timing spread
passed neither phase, so these magnitudes are diagnostic.

## Probability coverage

SmolLM2 generated an end-of-sequence token on the capital-of-France prompt.
Ollama counted one output token and returned zero selected-token probabilities.
Ollama 0.40.16's llama-server bridge forwards probabilities on nonempty,
nonterminal chunks. A missing API probability leaves that sequence unscored.
[Ollama 0.40.16 bridge source](https://github.com/ollama/ollama/blob/v0.40.16/llm/llama_server.go#L1729-L1750)

The bundled llama-server endpoint returned the same EOS token ID and a
log-probability difference of `1.76e-4` on that prompt. The other failed
SmolLM2 prompt still differed by `2.65e-4`. The direct endpoint matched
Leaf's token IDs on all eight failed Qwen prompts and reproduced its
probability differences. In the independent PyTorch FP32 check, worst
Qwen selected-token error was `1.50e-5` for Leaf and `8.51e-4` for Ollama.
The standalone TinyLlama preflight on Ollama 0.40.16 passed six of eight
prompts. One response reported seven probabilities for eight generated
tokens, and another differed by `2.18e-4`.

[Raw timings, backend token IDs and reference checks](../benchmark/validation/ollama-revalidation/20261116/manifest.json)

## 256-token scheduling experiment

The existing SmolLM2 path spent 70.8% of profiled prefill time in FP32
projections and 21.5% in attention. A candidate enables the existing
projection-panel and attention-head worker scheduling at 256 tokens for
eligible source-BF16 artifacts. The default code still starts those paths at
768 tokens. The candidate leaves FP32 storage and cached decode dispatch
at their existing thresholds.

Both SmolLM2 and Qwen2.5 matched the current Leaf logits bit-for-bit at
255, 256 and 257 prompt tokens plus one cached step. Their tested continuations
also matched. On four physical P-cores, paired prefill medians improved by
13.3% on SmolLM2 and 18.1% on Qwen2.5, with a 256-token prompt. Neither complete
timing comparison passed the existing spread gate.

In the repeated SmolLM2 workload against Ollama, the candidate's prefill
median was 664 ms versus 621 ms, with a paired Leaf/Ollama ratio of 1.053.
Cached generation was 544 ms versus 724 ms. Adding the two phases per request
gave medians of 1,210 ms for Leaf and 1,348 ms for Ollama and a paired ratio
of 0.889. These timings exclude HTTP and model load.

All 66 paired continuations matched. Both runtimes exceeded the pooled
spread limit, and the public probability preflight again passed six of
eight prompts. The experiment remains outside the production dispatch.

A separate instrumented run found a tradeoff. Mean attention time fell from
231 ms to 118 ms, while the linear projections rose from 820 ms to 954 ms.
Total profiled prefill was 1,133 ms before and 1,153 ms after. Median
process-cycle counts fell about 5% in this instrumented run. This identifies
earlier attention scheduling as a candidate to test independently from
earlier weight expansion.

The attention-only candidate leaves weight expansion at the original
threshold and starts parallel attention at 256 tokens. It matched SmolLM2
logits bit-for-bit at 255, 256 and 257 tokens and reproduced the tested
continuation. Across six Leaf-only timing pairs, pooled prefill went from
1,284 ms to 1,174 ms, with a median paired ratio of 0.822. Both prefill
and decode missed the spread gate. The attention-only candidate also remains
outside production dispatch.
[Candidate patch, profiles and raw samples](../benchmark/validation/ollama-mid-prefill/20261116/manifest.json)

## MLP GEMM across prompt lengths

Two candidates keep the existing attention policy and cached single-token
kernel. They change only fast AVX2/FMA MLP projections with source-BF16
matrices at 128–767 prompt tokens on the four-worker policy. Both expand
up to 48 output rows into worker-local FP32 scratch. **Batched** keeps
the full-K row-reuse kernel. **K-blocked** uses the existing 256-column tile
and carries its accumulators across tiles in the original FMA order. This
isolates the effect of blocking from larger conversion batches.

The native token-panel test passed. The two candidates matched over
103 million full logits against the unchanged Leaf build, spanning
127, 128 and 256 prompt tokens plus one cached step on SmolLM2 and Qwen.
The tested greedy continuations also matched. Timing used AC power,
the existing Better Battery mode, and the same four physical P-cores.
The runs below compare Leaf builds with profiling disabled.

### Six balanced rounds at 256 tokens

| Model | Existing Leaf prefill p50 | Batched p50 / paired ratio | K-blocked p50 / paired ratio |
| --- | ---: | ---: | ---: |
| SmolLM2-360M | 979.68 ms | 953.01 ms / 0.978 | 954.88 ms / 0.976 |
| Qwen2.5-0.5B | 1,092.01 ms | 1,319.34 ms / 0.947 | 1,119.86 ms / 0.990 |

Each ordering of the three binaries ran once, with two warmups and five
samples per pass. A paired ratio is the median of the candidate-to-baseline
ratios within rounds. The Qwen batched result changes direction between
that metric and pooled p50, and neither candidate passes the existing
prefill or decode gates. The reported decode ratios ranged from 0.984
to 1.127 despite unchanged single-token dispatch.

### Prompt-length sweep

Each model used prefixes of the same recorded token stream. Every length
had two opposite-order rounds with one warmup and five measured samples
per pass. The table reports paired prefill ratios relative to unchanged
Leaf. Asterisks mark comparisons where both phases met the spread check.

| Prompt tokens | SmolLM2 batched | SmolLM2 K-blocked | Qwen batched | Qwen K-blocked |
| ---: | ---: | ---: | ---: | ---: |
| 64 (inactive) | 0.813 | 1.147 | 1.196 | 1.193 |
| 128 | 0.948* | 0.998* | 0.949* | 0.950* |
| 256 | 0.986 | 0.995 | 0.957* | 0.943* |
| 384 | 1.330 | 1.061 | 0.732 | 0.624 |
| 512 | 1.186 | 0.837 | 0.868 | 1.158 |
| 768 (inactive) | 1.195 | 1.001* | 1.218 | 0.905 |

Both models showed about 5% lower prefill with batching at 128 tokens,
where the spread checks passed. All active comparisons at 384 and 512
tokens missed the spread check. Timing moved at 64 and 768 tokens as
well, where the candidate dispatch was inactive. The sweep alone cannot
attribute the longer-length differences to GEMM.

### Narrow batching confirmation

The final candidate restricts batching to 128–192 tokens and caps the
FP32 scratch at 1 MiB per worker. Both models matched full baseline logits
at 127, 128, 160, 192 and 193 tokens plus a cached step. Each also matched the
greedy continuation from a 160-token prefix.

Each cell used four balanced pairs, with two warmups and five samples per
pass. AC and Better Battery checks passed throughout. Neither phase passed
the all-pass and pooled spread checks in any cell, using the existing
p90/p10 limit of 1.25.

| Model | Prompt tokens | Prefill p50 before → after | Paired prefill ratio | Paired decode ratio |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2-360M | 128 | 426.89 → 390.83 ms | 0.969 | 1.003 |
| SmolLM2-360M | 160 | 494.05 → 453.81 ms | 0.971 | 0.995 |
| SmolLM2-360M | 192 | 689.61 → 667.81 ms | 0.977 | 0.989 |
| Qwen2.5-0.5B | 128 | 646.21 → 622.34 ms | 0.969 | 1.023 |
| Qwen2.5-0.5B | 160 | 826.87 → 791.22 ms | 0.961 | 1.011 |
| Qwen2.5-0.5B | 192 | 1,009.56 → 714.52 ms | 0.653 | 0.800 |

The 128-token case showed lower paired prefill in both campaigns and both
models, but its confirmation missed the gate. The large observed Qwen
difference at 192 tokens also included a large change in cached decode, where
dispatch was identical. All three candidates stay outside production. No
fresh Ollama comparison or cross-runtime speed claim follows from these runs.

[Candidate patches, harnesses and raw samples](../benchmark/validation/medium-gemm/20261116/manifest.json)

## November 17 projection-only scheduling and timing control

This candidate starts the existing BF16 weight-expansion and token-panel
worker path at 384 tokens instead of 768. Attention retains its existing
768-token cutoff, and cached one-token decode retains its existing kernel.
SmolLM2-360M and Qwen2.5-0.5B matched 463,708,928 full logits bit-for-bit
across 383, 384, 767 and 768 prompt tokens plus one cached token, and matched
the tested greedy continuation.

Each round launched the same baseline executable under two labels and ran
the candidate once. The three launches shared the source-BF16 artifact,
input and process affinity (logical CPUs 0, 2, 4 and 6). Each launch used
one warmup and five measured samples. The 384- and 767-token tests used
two reversed orders, and the 512-token test used four orders. AC and
Better Battery checks passed. Every sample was retained.

| Model | Prompt tokens | Candidate / first baseline | Candidate / baseline repeat | Largest same-binary prefill swing |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2-360M | 384 | 1.310 | 1.035 | 1.809× |
| SmolLM2-360M | 512 | 0.952 | 0.960 | 1.320× |
| SmolLM2-360M | 767 | 0.841 | 0.987 | 1.595× |
| Qwen2.5-0.5B | 384 | 0.958 | 1.097 | 1.163× |
| Qwen2.5-0.5B | 512 | 1.346 | 1.272 | 1.325× |
| Qwen2.5-0.5B | 767 | 0.846 | 0.914 | 1.608× |

Candidate ratios are medians of within-round prefill ratios. The same-binary
column reports the largest slower/faster ratio between the two baseline
launches in a round. For example, the first 384-token SmolLM2 round timed
the same executable at 1,695 and 3,067 ms. Same-binary cached decode reached
a 2.061× swing at 767 tokens. Neither phase met the existing spread check at
any tested length. Production dispatch stays at 768. These were Leaf-only
comparisons.

### Page-fault follow-up

A separate instrumented run launched the same SmolLM2 baseline three
times at 384 tokens. Prefill medians were 1,418.96, 1,417.10 and 1,427.33 ms.
All 15 measured prefills fell between 1,409.13 and 1,442.15 ms. Process cycle
medians were 7.909, 7.891 and 7.893 billion. Each launch recorded 11 faults
on its first measured prefill and zero faults in every later prefill.
All measured decode samples recorded zero faults. These process counters
include both soft and hard faults.

The earlier slowdown did not recur in the instrumented window. The
counter run shows a repeatable first-sample fault pattern and supplies no
fault attribution for the earlier timing swing. A previous Windows trace
observed changes in processor power-management profiles during a timing
swing under a different workload. The unchanged Better Battery mode in
this run alone cannot identify which processor policy or wait contributed.
For future comparisons, retain the two baseline launches within each
candidate round and report their drift alongside any candidate ratio.
An attribution trace must capture an actual slow sample.

[Raw samples, harness and candidate patch](../benchmark/validation/medium-projection/20261117/manifest.json)

### Warmup isolates startup faults, while latency still varies

A second SmolLM2 diagnostic used the same baseline binary, 384-token
prompt, source-BF16 artifact and four-CPU mask. It rotated zero, one and
three warmups across nine fresh launches. Each launch executed nine total
iterations. AC and the existing Better Battery mode checks passed.

| Warmups | Prefill faults observed by absolute iteration | Zero-fault measured prefills | Zero-fault prefill range |
| ---: | --- | ---: | ---: |
| 0 | Iteration 1: 193,922–193,932. Iteration 2: 5–11. Iteration 3: 0–1. | 19 of 27 | 1,403–3,502 ms |
| 1 | Iteration 2: 5–11. Later iterations: zero. | 21 of 24 | 1,548–2,234 ms |
| 3 | Zero in every measured iteration. | 18 of 18 | 1,447–3,545 ms |

The aggregate Windows fault count does not separate demand-zero and other
soft faults from hard faults. Three warmups moved the observed startup
faults outside the measured window. The same-binary prefill slowdown still
reproduced with zero faults. In one launch, a 1,945.9 ms sample used about
3,422 ms of aggregate process CPU time, while a 3,545.0 ms sample used
3,375 ms. The overlap indicated by process CPU time divided by wall time
fell from about 1.76 to 0.95 CPU-equivalents. That is consistent with lower
worker concurrency. Process-wide counters do not identify which threads
were waiting or why.

### Explicit worker placement stabilizes prefill in this run

Four balanced pairs compared the ordinary four-CPU process mask with
Leaf's existing explicit binding of the caller and workers to those same
CPUs. The binding policy also enables the existing 50 µs spin window.
Both sides used the same executable, artifact and 384-token prompt with
three warmups and five samples per launch. AC and Better Battery checks
passed. This was a separate instrumented diagnostic.

| Phase | Process-mask p50 | Explicit-binding p50 | Process-mask p90/p10 | Explicit-binding p90/p10 |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 1,679.42 ms | 1,426.86 ms | 2.485 | 1.088 |
| Cached decode | 47.58 ms | 40.44 ms | 1.940 | 1.396 |

Every measured sample had zero faults on both sides. Every explicit-binding
prefill pass met the 1.25 spread limit, and prefill ranged from 1,401 to
1,546 ms. With the process mask alone, prefill ranged from 1,413 to
3,658 ms. The bound prefill occupied about 2.17–2.64 equivalent CPUs on
average within a sample, compared with 0.94–2.49 under the mask alone.
Decode still failed the gate. The comparison tests binding and spinning
together, so it cannot attribute the improvement to either separately.
This experiment does not change production defaults or qualify a speed
claim against Ollama.
