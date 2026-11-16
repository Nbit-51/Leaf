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
