# Model onboarding and output quality

Leaf separates configuration compatibility, numerical fidelity, task quality and
speed. Passing one does not qualify the others. Current model support is not
complete and there is no universal Ollama/llama.cpp speed advantage.

## Install, inspect and run

A platform wheel includes both native runtimes. Installing a wheel does not
require a compiler; building from this repository still requires GCC/Clang.
There is no requirement to configure a binary environment variable.

```powershell
python -m pip install .
leaf doctor
leaf inspect EleutherAI/pythia-70m
leaf run EleutherAI/pythia-70m --bits 32 --prompt "Once upon a time" --max-tokens 32
leaf run EleutherAI/pythia-70m --offline --bits 32 --prompt "A computer can" --max-tokens 32
```

`doctor` launches the runtime capability checks without loading/downloading a
model. Source checkouts automatically build and cache the runtimes when needed;
installed wheels select their bundled executables. `--task generation` or
`--task embedding` checks just that runtime. Generation and embedding now resolve
their runtime before fetching weights, so a missing developer override fails early.
An explicitly set override is never silently ignored. Remove an obsolete one:

```powershell
Remove-Item Env:LEAF_DECODER_BIN -ErrorAction SilentlyContinue
Remove-Item Env:LEAF_EMBED_BIN -ErrorAction SilentlyContinue
leaf doctor
```

`inspect` currently checks decoder configuration only. It downloads metadata,
reports a lower bound on required FP32 weight memory, and never asserts trained
weight correctness or performance. Weight downloads use the inspected snapshot
revision. Pythia-70M and Pythia-160M are base language models, not instruction
chatbots; repeating or unhelpful answers can also occur in the reference model.

## Quality-only evaluation

Install validation dependencies once in the source checkout:

```powershell
python -m pip install ".[validation]"
leaf evaluate EleutherAI/pythia-70m --offline --dataset benchmark/data/wikitext2-test.parquet --dataset-source "WikiText-2 test split" --blocks 32 --sequence-length 128 --output build/quality-pythia70
```

Use an existing held-out TXT, JSON/JSONL, CSV or Parquet text dataset, with
`--text-column` when needed. The example dataset is not distributed in the wheel.
Use a fresh output directory for each run. The default evaluates a deterministic
4096-token corpus prefix as 32 independent 128-token blocks, scoring 4064 targets.
Context resets at each block and the last logit has no target. Consequently these
perplexities are not directly comparable to full-context or sliding-window
perplexities published elsewhere. The corpus may have been present in model
pretraining; this test does not claim training-set decontamination.

`quality.json` records:

- Native and reference perplexity, pooled through token-weighted log losses.
- Next-token accuracy against the dataset for each runtime, and their agreement.
- Logit tolerances, violation counts and the worst offending coordinate.
- Probability error, total variation and reference-to-Leaf KL divergence on
  scored rows, plus reference top-two margins at any argmax disagreements.
  These are explanatory diagnostics; they do not replace the acceptance gates.
- Six fixed raw-prompt greedy continuations, their actual text/tokens and first
  divergence. These include prose, a factual question, arithmetic, code and Unicode;
  matching an answer does **not** establish that the answer is correct.
- Dataset, source weights/configuration, native artifact/runtime and token hashes,
  versions and evaluation policy. Source weights are promoted to FP32 in both
  paths, not reconstructed to precision absent from the original checkpoint.

The predeclared FP32 fidelity gates are at least 99.9% next-token agreement,
perplexity ratio at most 1.001, logits within `atol=2e-4, rtol=2e-4`, and exact
greedy continuation tokens. Generation uses explicit greedy defaults and a
full-prefix PyTorch oracle; native generation uses its KV cache. Failures are
retained and return a nonzero exit code. Each block/prompt writes an atomic
checkpoint; incomplete reports cannot pass. Reference logits occupy one block
at a time, although native and reference model weights coexist in memory.
This path is intended for models that fit available RAM.

This command does not run speed gates or enable automatic quantization.
Instruction-following scores, retrieval relevance and broad model qualification
remain separate work, using task datasets appropriate to the selected model.

### Measured output checks — October 13, 2026

Both cached trained Pythia checkpoints were evaluated through the newly installed
Windows wheel on the same WikiText-2 test prefix: 4064 scored targets each and six
16-token greedy prompts. These are first measurements under this larger protocol,
not selected reruns. PyTorch 2.12.0+cpu / Transformers 5.14.1, one thread, FP32.

| Model | Leaf / PyTorch perplexity | Next-token agreement | Dataset next-token accuracy (both) | Exact continuations | Overall strict gate |
|---|---:|---:|---:|---:|---|
| Pythia-70M | 100.901026 / 100.900828 | 99.9754% | 24.1880% | 6/6 | Failed: block 30 logit tolerance |
| Pythia-160M | 63.255483 / 63.255465 | 100% | 28.3465% | 6/6 | Failed: block 32 logit tolerance |

Both pass the aggregate agreement/perplexity requirements. Neither passes the
entire strict suite, because one block per model fails elementwise logit tolerance.
The actual reports retain `complete: true, passed: false`. The low dataset accuracy
measures exact next-token prediction, not QA-answer accuracy; prose has many possible
continuations. Do not describe these results as identical logits or complete quality
qualification.

Focused Pythia-70M diagnosis found 101 out-of-tolerance values among 6,438,912
logits in block 30. The worst tolerance ratio is 6.72 (reference 0.5554304,
Leaf 0.5575199); scalar execution also fails. The old and new runtimes produce
bit-identical outputs on that block and on block 11, which contains the single
top-token disagreement. There the reference top-two margin is 0.00012207.
This establishes a pre-existing numerical discrepancy on the checked blocks;
it did not identify the first divergent operator. The October 14 trace now
localizes the first material amplification to layer index 3 attention. PyTorch
SDPA also fails the existing elementwise tolerance against eager on the same
70M input. A direct-scale candidate helps some checks but still fails 70M;
neither the default arithmetic nor the gate has changed. See the
[numerical findings and next steps](pythia-numerics-handoff.md) before repeating
tests. These are correctness diagnostics, not new performance results.

Evidence:
[70M quality](../benchmark/results/model-onboarding/pythia70-quality-20261013.json),
[160M quality](../benchmark/results/model-onboarding/pythia160-quality-20261013.json),
[70M diagnosis](../benchmark/results/model-onboarding/pythia70-logit-diagnosis-20261013.json).

The installed wheel also passes runtime startup for both bundled binaries,
fresh-artifact preparation and cached generation against the saved trained token
reference, plus actual cached Hub-ID execution. All inference checks remove binary
overrides and compiler paths and assert that Torch/Transformers are not imported.
The quality command intentionally imports those reference libraries.
[Package smoke](../benchmark/results/model-onboarding/package-smoke-20261013.json),
[runtime audit](../benchmark/results/model-onboarding/doctor-audit-20261013.json),
[Hub-ID audit](../benchmark/results/model-onboarding/hub-run-audit-20261013.json).
The wheel was built and installed locally; this does not claim a published PyPI
release or a fresh online download of already cached weights.

On October 14, the same wheel was installed into a fresh virtual environment
and its real `leaf.exe` launcher was run outside the repository root with only
Windows System32 on PATH. `doctor` and cached Hub-ID generation both passed.
The final installed decoder also passed the complete reduced sliding-window
suite again, including the 4096-token boundary and rejection by the old reader.
[Final installed-window validation](../benchmark/results/model-onboarding/installed-window-20261014.json).

## Mistral operators without 7B weights

The [Mistral-7B-v0.1 configuration](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json)
declares a 4096-token causal window. Leaf now imports a positive uniform window
and restricts query `q` to keys `max(0, q + 1 - window)` through `q`, preserving
absolute rotary positions. Mixed per-layer window policies remain unsupported.
The implementation uses shared decoder semantics, not a model-name-specific kernel.

Reduced models test windows 1, 8 and 4096, including length 4097, prefill,
two threads, scalar execution, chunked cache and generation. Maximum observed
absolute logit error is `9.5367431640625e-7`. A full-attention negative control
confirms the mask changes outputs. Existing ten reduced decoder configurations
and precisions remain bit-identical to the prior binary.

Transformers 5.14.1 cached generation disagreed with its own full-prefix result
at window 1. The initial failure is preserved alongside the successful
full-prefix-oracle validation; windows 8 and 4096 also require cached-reference
agreement. This discrepancy is not hidden by a tolerance change.

Windowed single-EOS artifacts use `LEAFDC03`/version 3 so older binaries reject them.
Unwindowed single-EOS artifacts remain version 2. Multi-EOS uses version 4,
including when combined with a window. KV storage still grows with context and
the mask allocation remains quadratic; this is **not** bounded-memory sliding KV
or a speed claim. No trained Mistral 7B weights were downloaded or executed.

Earlier Pythia onboarding checks cover 237 targets at four lengths, one/two
threads, scalar and chunked execution, plus 16 exact greedy tokens per model.
Those limited checks are distinct from the larger corpus command above.
All retained onboarding evidence is in
[benchmark/results/model-onboarding](../benchmark/results/model-onboarding).

## SmolLM2 and Qwen2.5 — October 17, 2026

Both models were downloaded through Leaf's normal Hub-ID resolver at pinned
revisions: SmolLM2-360M-Instruct `a10cc1512eabd3dde888204e902eca88bddb4951`
and Qwen2.5-0.5B-Instruct `7ae557604adf67be50417f59c2c2f167def9a775`.
Their metadata was preserved. They use the shared decoder and kernels.

Qwen exposed a generation-policy gap: its configuration declares one primary
EOS, while `generation_config.json` declares `[151645, 151643]`. Built-in plans
now resolve that complete policy consistently for import, export, calibration
and reference checks. Custom plans must match the declared policy. Multi-EOS
artifacts use `LEAFDC04`; existing single-EOS/no-stop artifacts keep versions
2/3. Older runtimes reject version 4 rather than discard stop tokens.
Native tests force the secondary EOS and verify no additional decode forward.

The reference uses literal full-prefix forward/argmax steps with the same EOS
list. It cannot inherit Hub repetition penalties, sampling, forced tokens or
suppression rules; Leaf explicitly implements greedy generation. The cache
namespace and validation identity include the changed generation policy.

Both FP32 evaluations use eight independent 128-token WikiText-2 test-prefix
blocks, **1,016 scored targets**, and six fixed raw prompts with up to sixteen
new tokens. These are bounded checks, not full-corpus or task benchmarks.

| Model | Next-token agreement | Exact continuations | Perplexity ratio to PyTorch | Strict logit gate |
|---|---:|---:|---:|---|
| SmolLM2-360M-Instruct | 100% | 6/6 | 0.99999946 | Fail: 154 tolerance violations |
| Qwen2.5-0.5B-Instruct | 100% | 6/6 | 1.00000058 | Fail: 1,000 tolerance violations |

The `atol=2e-4, rtol=2e-4` limits are unchanged, and both overall fidelity
verdicts remain **false**. Maximum probability errors on scored rows are
`1.50e-5` and `1.04e-5` respectively. Matching tokens and near-identical
perplexity do not erase the all-logit failures. No model-speed qualification
or automatic quantization profile is granted by these checks.

The final unit suite passes **1,029 tests**, including real native v2/v3/v4
reader and termination cases. The current-source Windows wheel passes offline
GPT-2 first preparation/cache reuse and actual cached Qwen Hub-ID execution,
with no compiler on PATH or Torch/Transformers loaded. Qwen reuses its version-4
artifact and generates `there was` from `Once upon a time,` in the two-token
installed smoke check. This is install/functional evidence, not latency evidence.

`inspect` also deduplicates identical source/transform/shape mappings when
estimating required FP32 payloads. Its bounds exclude optional biases, KV,
activations, source weights and operating-system memory.

Evidence: [Smol quality](../benchmark/results/family-onboarding-20261017/smol-quality-8.json),
[Qwen quality](../benchmark/results/family-onboarding-20261017/qwen-quality-8.json),
[generation/format checks](../benchmark/results/family-onboarding-20261017/eos-validation.json),
[full unit summary](../benchmark/results/family-onboarding-20261017/unit-validation.json),
[installed GPT-2](../benchmark/results/family-onboarding-20261017/package-v4.json),
[installed Qwen](../benchmark/results/family-onboarding-20261017/package-qwen-v4.json).

Next, diagnose the remaining shared numerical differences before extending
quality claims, then run audited matched competitor comparisons for these two
models. The TinyLlama/Ollama timing result does not transfer to them.

## Qwen cached CLI comparison — November 9, 2026

Leaf and llama.cpp b11540 ran Qwen2.5-0.5B-Instruct on one Windows CPU thread in
Better Battery mode. Both used artifacts audited against the same checkpoint
values, FP32 weights and KV cache, and matching raw prompt token IDs. Each
command started fresh processes with a warm OS file cache. Six interleaved
pairs per prompt length retained the first visible text and total command
times. Both runtimes produced matching text for the eight-token continuations.

| Prompt tokens | First visible text, Leaf / llama.cpp | Total command, Leaf / llama.cpp |
|---:|---:|---:|
| 63 | 1,907.99 / 2,214.60 ms | 3,007.09 / 3,318.30 ms |
| 127 | 2,505.48 / 2,842.25 ms | 3,628.55 / 3,948.04 ms |

Both external metrics passed the 1.25× p90/p10 spread check. Leaf's total
command time was 8.39% and 8.64% lower by median paired ratio. These timings
cover two prompts, eight generated tokens and this exact CLI configuration.
The comparison checked output text and generation length. Full-logit parity
between the two runtimes was outside this run.
[Retained samples, settings and hashes](../benchmark/validation/bf16-expansion/20261109/manifest.json)

### Reuse the comparison on another checkpoint

Run `python -m tools.benchmark_llama_cli --help` for the Windows runner. Supply
the same local checkpoint, its held-out token fixture, an audited FP32 Leaf
artifact and its provenance, and an audited FP32 GGUF with `llama-completion`
and `llama-tokenize` from the same llama.cpp build. Choose the prompt lengths,
generation budget, logical CPU and effective power mode explicitly. The
checkpoint currently needs one safetensors shard for the GGUF audit binding.

The runner checks checkpoint and artifact hashes, exact prompt IDs, reported
work counts and continuation text. It retains every paired timing, applies
the spread check separately to first visible text and total command time, and
marks mismatched or unstable workloads unqualified. Each command starts a fresh
process against staged cached files. Full-logit parity and peak memory are
separate measurements.

To require a passing Leaf reference check before collecting timings, pass its
`--quality-report` and `tokens.json` from `leaf evaluate`. Use one thread and the
same `--fp32-math` for both commands. The comparator verifies the original strict
fidelity thresholds and binds the report to the exact checkpoint, Leaf artifact,
runtime and token fixture. The GGUF audit checks the stored source parameters;
matching llama.cpp continuation text remains a separate check.

The reusable runner passed its workload and timing checks on Qwen2.5-0.5B
at 63 and 127 prompt tokens and SmolLM2-360M at 127. These checks used the
existing FP32 checkpoints and eight generated tokens.
[Reusable comparison evidence](../benchmark/validation/decoder-operators/20261109/manifest.json)

### Decoder operator coverage

Decoder format v7 records static linear or Llama 3 inverse rotary frequencies,
per-head Q/K RMSNorm for dense Qwen3, or both. Qwen3 may project into an attention
width different from the residual width. This format supports FP32 and original
BF16 storage. Existing artifacts remain readable and unchanged checkpoints keep
their original format. Use a current native runtime to load v7.

The reduced-model tests cover a Llama 3 context boundary, unequal Qwen3 attention
and residual widths, full and chunked logits, greedy outputs and exact BF16
storage parity. Sequence-length-dependent RoPE and mixed per-layer windows
still require separate operator work.

`leaf inspect` accepts the published Qwen3-0.6B configuration. A subsequent
trained run at revision `c1899de289a04d12100db370d81485cdf75e47ca` passed the
strict FP32 reference gate on eight held-out 128-token blocks. It matched all
1,016 next-token predictions and all six greedy continuations, with zero logit
tolerance violations. The source-BF16 artifact was 50.01% of the FP32 artifact
size. It preserved every stored parameter value and all 155,582,464 tested
logits bit-for-bit, and produced the same six continuations as Leaf FP32.

### Accuracy-bound comparisons on additional small checkpoints

Each run used one Windows CPU thread, llama.cpp b11540, FP32 artifacts,
127 prompt tokens and eight generated tokens. Both runtimes produced matching
continuation text on all four checkpoints. The percentages use median paired total
command latency across six pairs per model.

| Model | Leaf / llama.cpp total p50 | Paired result | Strict Leaf FP32 gate |
|---|---:|---:|---|
| Qwen3-0.6B | 4,400.97 / 4,876.30 ms | 10.03% lower, stable | Passed, bound to run |
| SmolLM2-360M | 2,807.61 / 2,867.62 ms | 2.02% lower, stable | Passed, bound to run |
| GPT-2 124M | 1,233.35 / 1,161.63 ms | Diagnostic, first-visible spread failed | Failed, 7 logits outside tolerance |
| Pythia-70M | 734.61 / 538.61 ms | 35.76% higher, stable diagnostic | Failed, 131 logits outside tolerance |

The two diagnostic models still matched all six greedy reference continuations.
[Retained quality, timings and audited inputs](../benchmark/validation/trained-decoder-and-retrieval/20261109/manifest.json)

### Match Ollama's CPU workload

`python -m tools.benchmark_ollama_decoder --help` accepts prompts from 16
to 1,024 tokens and can run across selected physical cores. Use
`--prompt-tokens 256 --worker-cpus 0 2 4 6` on a Windows system where those
CPU indices identify four distinct physical cores. The default remains a
64-token prompt on CPU 2.

The runner requires an audited FP32 GGUF and a Leaf artifact from the same
checkpoint. It checks the Ollama backend's token IDs, thread count, context,
batch sizes, CPU placement and FP32 KV cache. It verifies cached prompt work
and the number of generated tokens for each measured request. Quality and
timing gates apply independently to the comparison.

The four-core SmolLM2-360M comparison used 256 uncached prompt tokens and one
cached decode forward. Leaf and Ollama generated the same text in every
measured pair. Leaf's observed median paired latency was 1.46× Ollama's for
prefill and 1.22× for decode. Ollama's timing spread exceeded the limit in
both phases, and two of eight quality checks failed. PyTorch reproduced
Leaf's greedy tokens on those two prompts with selected log-probability
errors of at most `6.58e-6`. Ollama omitted one probability and differed
from PyTorch by up to `2.62e-4` on the other prompt.
[SmolLM2 comparison and reference checks](../benchmark/validation/ollama-and-packing/20261111/manifest.json)

## Dataset and individual-network scope

Text dataset adapters and graph fusion/import/memory-planning checks pass (22
focused tests). A seeded random-weight ResNet-18 ONNX graph matches native output
within `8.6e-7` on one `1x3x32x32` input. This is operator correctness, not trained
classification accuracy or throughput. Native embeddings already process a
document collection with the model loaded once per invocation.

`python -m tools.evaluate_retrieval --help` scores aligned JSONL rows and
their `.npy` embeddings against BEIR-format judgments. It records per-query
cosine rankings, nDCG and Recall. A SciFact pilot with 32 documents and four
queries reproduced the FP32 reference's top-10 rankings and stayed within
`3.28e-7` maximum absolute vector error. The pilot includes the selected
queries' known relevant documents, so it is a workflow and parity check.
[SciFact pilot evidence](../benchmark/validation/trained-decoder-and-retrieval/20261109/manifest.json)

For longer jobs, run `python -m tools.embed_dataset MODEL --offline --dataset documents.jsonl --directory build/document-index`.
Use `--task raw` if the JSONL text already contains the document title and
retrieval prefix. A query collection can use `--task query` with its own directory.
The command writes an ordered `embeddings.npy` and `result.json` when the job
completes. Pass `--max-new-chunks 1` to stop after one newly completed batch,
then rerun with the same settings to continue. The job verifies the dataset,
model, native runtime, and saved vector hashes before reusing checkpoints. A
crash during a batch leaves the previous batches available for the next run.

A paused and resumed SciFact pilot produced all 32 vectors bit-for-bit equal
to the earlier single-command output. The full SciFact inputs contain 5,183
documents and 300 judged queries. The longest formatted inputs are 1,901 and
59 tokens respectively, within EmbeddingGemma's 2,048-token limit.

### Reproduce the SciFact test split

Save the corpus and query Parquet files and the test judgments from the
pinned BEIR revisions. [Corpus and queries](https://huggingface.co/datasets/BeIR/scifact/tree/b3b5335604bf5ee3c4447671af975ea25143d4f5)
[Test judgments](https://huggingface.co/datasets/BeIR/scifact-qrels/blob/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029/test.tsv)

From the repository root, with PyArrow available, pass those files to the
preparation command. It verifies their hashes, preserves corpus order, and
keeps the 300 queries with test judgments. The offline commands assume a cached
EmbeddingGemma snapshot. A local snapshot path also works in place of the model ID.

```text
python -m benchmark.prepare_scifact --corpus corpus-00000-of-00001.parquet --queries queries-00000-of-00001.parquet --qrels test.tsv --output build/scifact
python -m tools.embed_dataset google/embeddinggemma-300m --offline --dataset build/scifact/documents.jsonl --task raw --threads 4 --chunk-size 64 --directory build/scifact/documents
python -m tools.embed_dataset google/embeddinggemma-300m --offline --dataset build/scifact/queries.jsonl --task query --threads 4 --chunk-size 128 --directory build/scifact/queries
python -m tools.evaluate_retrieval --query-rows build/scifact/queries.jsonl --document-rows build/scifact/documents.jsonl --query-vectors build/scifact/queries/embeddings.npy --document-vectors build/scifact/documents/embeddings.npy --qrels build/scifact/test.tsv --k 10 --output build/scifact/retrieval.json
```

### Full SciFact test result

Leaf embedded all 5,183 documents and 300 judged queries with EmbeddingGemma
300M FP32 on Windows using four CPU threads. Resuming after the interruption
reused 63 saved batches and computed the remaining 18. All 32 pilot document
vectors and four pilot query vectors were bit-identical to the full run. A
replay of the completed job reused all 81 batches.

| nDCG@10 | Recall@10 | Queries with a relevant result in the top 10 |
|---:|---:|---:|
| 0.79209 | 0.92189 | 278 / 300 |

An independent float64 cosine calculation reproduced all 300 top-10 rankings
and both scores. The PyTorch FP32 reference used identical token IDs for all
300 queries and 31 documents sampled across input lengths. The largest absolute
vector difference was `4.99e-7`. Substituting the reference query vectors while
keeping the Leaf document index produced the same 300 top-10 rankings.
[Reference vector checks](../benchmark/validation/scifact/20261110/reference-parity.json)

Document batch wall times totaled 51.62 minutes across the
resumed sessions, or 1.67 documents per second. This includes per-batch model
preparation and loading and excludes pauses and final assembly. A matched
runtime comparison and peak-memory measurement are still needed.
[Full retrieval results and batch measurements](../benchmark/validation/scifact/20261110/manifest.json)

The graph milestone is a supported trained image classifier and held-out labels.

### Parallel attention heads

With multiple workers, the encoder distributes attention heads across those
workers for inputs of at least 128 tokens. Each task writes a disjoint output
slice and keeps the original order of dot products, softmax and value accumulation.

On Windows with four physical P-cores, six paired offline `leaf embed` commands
over 16 SciFact documents reduced median command time from 16.98 to 13.80 seconds.
The median paired reduction was 15.8%, with before/after spread of 1.06×/1.14×
against the 1.25× limit. Every output coordinate matched the earlier FP32 vectors.
The native CLI also reproduced all 300 query vectors and 31 length-stratified
document vectors bit-for-bit.

The separate warm comparison against FP32 llama.cpp over six documents from
71 to 1,014 tokens still favored llama.cpp by a paired ratio of 1.173.
Short embedding queries of 14–59 tokens favored Leaf in the observed samples,
but Leaf's 1.47× spread exceeded the timing limit.

A four-thread operation profile of the 1,014-token input measured about 57%
of Leaf time in projection and MLP GEMMs, 28% in attention, and 5% in input
packing. Attributing the entire llama.cpp gap still requires matched
measurements of both runtimes.

The decoder prototype produced bit-identical logits at one, two and four
threads on trained Qwen3-0.6B and SmolLM2-360M prompts. At 256 prefill tokens
and four P-cores, Smol passed the raw-sample timing gate at above-normal process
priority with a 24.6% paired prefill reduction. Qwen's independent repeat
showed a 22.5% paired prefill reduction and a 1.42× decode spread in one pass.
Decoder adoption remains pending.
[Raw samples, quality checks and operation profile](../benchmark/validation/parallel-attention/20261111/manifest.json)

### Shared input packing experiment

The candidate reused packed inputs across the Q/K/V and MLP gate/up
projections and enabled the existing SIMD transpose. It reproduced all 331
sampled SciFact vectors bit-for-bit. The isolated packing probe reduced
paired latency by 73–80% and passed its spread limit at 129, 320 and 1,014
tokens. End-to-end tests showed a 1.1% paired reduction on the 16-document
batch and a 3.1% increase on the six-document batch of 711–1,334-token
inputs. Both runs exceeded the spread limit, so the packing candidate
remains experimental.
[Packing samples and candidate patch](../benchmark/validation/ollama-and-packing/20261111/manifest.json)

### Decoder costs by prompt length

The instrumented four-core Windows FP32 baseline spent the following shares
of prefill time in linear projections and attention.

| Model | Tokens | Linear projections | Attention |
|---|---:|---:|---:|
| SmolLM2-360M | 64 | 92.2% | 3.4% |
| SmolLM2-360M | 768 | 39.1% | 55.1% |
| Qwen2.5-0.5B | 64 | 88.3% | 4.1% |
| Qwen2.5-0.5B | 768 | 49.1% | 44.4% |

The native profiler includes warmup and first use. Input packing was around
1% of prefill. At a 768-token context, projections accounted for 63% and 74%
of decode on SmolLM2 and Qwen respectively, including 34% and 41% in MLP
projections. Cached attention accounted for 36% and 25%. Use `python -m tools.profile_decoder --help` with
`--threads 4 --worker-cpus 0 2 4 6` to collect the same operation breakdown
where those CPU indices are available.

The token-panel scheduling prototype produced bit-identical logits on both
trained models. It reduced paired 768-token prefill latency by 9.0% on SmolLM2
and 8.3% on the Qwen repeat, with stable prefill samples in both. One Qwen
baseline decode pass reached 1.261× spread against the 1.25× limit, so decoder
adoption remains pending.

In a separate five-sample Qwen decode counter replay per build, operation
and token counts matched. Median wall time was 82.71 ms for the baseline
and 86.22 ms for the candidate. Median process CPU time was 250 ms for both,
median cycles fell from 550.5 million to 542.0 million, and every sample
reported zero page faults. The earlier 112 ms sample did not recur. Hardware
cache misses were not measured.

A second prototype combined token-panel scheduling with parallel attention
heads. On the matched FP32 SmolLM2-360M workload against Ollama 0.40.4,
paired prefill was 5.3% faster (2,251 vs 2,377 ms median) and paired decode
was 12.5% slower (56.36 vs 49.99 ms median). The screen used two balanced
pairs with five samples per pass, a 768-token prompt and two generated tokens.
Both phases met the spread limit. All timed generated text matched, and
the combined build matched 37,797,888 prior Leaf logits bit-for-bit.
The Ollama quality gate passed six of eight checks. This screen supports
investigating the remaining decode gap before drawing an overall speed
conclusion.

[Profiles, kernel shapes and raw timing samples](../benchmark/validation/panel-scheduling/20261112/manifest.json)
