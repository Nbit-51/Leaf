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
- Logit tolerances and observed errors.
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
it does not identify the first divergent operator. The next numerical investigation
should trace layer outputs on these saved inputs, without rerunning speed gates or
relaxing the tolerance to manufacture a pass.

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

Windowed artifacts use `LEAFDC03`/version 3 so older binaries reject them.
Unwindowed artifacts remain version 2. KV storage still grows with context and
the mask allocation remains quadratic; this is **not** bounded-memory sliding KV
or a speed claim. No trained Mistral 7B weights were downloaded or executed.

Earlier Pythia onboarding checks cover 237 targets at four lengths, one/two
threads, scalar and chunked execution, plus 16 exact greedy tokens per model.
Those limited checks are distinct from the larger corpus command above.
All retained onboarding evidence is in
[benchmark/results/model-onboarding](../benchmark/results/model-onboarding).

## Dataset and individual-network scope

Text dataset adapters and graph fusion/import/memory-planning checks pass (22
focused tests). A seeded random-weight ResNet-18 ONNX graph matches native output
within `8.6e-7` on one `1x3x32x32` input. This is operator correctness, not trained
classification accuracy or throughput. Native embeddings already process a
document collection with the model loaded once per invocation.

The next dataset milestone is a complete document-embedding job with relevance
labels, documents/second, peak memory and total time at matched output quality.
The graph milestone is a supported trained image classifier and held-out labels.
Neither requires claiming arbitrary neural-network or training support.
