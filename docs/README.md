# Leaf documentation

The [project README](../README.md) covers installation, implemented capabilities,
architecture, current results and scope. Use this index for details and evidence.

## Using and developing Leaf

| Topic | Reference |
|---|---|
| Model onboarding, runtime setup and quality-only evaluation | [Model and output checks](model-onboarding.md) |
| Original BF16 matrix/output-head storage with FP32 computation | [Source-preserving storage](source-bf16.md) |
| Explicit accurate FP32 execution and its trained-output checks | [Accumulation policy](shared-accumulation.md#native-accuracy-option--october-26-2026) |
| Shared FP32 input packing, safeguards and measured scope | [Input packing](shared-input-packing.md) |
| Optional per-thread CPU placement, cleanup and validation | [Worker placement](worker-placement.md) |
| Optional Windows CPU-work counters and paired math diagnostics | [Benchmark accounting](benchmark-counters.md) |
| Installation, local/Hub models, quantization and cache controls | [Usage](project-reference.md#2-install-and-run-with-one-command) |
| Native document embeddings, JSONL inference and numerical validation | [Embeddings](embedding.md) |
| Supported decoder configurations and graph operators | [Compatibility](project-reference.md#3-coverage-and-explicit-boundaries) |
| Architecture and implementation pipeline | [Architecture](project-reference.md#4-architecture) |
| Building, tests and trained-model reproduction | [Validation guide](project-reference.md#6-reproduce-validation-and-benchmarks) |
| Native formats and artifact trust boundary | [Artifacts](project-reference.md#10-artifact-formats-and-trust-boundary) |
| Buffer liveness and memory reuse | [Memory plan](project-reference.md#11-memory-plan-format-and-runtime-reuse) |
| Caller-owned C++ logits buffers and generation reuse | [Output storage](output-buffers.md) |
| Repository layout | [Source map](project-reference.md#12-repository-map) |
| Benchmark runners, datasets and retained evidence | [Benchmark directory](../benchmark/README.md) |

## Current Windows release evidence

- [Final-layer FP32 row selection](final-layer-row-selection.md): byte-identical
  tested outputs and stable Pythia/GPT-2 prefill gains over the previous Leaf build.
- [FP32 default kernel stack](windows-fp32-release.md): default dispatch,
  correctness, installed wheel and conservative-build comparison.
- [Attention/softmax follow-up](windows-attention-softmax.md): accepted GPT-2
  prefill improvement, fresh PyTorch comparison and retained failed attempts.
- [FP32/W8A8 vector SiLU releases](silu-gate-experiment.md): stable TinyLlama
  incremental improvements, default dispatch and correctness evidence; W8A8
  also has a matched PyTorch comparison.
- [Current TinyLlama evaluation](windows-tinyllama-current.md): FP32/W8A8 quality,
  all timing results, stability failures and the next measured hot paths.
- [TinyLlama / Ollama FP32](tinyllama-ollama-comparison.md): stable new-default
  prefill/decode advantages, the earlier baseline and the retained probability miss.
- [Raw benchmark records](../benchmark/results): full samples, gates and hashes.

These reports have different workloads and runtime identities. Their results
should not be combined into a universal speedup claim.

## Investigation history

| Investigation | Report |
|---|---|
| Vector SiLU/gating implementation, retained failures and staged reproduction | [SiLU report](silu-gate-experiment.md) |
| Windows scheduling/timing variability | [Timing diagnosis](windows-timing-diagnosis.md) |
| Original prefill profile and GEMM shapes | [Prefill investigation](prefill-investigation.md) |
| GELU methodology and activation follow-up | [Audit](gelu-benchmark-audit.md), [GEMM/activation](gemm-activation-followup.md) |
| Linear and attention experiments | [Follow-up](linear-attention-followup.md), [row reuse](linear-row-reuse.md) |
| Decode GEMV paths | [Investigation](decode-gemv-investigation.md), [results](decode-gemv-results.md) |
| MLP experiments and rejected variants | [MLP report](mlp-prefill-experiment.md), [preserved patches](../benchmark/experiments) |
| Detailed measurements, previous defaults and older model evaluations | [Full reference and history](project-reference.md) |
| Resume checkpoint | [Prefill checkpoint](prefill-checkpoint-2026-10-07.md) |
| October 15–17 release work and next actions | [Current checkpoint](checkpoint-2026-10-15.md) |
| October 20 source-preserving head storage, complete timing and resume state | [Storage checkpoint](checkpoint-2026-10-20-source-storage.md) |
| October 20 rejected prefill scheduling and worker-placement diagnosis | [Prefill jobs checkpoint](checkpoint-2026-10-20-prefill-jobs.md) |
| October 22 explicit worker placement and validation | [Worker placement checkpoint](checkpoint-2026-10-22-worker-placement.md) |
| October 27 benchmark accounting and output-buffer reuse | [Current runtime checkpoint](checkpoint-2026-10-27-output-buffers.md) |
| October 29 shared input packing, quality controls and profile | [Packing checkpoint](checkpoint-2026-10-29-shared-packing.md) |
| October 30 MLP alignment investigation and broader source storage | [Runtime checkpoint](checkpoint-2026-10-30-gemm-alignment.md) |
| Pythia attention-score sensitivity and retained tolerance failures | [Numerical investigation](pythia-numerics-handoff.md) |
| Matched Pythia-70M Leaf/llama.cpp baseline and next prefill experiment | [Runtime comparison](pythia-runtime-comparison.md) |

Older reports preserve what was known at the time; a rejected individual
experiment is distinct from a later qualified combined default. The current
release reports above explain what is actually enabled.

## Project identity

- [Leaf logo and design notes](assets/README.md)
- [Apache 2.0 license](../LICENSE)
