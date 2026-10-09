# Leaf documentation

The [project README](../README.md) covers installation, implemented capabilities,
architecture, current results and scope. Use this index for details and evidence.

## Using and developing Leaf

| Topic | Reference |
|---|---|
| Installation, local/Hub models, quantization and cache controls | [Usage](project-reference.md#2-install-and-run-with-one-command) |
| Supported decoder configurations and graph operators | [Compatibility](project-reference.md#3-coverage-and-explicit-boundaries) |
| Architecture and implementation pipeline | [Architecture](project-reference.md#4-architecture) |
| Building, tests and trained-model reproduction | [Validation guide](project-reference.md#6-reproduce-validation-and-benchmarks) |
| Native formats and artifact trust boundary | [Artifacts](project-reference.md#10-artifact-formats-and-trust-boundary) |
| Buffer liveness and memory reuse | [Memory plan](project-reference.md#11-memory-plan-format-and-runtime-reuse) |
| Repository layout | [Source map](project-reference.md#12-repository-map) |

## Current Windows release evidence

- [FP32 default kernel stack](windows-fp32-release.md): default dispatch,
  correctness, installed wheel and conservative-build comparison.
- [Attention/softmax follow-up](windows-attention-softmax.md): accepted GPT-2
  prefill improvement, fresh PyTorch comparison and retained failed attempts.
- [Current TinyLlama evaluation](windows-tinyllama-current.md): FP32/W8A8 quality,
  all timing results, stability failures and the next measured hot paths.
- [Raw benchmark records](../benchmark/results): full samples, gates and hashes.

These reports have different workloads and runtime identities. Their results
should not be combined into a universal speedup claim.

## Investigation history

| Investigation | Report |
|---|---|
| Windows scheduling/timing variability | [Timing diagnosis](windows-timing-diagnosis.md) |
| Original prefill profile and GEMM shapes | [Prefill investigation](prefill-investigation.md) |
| GELU methodology and activation follow-up | [Audit](gelu-benchmark-audit.md), [GEMM/activation](gemm-activation-followup.md) |
| Linear and attention experiments | [Follow-up](linear-attention-followup.md), [row reuse](linear-row-reuse.md) |
| Decode GEMV paths | [Investigation](decode-gemv-investigation.md), [results](decode-gemv-results.md) |
| MLP experiments and rejected variants | [MLP report](mlp-prefill-experiment.md), [preserved patches](../benchmark/experiments) |
| Detailed measurements, previous defaults and older model evaluations | [Full reference and history](project-reference.md) |
| Resume checkpoint | [Prefill checkpoint](prefill-checkpoint-2026-10-07.md) |

Older reports preserve what was known at the time; a rejected individual
experiment is distinct from a later qualified combined default. The current
release reports above explain what is actually enabled.

## Project identity

- [Leaf logo and design notes](assets/README.md)
- [Apache 2.0 license](../LICENSE)
