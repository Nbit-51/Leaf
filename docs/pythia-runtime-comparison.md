# Pythia-70M: Leaf and llama.cpp, October 14, 2026

The first matched decoder comparison is complete. Observed prefill medians favor
llama.cpp; cached single-token decode medians slightly favor Leaf. **Neither is
a qualified speed claim:** some timing passes fail stability, and the engines
have different observed numerical fidelity. No runtime optimization was promoted.

| Workload | Leaf FP32 median | llama.cpp FP32 median | Median paired Leaf/llama.cpp |
|---|---:|---:|---:|
| 63-token prefill | 56.59 ms | 43.16 ms | 1.3010 |
| One cached token after 63 tokens | 10.59 ms | 10.99 ms | 0.9539 |
| 127-token prefill | 112.84 ms | 77.28 ms | 1.4489 |
| One cached token after 127 tokens | 11.16 ms | 11.45 ms | 0.9711 |

Medians pool 66 measured samples per engine/workload. Paired ratios use the
median of six within-round median ratios, so they need not equal the pooled
median ratio. Every phase fails the complete stability criterion: p90/p10 must
be at most 1.25 in every pass and pooled samples. All raw samples are retained.
The CPU was plugged in and idle per the operator. No samples were discarded.

## Matched execution policy

- Cached `EleutherAI/pythia-70m`, snapshot
  `a39f36b100fe8a5377810d56c3f4789b9c53ac42`.
- Windows, one thread pinned to logical CPU 2 on this host, FP32 weights and
  FP32 K/V storage, zero GPU layers. llama.cpp retains its automatic attention
  selection. FP32 storage does not imply identical intermediate arithmetic.
- First 64 and 128 IDs from the saved WikiText-2 quality fixture. Each request
  prefills all but the final ID and then executes that teacher-forced final ID
  against the cache. Only the last output-logit row is requested for timing.
- Six alternating engine-order rounds; each pass uses three warmups and eleven
  samples per input. Each model remains loaded within a pass; contexts reset
  between samples. Loading, tokenization, reset, file I/O and HTTP are excluded.
  Execution and host-logit return are included. This does not measure sustained
  autoregressive generation, startup latency or an Ollama server.
- Leaf uses the normal default-policy build whose scalar and vector outputs
  match the shipped binary byte-for-byte on the three retained Pythia blocks.
- llama.cpp b11540, source revision
  `1e6f04a75e91da4bb66b5248f3fd4bdeff169d41`, matching headers and official Windows
  CPU DLLs. The unmodified upstream converter produced all-F32 GGUF. All 76
  parameter tensors match source values exactly after its QKV permutation.
  Known causal-mask/rotary buffers are metadata, excluded from parameter mapping.

## Numerical preflight

Full logits on the same 64/128-token inputs were checked against FP32 PyTorch
eager before timing. Leaf passes the existing elementwise tolerances and agrees
on all scored argmax tokens for both inputs. llama.cpp fails elementwise
tolerance and agrees on 61/63 and 120/127 scored predictions, respectively.
Its perplexity ratios against eager are 0.98915 and 0.98298; a lower loss on
these tiny inputs does not establish better task quality.

The initial run stopped after numerical preflight. It is preserved separately.
The second run explicitly selected `--diagnostic-timing`, retaining numerical
failures and keeping `overall_speed_claim=false`. This is not a relaxed quality
gate or evidence that these engines deliver equal accuracy.

The pinned llama.cpp GPT-NeoX graph selects `LLM_FFN_GELU`; its CPU GELU uses the
tanh approximation. Leaf and Pythia's configured reference use exact GELU.
This is a confirmed semantic difference, not a demonstrated explanation for the
entire discrepancy. A follow-up should isolate it before attributing errors to
conversion or a particular attention operation.

Evidence:
[timings and numerical preflight](../benchmark/results/model-onboarding/pythia70-llama-diagnostic-20261014.json),
[initial stopped run](../benchmark/results/model-onboarding/pythia70-llama-preflight-failed-20261014.json),
[GGUF conversion audit](../benchmark/results/model-onboarding/pythia70-gguf-audit-20261014.json),
[normal-build regression and probability diagnostics](../benchmark/results/model-onboarding/pythia-diagnostics-validation-20261014.json).
The larger 32-block quality failures remain open; see the
[numerical investigation](pythia-numerics-handoff.md).

## Reproduction and next experiment

With the pinned source/CPU DLLs already present, convert the original checkpoint
using `convert_hf_to_gguf.py --outtype f32` (conversion dependencies are separate
from Leaf), then build the adapter:

```powershell
./scripts/build_decoder_comparison.ps1
python tools/benchmark_decoder_competitor.py --quality-record build/output-quality-20261013/pythia70/quality.json --tokens build/output-quality-20261013/pythia70/tokens.json --leaf-executable build/pythia-diagnostics-20261014/leaf_decoder.exe --leaf-artifact build/manual-model-cache/models/0ba2eff1d3343234ba895c0d/decoder-32.leaf --llama-executable build/competitors/llama-b11540/bin/llama-decoder-benchmark.exe --gguf build/pythia-llama-20261014/pythia70-f32.gguf --output build/pythia-llama-new --cpu 2 --rounds 6 --runs 11 --warmups 3 --diagnostic-timing
```

Local build/model paths are examples; model files are not distributed in Git.
Choose a physical performance core appropriate to the host. Reports checkpoint
after every pass. Do not start another benchmark concurrently.

The next prefill implementation experiment should eliminate unneeded final-layer
rows when only last-token logits are requested. The pinned llama.cpp GPT-NeoX
graph selects output rows before its final MLP; Leaf currently computes those
MLP rows before selecting the final output row. Preserve every layer's required
K/V entries and full-logit verification behavior. Measure this shared decoder
change against Leaf's current binary, then compare the accepted build with
llama.cpp. No speedup estimate is claimed before testing.
