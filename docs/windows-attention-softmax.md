# Windows attention and softmax improvement

The Windows FP32 default now resolves the position-embedding tensor once per
forward, evaluates four attention keys together, retains value accumulators
in registers, and evaluates four ordinary softmax exponentials together.
No experimental flag is needed. Artifact format, worker scheduling, scalar
fallback and quantized dispatch are unchanged.

The exponential uses double-precision range reduction and a degree-13 Taylor
polynomial before rounding to FP32. Extreme arguments and exceptional values
retain scalar libm handling; denominator accumulation retains its order.
There is no global fast-math or flush-to-zero change. This removes overhead
observed in the MinGW scalar exponential path without assuming persistent
MLP weight packing would help an already efficient GEMM.

## Qualified incremental comparison

GPT-2 FP32, batch one, 63-token prefill, one cached decode token, last-token
logits, one thread on logical CPU 2 (a P-core), Above Normal priority.
Six fresh balanced AB/BA pairs used ten warmups and 101 retained samples per
process: 606 samples per binary. No compilation or other validation overlapped
the timing window.

| Runtime | Prefill median | Decode median |
|---|---:|---:|
| Previous optimized Windows default | 161.1558 ms | 28.64235 ms |
| Attention/softmax candidate | 145.4869 ms | 28.48115 ms |

Prefill time is **9.72% lower**; decode time is 0.56% lower. All twelve passes
and both pooled phases satisfy p90/p10 <=1.25. The predeclared thresholds,
prefill ratio <=0.98 and decode ratio <=1.02, pass. Median paired ratios are
0.90221 and 0.99026. This is a warmed native comparison on one host/workload,
not an end-to-end command or universal hardware result.

The [confirmation record](../benchmark/results/attention_softmax_confirmation_windows.json)
retains every sample. The [initial 31-sample run](../benchmark/results/attention_softmax_initial_windows.json)
showed 9.29% less prefill time but failed one individual decode spread check;
it remains a failed run. The longer confirmation was declared before execution
and did not replace the failed record. The earlier
[ILP-only candidate](../benchmark/results/attention_ilp_v1_windows.json)
also failed qualification and is preserved as an experiment patch.

## Correctness and installed runtime

- 1,052,672 sampled exponential inputs: zero observed FP32 ULP difference from
  this compiler's scalar libm; the portable test permits at most one ULP.
  This is empirical coverage, not a proof of universal correct rounding.
- Full trained GPT-2 logits, chunked logits and greedy tokens exactly match
  the previous Leaf default at one and two threads, covering 1,016 scored
  targets. Exact equality is to Leaf; the established PyTorch FP32 quality
  tolerances remain the reference requirements.
- Eight reduced architecture cases pass paired regression, including scalar,
  quantized, threaded and cached paths.
- Full Python suite: 831 passed, zero skipped, four existing ONNX warnings.
- A fresh installed wheel matches the candidate's logits/chunks/generation
  at one and two threads. Offline first and cached CLI runs generate the
  reference tokens with `optimized_fp32_active: true`, no compiler and no
  heavy framework imports. This is a local wheel verification, not a PyPI
  publication.

Evidence: [trained parity](../benchmark/results/attention_softmax_identity.json),
[architecture regression](../benchmark/results/attention_softmax_architectures.json),
[installed parity](../benchmark/results/attention_softmax_installed_identity.json),
[package check](../benchmark/results/attention_softmax_package.json).

## Fresh PyTorch comparison

The installed-wheel comparison is recorded separately from the source-build
incremental comparison. Both frameworks use the same frozen source weights,
tokens, last-token logits and cache workload. The measurement runs Leaf,
PyTorch eager and PyTorch SDPA in forward and reverse order, in fresh serial
processes, with 101 samples and ten warmups each, one thread on CPU 2 and
Above Normal priority.

The completed Windows comparison qualifies a **4.85% prefill time reduction
against PyTorch SDPA**, the fastest measured PyTorch implementation in that
phase. Decode is 1.62% lower, passing non-regression but falling short of the
2% threshold for a separate improvement claim.

| Installed runtime / framework | Prefill median | Decode median | All stability checks |
|---|---:|---:|---|
| Leaf FP32 | 144.30735 ms | 28.11020 ms | Pass |
| PyTorch eager FP32 | 153.51550 ms | 28.68440 ms | Pass |
| PyTorch SDPA FP32 | 151.66305 ms | 28.57335 ms | Pass |

Each median pools 202 samples from two opposite-order passes. Leaf's two
prefill medians are 146.1323 and 142.9671 ms; both are below their corresponding
PyTorch SDPA medians (151.5265 and 151.7344 ms). Each individual pass, pooled
phase, and between-pass drift check passes the existing 1.25 spread limit.
This supports the stated warmed GPT-2 result on this Windows CPU/configuration;
it does not establish results for other models, CPUs, operating systems,
prompt lengths or complete-command latency. Automatic precision selection is
unchanged, and this FP32 test does not requalify old quantized profiles.

The [raw framework record](../benchmark/results/attention_softmax_pytorch_windows.json)
binds the model, artifact, harness and installed runtime hashes. Its generic
timing harness does not itself perform quality validation; the separate
[combined validation record](../benchmark/results/attention_softmax_validation.json)
links that binary to the trained and installed parity checks.

## Reproduction

```powershell
python tools/benchmark_default_abba.py --before build/fp32-release/default/leaf_decoder.exe --after build/attn-softmax/candidate/leaf_decoder.exe --workdir build/gpt2_prefill_followup --output build/attn-softmax/new-native.json --pairs 6 --runs 101 --warmup 10
python tools/benchmark_core_scaling.py --executable build/attn-softmax/installed/leaf/bin/leaf_decoder.exe --workdir build/gpt2_prefill_followup --model build/models/gpt2 --output build/attn-softmax/new-framework.json --cores 2 --counts 1 --default-kernels --runs 101 --warmup 10
```

Output paths must be fresh. Historical source and wheel binary hashes differ;
each performance claim belongs to the exact executable recorded with it.
