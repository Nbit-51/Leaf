# Benchmark result records

Every record belongs to a specific workload, model/data identity, precision,
runtime binary, thread configuration, and device environment. Compare latency
only when these conditions match. Large datasets, model weights, exported
artifacts, and intermediate caches remain local and Git-ignored.

The repository [README](../../README.md) contains the step-by-step methods,
current trained-model tables, all historical benchmark tables, and remaining
coverage/quality/hardware work.

## Current trained-model and full-dataset records

| File | What it records | Decision and scope |
|---|---|---|
| [tinyllama_full_decoder.json](tinyllama_full_decoder.json) | Complete trained 1.1B decoder, native FP32/W8A32/W4A32/smoothed W8A8, PyTorch eager/SDPA, generation, memory, and raw latency samples | Smoothed W8A8 qualifies on the frozen tested Windows CPU/runtime configuration; held-out 1,016-target subset, not whole-corpus perplexity |
| [gpt2_full_decoder.json](gpt2_full_decoder.json) | Complete trained GPT-2 across seven precision policies, independent calibration, same quality/generation/latency gates | Protected INT8 passes quality but loses prefill; other quantized policies fail quality; none automatically selected |
| [cifar10_full_trained.json](cifar10_full_trained.json) | Trained ResNet-20 on all 10,000 test images; separate 64-training-image calibration; two full PyTorch and order-alternated native passes | FP32 matches accuracy/logits; INT8 loses 0.42 percentage points but passes task-quality limit; unstable native baseline and slower-than-PyTorch candidates remain unpromoted |
| [decoder_architectures_final.json](decoder_architectures_final.json) | Final guarded default runtime: eight complete reduced random-weight cases across Llama, Qwen2, GPT-2, GPT-NeoX, and OPT families, experimental float flags disabled | Semantic, scalar/vector, threads, cache, generation, and precision checks; not trained model-quality evidence |
| [decoder_architectures_experimental.json](decoder_architectures_experimental.json) | Same eight reduced random-weight cases with both experimental float flags enabled | Correctness passes; failed stability/speed records still prevent promotion |
| [decoder_portability_final.json](decoder_portability_final.json) | Final guarded default runtime: eighteen reduced-model/precision cases across Windows and Linux under WSL on the same physical CPU | Cross-OS execution sanity; no independent hardware, trained Linux quality, or latency claim |
| [decoder_portability_windows.json](decoder_portability_windows.json) | Local Windows protocol/precision/cache/generation portability checks | Same reduced-model scope; not another device |
| [package_runtime_final.json](package_runtime_final.json) | Final actual installed platform-wheel first offline preparation and cached repeat, no Torch imports/compiler on PATH, exact GPT-2 reference generation, native/profile identity | External wall observations 3,619.4554 and 1,562.5659 ms; one observation per stage, OS file cache not flushed, not a speed gate |
| [tinyllama_cli_runtime.json](tinyllama_cli_runtime.json) | Five fresh cached full-command FP32 and five genuinely eligible auto W8A8 runs with the frozen validated binary and saved prompt IDs | Stable complete-command medians: 11,040.3993 versus 4,778.9251 ms (2.3102×); FP32 first-visible timing is unstable, so no first-token speed qualification; no new PyTorch/wheel qualification |

Trained records preserve the completed model measurements and their actual
runtime/artifact hashes. Newer export-request provenance hardening does not
retroactively reseal those earlier records as a new validation. Fresh quality
validation generates sealed cache/export identities for later timing-only
refreshes.

A platform/CPU signature is not a unique machine identity. Never copy a profile
to another machine and describe it as validated there. Changed binaries,
precision policies, calibration, models, or inputs require the applicable new
checks.

The final guarded runtime/wheel has not received a fresh trained latency
qualification. Older profiles with a different native binary identity correctly
fall back to FP32; architecture or installation parity is not speed evidence.

## Iterative decoder experiments and rejected candidates

| File | What it preserves |
|---|---|
| [tinyllama_decoder_initial.json](tinyllama_decoder_initial.json) | Initial complete trained native FP32, weight-only INT8/INT4 quality and slow latency |
| [tinyllama_decoder_dynamic_i8.json](tinyllama_decoder_dynamic_i8.json) | Unsmoothed activation-INT8 candidates; failed trained quality and latency results |
| [tinyllama_decoder_weight_only.json](tinyllama_decoder_weight_only.json) | Revised weight-only paths; quality/speed decisions before later prefill work |
| [tinyllama_decoder_smoothing_pre_tiling.json](tinyllama_decoder_smoothing_pre_tiling.json) | Smoothing restores the predefined INT8 quality threshold, but prefill still fails the speed gate |
| [tinyllama_decoder_unstable_latency.json](tinyllama_decoder_unstable_latency.json) | Noisy full decoder timing window; raw outliers retained |
| [tinyllama_decoder_before_latency_refresh.json](tinyllama_decoder_before_latency_refresh.json) | Full native quality record before the accepted stable timing refresh |
| [gpt2_decoder_initial.json](gpt2_decoder_initial.json) | Initial trained GPT-2 FP32 and four standard precision cases before protected/grouped experiments |
| [decoder_prefill_tiling.json](decoder_prefill_tiling.json) | Same-session AVX2 4-row/2-token prefill comparison, bit-exact outputs, raw ABBA samples |
| [decoder_prefill_tile2_rejected.json](decoder_prefill_tile2_rejected.json) | Rejected AVX2 2-row/4-token alternative; inconsistent small difference |
| [decoder_prefill_vnni.json](decoder_prefill_vnni.json) | Same-binary optional VNNI comparison, constructor cost, exact zero-point correction, and rejected register-spilling wider tile |
| [gpt2_float_prefill_abba.json](gpt2_float_prefill_abba.json) | Specialized float prefill candidate with full 1,016-target quality/chunk/generation checks; rejected for instability and decode regressions |
| [gpt2_float_gemv_abba.json](gpt2_float_gemv_abba.json) | Float GEMV candidate with full trained checks; apparent pooled speed improvements rejected because stability fails |
| [gpt2_float_high_qos_abba.json](gpt2_float_high_qos_abba.json) | Verified process-scoped HighQoS applied to both native stages; both quality checks pass, instability and protected-INT8 decode regression prevent acceptance |
| [gpt2_float_cpu6_abba.json](gpt2_float_cpu6_abba.json) | Fresh native ABBA on logical CPU 6 with verified child HighQoS; quality passes but stability fails; CPU-2 historical PyTorch timing is context only |

Early agreement records included the final position of each block in an
argmax-only comparison; current trained tables score the same 1,016 causal
target positions for agreement and perplexity. Do not compare those agreement
denominators without accounting for that change.

Kernel-candidate timing records do not replace full trained quality and stable
PyTorch/native speed validation. Measurements from distinct sessions cannot be
subtracted to claim a controlled speed improvement. Failed or unstable
candidates are part of the evidence.

New native comparisons never rewrite the frozen before record or authorize
automatic precision selection. Every pass, pooled sample, and between-pass
stability check is retained. The unqualified float tile/GEMV paths require
explicit `LEAF_EXPERIMENTAL_FLOAT_TILES` / `LEAF_EXPERIMENTAL_FLOAT_GEMV` opt-ins,
remain off by default, and invalidate automatic lower-precision selection.
HighQoS is not a production CLI default or persistent system power-plan change;
its measurements do not prove the cause of timing variation.

The full-command record uses cached artifacts, no export/build/download, and
alternating case order. Its integrity preflight warms the OS file cache. External
timers include interpreter startup; native constructor and prompt forward
timings do not. Startup, hashing, tokenizer, IPC, and rendering share the remaining
interval, which is not an isolated Python timer. Auto-generated tokens are
checked against the selected quantized reference rather than declared identical
to FP32/PyTorch.

`LEAF_DECODER_PROFILE` phase diagnostics aggregate all calls, including warmups.
They are not additive independent forward components and are excluded from
timed experiment children. Profiling is not a precision/performance qualification.

## Current automated checks

| File | What it records |
|---|---|
| [hosted_cpu_validation.json](hosted_cpu_validation.json) | Successful Ubuntu 24.04, Windows Server 2022, and macOS 14 hosted builds, selected Python tests, and eight reduced-model architecture cases for implementation commit `ce5fe66`; not trained-model or latency qualification |
| [native_latest.json](native_latest.json) | Refreshed scalar versus optimized native FP32/INT8 GEMM and Conv speed gates |
| [latest.json](latest.json) | Refreshed deterministic synthetic CNN/FFN parity, calibration, Python-oracle latency, and memory plans |
| [memory_plans/](memory_plans/) | Refreshed deterministic workload liveness plans |

These are microkernel and pipeline checks, not trained-model accuracy, full
decoder perplexity, or whole-model speed claims.

## Historical measurements retained

| File | What it records |
|---|---|
| [history/pre-decoder/native_latest.json](history/pre-decoder/native_latest.json) | Byte-for-byte snapshot of the earlier native kernel table before the latest complete verification rerun |
| [history/pre-decoder/latest.json](history/pre-decoder/latest.json) | Earlier deterministic synthetic parity/latency table retained in the repository README |
| [history/pre-decoder/decoder_architectures.json](history/pre-decoder/decoder_architectures.json) | Earlier reduced-model decoder verification record |
| [history/pre-decoder/memory_plans/](history/pre-decoder/memory_plans/) | Matching earlier deterministic liveness plans |
| [decoder_architectures.json](decoder_architectures.json) | Earlier eight reduced random-weight architecture cases before the final guarded-default verification |
| [decoder_portability.json](decoder_portability.json) | Earlier eighteen same-physical-CPU Windows/WSL reduced-model checks |
| [package_runtime.json](package_runtime.json) | Earlier installed-wheel first preparation/cached repeat observations; preserved separately from the final wheel check |
| [cifar10_cpu.json](cifar10_cpu.json) | Real CIFAR subset on a small untrained optimizer graph; PyTorch versus NumPy FP32/INT8 simulation |
| [cifar10_native_int8.json](cifar10_native_int8.json) | Same untrained real-image subset with native FP32/INT8 parity and latency |
| [native_int8_graph.json](native_int8_graph.json) | Synthetic native INT8 integration, FP32 comparison, latency, and peak RSS |
| [memory_plan_native.json](memory_plan_native.json) | Graph v3 arena versus v2 buffer-pool ResNet parity, latency, and peak RSS |
| [native_transformer_ops.json](native_transformer_ops.json) | Standalone RMSNorm PyTorch parity and portable/AVX2 latency |
| [native_attention.json](native_attention.json) | Standalone masked attention prefill/decode parity and portable/AVX2 latency |
| [native_swiglu.json](native_swiglu.json) | Fused/unfused SwiGLU parity and order-alternated latency |
| [native_rope_repeatkv.json](native_rope_repeatkv.json) | Standalone dual-output RoPE and RepeatKV parity/latency, including graph v3 plans |
| [native_kv_cache.json](native_kv_cache.json) | Single-attention-node two-step GQA session/cache parity and latency |
| [tinyllama_cached_cpu.json](tinyllama_cached_cpu.json) | Historical complete TinyLlama PyTorch recompute versus populated KV-cache reference |
| [tinyllama_leaf_coverage.json](tinyllama_leaf_coverage.json) | Initial reduced ONNX decoder-coverage probe; trained weights were not loaded |
| [cifar10_resnet18_cpp.json](cifar10_resnet18_cpp.json) | First 20 real test images, complete seeded FP32 ResNet-18, PyTorch versus C++ |
| [cifar10_resnet18_cpp_repeat.json](cifar10_resnet18_cpp_repeat.json) | Independent repeat of that untrained full-model comparison |
| [cifar10_resnet18_cpp_transformer_update.json](cifar10_resnet18_cpp_transformer_update.json) | Untrained full ResNet real-image parity/latency after Transformer additions |
| [cifar10_resnet18_transformer_no_regression.json](cifar10_resnet18_transformer_no_regression.json) | Alternating prior/current executor ResNet latency check |
| [cifar10_resnet18_kv_no_regression.json](cifar10_resnet18_kv_no_regression.json) | Alternating ResNet check after RoPE/RepeatKV/KV changes |
| [qwen25_cached_cpu.json](qwen25_cached_cpu.json) | Cached Qwen2.5-0.5B PyTorch-only recompute/decode reference |
| [wsl_dev_machine.csv](wsl_dev_machine.csv) | Historical 224×224 FP32 ResNet runtime stages; different measurement sessions are labeled |

Historical ResNet/CNN records use untrained weights and measure output parity
and latency, not classifier accuracy. Qwen's cache record is a PyTorch
reference, not a full-Qwen Leaf result. The initial ONNX-coverage probe remains
historical evidence for that import route; complete trained decoders now run
through the separate declarative `LEAFDC02` route.

The graph INT8 measurements do not establish a latency default. Portable
execution and a passing local profile do not establish performance on every CPU.
Reproduce model-quality and whole-workload speed checks on the intended device.
