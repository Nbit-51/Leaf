# Benchmark result records

Every record belongs to a specific workload, model/data identity, precision,
runtime binary, thread configuration, and device environment. Compare latency
only when these conditions match. Large datasets, model weights, exported
artifacts, and intermediate caches remain local and Git-ignored.

The repository [README](../../README.md) contains the step-by-step methods,
current trained-model tables, all historical benchmark tables, and remaining
coverage/quality/hardware work.

## Matched evaluations and preserved model/dataset records

| File | What it records | Decision and scope |
|---|---|---|
| [gpt2_windows_fair.json](gpt2_windows_fair.json) | Fresh default RoPE-cache Windows runtime, full 1,016-target quality and matched last-token-only PyTorch/native prefill/decode, ten warmups and 21 samples per phase | Quality/workload/stability pass. Protected INT8 380.2857/18.3121 ms versus fastest PyTorch 152.3876/28.6706 ms; no automatic promotion because prefill loses |
| [gpt2_linux_fair.json](gpt2_linux_fair.json) | Fresh matched GPT-2 CPU evaluation under Linux/WSL on the same host, full quality/generation and 21 samples per phase | Quality/workload/stability pass. Protected INT8 419.546159/19.064593 ms versus fastest PyTorch 151.412496/28.872263 ms; no promotion because PyTorch prefill is faster. Distinct software environments, not a controlled OS comparison |
| [tinyllama_windows_fair.json](tinyllama_windows_fair.json) | Fresh matched complete 1.1B decoder, three precision policies, independent 512-token calibration and 1,016-target quality evaluation | Smoothed W8A8 passes quality/workload/stability and automatic gates on exact Windows SHA `e952ae…`: 1,311.0678/87.1424 ms, 1.3242×/2.3172× faster than fastest matched PyTorch phases; generation is not exact |
| [tinyllama_full_decoder.json](tinyllama_full_decoder.json) | Complete trained 1.1B decoder, native FP32/W8A32/W4A32/smoothed W8A8, PyTorch eager/SDPA, generation, memory, and raw latency samples | Held-out 1,016-target quality remains valid; old automatic eligibility/prefill speedup is superseded because PyTorch projected all prefix logits while Leaf projected the last token only |
| [gpt2_full_decoder.json](gpt2_full_decoder.json) | Complete trained GPT-2 across seven precision policies, independent calibration, quality/generation and historical latency | Protected INT8 passes quality; other quantized policies fail quality. Old unmatched-prefill timings are retained, not current automatic qualification |
| [cifar10_full_trained.json](cifar10_full_trained.json) | Trained ResNet-20 on all 10,000 test images; separate 64-training-image calibration; two full PyTorch and order-alternated native passes | FP32 matches accuracy/logits; INT8 loses 0.42 percentage points but passes task-quality limit; unstable native baseline and slower-than-PyTorch candidates remain unpromoted |
| [decoder_architectures_final.json](decoder_architectures_final.json) | Final guarded default runtime: eight complete reduced random-weight cases across Llama, Qwen2, GPT-2, GPT-NeoX, and OPT families, experimental float flags disabled | Semantic, scalar/vector, threads, cache, generation, and precision checks; not trained model-quality evidence |
| [decoder_architectures_experimental.json](decoder_architectures_experimental.json) | Same eight reduced random-weight cases with both experimental float flags enabled | Correctness passes; failed stability/speed records still prevent promotion |
| [decoder_rope_regression.json](decoder_rope_regression.json) | RoPE-cache change paired with the preceding default binary across eight reduced cases, all precision policies, scalar/two-thread execution, custom plans, odd lengths and cache/reset/generation requests | Bit-exact logits/tokens on identical requests; no trained latency qualification |
| [decoder_token_panel_default_regression.json](decoder_token_panel_default_regression.json) | Packed-candidate binary with default flags versus RoPE-cache baseline: eight reduced cases, all precisions and cache/reset/generation requests | Paired bit-exact default outputs; no latency or automatic promotion claim |
| [decoder_portability_final.json](decoder_portability_final.json) | Final guarded default runtime: eighteen reduced-model/precision cases across Windows and Linux under WSL on the same physical CPU | Cross-OS execution sanity; no independent hardware, trained Linux quality, or latency claim |
| [decoder_portability_windows.json](decoder_portability_windows.json) | Local Windows protocol/precision/cache/generation portability checks | Same reduced-model scope; not another device |
| [package_runtime_final.json](package_runtime_final.json) | Earlier installed platform-wheel first offline preparation and cached repeat, no Torch imports/compiler on PATH, exact GPT-2 reference generation, native/profile identity | Historical external wall observations 3,619.4554 and 1,562.5659 ms; one observation per stage, OS file cache not flushed, not a speed gate |
| [package_runtime_checkpoint.json](package_runtime_checkpoint.json) | Current installed Windows wheel first offline preparation/cached repeat: exact 16 GPT-2 tokens, no compiler/heavy-library imports, artifact reuse and FP32 fallback | External wall 3,130.2889/1,492.7168 ms; one observation per stage, not a speed gate. Earlier package records stay historical |
| [tinyllama_cli_runtime.json](tinyllama_cli_runtime.json) | Five fresh cached full-command FP32 and five historically selected auto W8A8 runs with the frozen binary and saved prompt IDs | Observed complete-command medians: 11,040.3993 versus 4,778.9251 ms (2.3102×). Its legacy profile is now rejected for workload mismatch; no current automatic qualification or first-token speed claim |

Trained records preserve the completed model measurements and their actual
runtime/artifact hashes. Newer export-request provenance hardening does not
retroactively reseal those earlier records as a new validation. Fresh quality
validation generates sealed cache/export identities for later timing-only
refreshes.

The legacy decoder PyTorch prefill projected logits for all prefix positions;
Leaf projected only the last token. Raw JSON remains unchanged, including old
eligibility fields, but those fields are not current authorization. Updated
validation requires `leaf-decoder-latency-workload-v1`, batch-one prefix prefill,
KV-cached one-token decode, and `logits: last_token_only` on both runtimes.
PyTorch uses and verifies `logits_to_keep=1`. The current CLI/gates reject absent
or mismatched workload tags. Old decode timings remain observations; they alone
cannot authorize lower precision. Held-out all-position quality and the full
CIFAR workload are separate from this latency correction.

Automatic selection revalidates every PyTorch implementation, native FP32,
and candidate workload tag and recomputes gates rather than trusting stored
positive decisions. Reported medians must match retained raw samples within
the narrow six-significant-digit native-JSON rounding tolerance; inconsistent
data fail closed. Native comparisons recheck executable/artifact/reference/cache
identities before and after the run. These checks do not retroactively rewrite
historical records or turn stored rejections into approvals.

A platform/CPU signature is not a unique machine identity. Never copy a profile
to another machine and describe it as validated there. Changed binaries,
precision policies, calibration, models, or inputs require the applicable new
checks.

The installed-wheel smoke is not trained performance qualification. The newer
matched GPT-2 Windows record passes quality/stability but still fails the
automatic prefill gate, as does the Linux record. The fresh matched TinyLlama
record qualifies smoothed W8A8 only for its exact tested Windows native binary
and configuration. Older profiles with a different native binary identity
correctly
fall back to FP32, as do legacy workload profiles even if the binary matches;
architecture or installation parity is not speed evidence.

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
| [gpt2_default_runtime_abba.json](gpt2_default_runtime_abba.json) | Default guarded binary, both float flags unset and no HighQoS: quality passes; FP32 prefill/decode ratios 1.9335/1.3258 and protected INT8 0.8447/1.0118 are unstable, so both rejected; not fresh matched PyTorch/automatic qualification |
| [decoder_token_panel_architectures.json](decoder_token_panel_architectures.json) | Eight reduced configurations/five architecture families with the packed FP32 token-panel flag enabled; correctness only |
| [gpt2_token_panel_windows_abba.json](gpt2_token_panel_windows_abba.json) | AFTER-only packed-FP32 policy; full trained checks pass, but FP32 and protected-INT8 timings are unstable and rejected |
| [gpt2_token_panel_linux_abba.json](gpt2_token_panel_linux_abba.json) | Linux FP32 native experiment passes quality/stability/no-slowdown: 450.8723145→220.6642555 ms prefill (2.0433×). Protected INT8 fails stability/decode; no fresh PyTorch, automatic promotion, or default-policy authorization |
| [gpt2_token_panel_windows_highqos_abba.json](gpt2_token_panel_windows_highqos_abba.json) | Child-only verified HighQoS on both stages; quality passes but both policies are unstable and rejected; no global power change |
| [tinyllama_token_panel_windows_abba.json](tinyllama_token_panel_windows_abba.json) | Full held-out larger-model comparison: FP32 quality/stability/native gates pass, 3,636.238→2,167.06375 ms prefill (1.6780×), decode +0.5896% within the 2% allowance. Smoothed W8A8 is unstable/no improvement; no fresh PyTorch, automatic promotion, or default-policy authorization |

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

At this checkpoint fresh repeated full-command measurements using the matched
TinyLlama profile are incomplete. The current installed-wheel offline smoke
passes but its single observations are not promotion evidence. The packed
candidate stays off by default;
current command observations remain the separately labeled historical record.
The separate RB96/BK256 blocked-K primitive passes Windows/WSL vector/scalar,
bit-exactness, stride and tail tests but is not decoder-dispatched. The
2026-10-03 shape experiments below show no demonstrated advantage.

| Follow-up record | Scope and outcome |
|---|---|
| [gpt2_phase_diagnostics_windows.json](gpt2_phase_diagnostics_windows.json) | Same-binary default/packed/packed/default phase profiles; warmup-inclusive means, no acceptance timing. Linear occupies about 81% default and 65–66% packed prefill; activation is next |
| [gpt2_token_panel_shapes_windows.json](gpt2_token_panel_shapes_windows.json) | Packing-inclusive full-K vs RB96/BK256 on four M=63 shapes. FP64 reference and bit-exactness pass; all timing comparisons unstable |
| [gpt2_token_panel_shapes_linux.json](gpt2_token_panel_shapes_linux.json) | Same-host WSL shape checks pass; three stable comparisons show blocked-K 0.8–3.0% slower; prospective fused QKV unstable. No integration or promotion |
| [gpt2_prefill_followup_matched_windows.json](gpt2_prefill_followup_matched_windows.json) | Fresh Windows 26300 PyTorch/native FP32 baseline; quality and timing stability pass. Fastest PyTorch 156.2839 / 28.1867 ms prefill/decode; default Leaf 392.9278 / 33.3914 ms |
| [gpt2_prefill_followup_windows_abba.json](gpt2_prefill_followup_windows_abba.json) | Same-binary default/packed comparison against the new frozen quality reference. Quality passes; unstable timings and +3.69% decode reject promotion, despite lower observed prefill |
| [decoder_prefill_followup_default_architectures.json](decoder_prefill_followup_default_architectures.json) | Fresh binary, default policy: eight reduced cases/five families pass scalar/vector, two-thread FP32, cache and generation checks |
| [decoder_prefill_followup_packed_architectures.json](decoder_prefill_followup_packed_architectures.json) | Same reduced architecture checks with packed FP32 enabled; correctness only |

See [the investigation](../../docs/prefill-investigation.md) for commands,
phase normalization, dimensions, limitations, and the prioritized next steps.
These new records preserve the earlier measurements. The current Windows
build is 26300, while the historical matched baseline used 26200; the frozen
comparison harness rejects that platform mismatch.

The [GEMM/activation follow-up](../../docs/gemm-activation-followup.md) retains
the next experiments separately:

| Records | Outcome |
|---|---|
| [Weight-panel Windows shapes](gpt2_weight_panel_shapes_windows.json) | Numerically correct, slower observed medians and unstable; undispatched |
| [Wide-token Windows](gpt2_wide_token_shapes_windows.json), [Linux](gpt2_wide_token_shapes_linux.json) | Correct; no stable improvement; undispatched |
| [Unrolled Linux shapes](gpt2_unrolled_token_shapes_linux.json) | Correct and stable, but all four ratios within about 1% of full-K; below the 2% improvement threshold |
| [GELU Windows BEFORE](gpt2_gelu_before_packed_windows.json), [initial ABBA](gpt2_gelu_packed_windows_abba.json), [final ABBA](gpt2_gelu_packed_windows_final_abba.json) | Trained quality passes; 21–24% lower observed prefill, unstable timing rejects promotion |
| [GELU Linux BEFORE](gpt2_gelu_before_packed_linux.json), [ABBA](gpt2_gelu_packed_linux_abba.json) | Trained quality and AFTER stability pass; unstable first BEFORE pass rejects the combined gate |
| [GELU Windows phases](gpt2_gelu_phase_windows.json) | Diagnostic means with warmups: packed-prefill activation around 2 ms / 1%; linear remains dominant |
| [GELU Linux confirmation](gpt2_gelu_packed_linux_confirmation_abba.json) | Longer predeclared run; prefill and AFTER stable, first BEFORE decode unstable; combined gate rejects promotion |
| [Default regression](gemm_activation_default_regression.json) | Eight reduced architecture cases bit-exact to the prior executable |
| [Experimental architectures](gemm_activation_experimental_architectures.json) | Eight reduced architecture cases pass; implementation parity, not trained performance qualification |
| [GELU error/benchmark audit](gpt2_gelu_error_benchmark_audit.json) | Fresh paired logits, independent float64 cross-entropy, exact replay of all four saved timing verdicts; no new latency qualification |
| [Identical-binary Windows A/A](gpt2_gelu_benchmark_aa_windows.json) | Same binary/settings appear 15.31% faster in pooled prefill; unstable timing and slower decode correctly reject the control |
| [Windows environment diagnosis](gpt2_windows_environment_summary.json) | Aggregate telemetry and raw timing; monitored controls stable, HighQoS not a large remedy; initial high-overhead probe retained separately |
| [Quiet Windows GELU ABBA](gpt2_gelu_quiet_windows_abba.json) | Quality passes; first baseline decode spread 1.26092 rejects the full gate despite lower prefill |
| [Quiet matched PyTorch/Leaf](gpt2_gelu_quiet_matched_windows.json) | Experimental Leaf stable, PyTorch decode unstable; observed Leaf gap 8.71% prefill / 6.81% decode, no promotion |
| [Initial linear/attention Windows](linear_attention_shapes_windows.json), [Linux](linear_attention_shapes_linux.json) | Synthetic attention/QKV ABBA; stable Linux shared packing is 1.16% slower, so no linear dispatch change |
| [Refined linear/attention Windows](linear_attention_shapes_windows_refined.json), [Linux](linear_attention_shapes_linux_refined.json) | Stable attention operator cases reduce time 35–58%; QKV evidence mixed; no whole-model claim |
| [Attention default regression](attention_default_regression.json), [experimental architectures](attention_vector_architectures.json) | Eight cases each: default bit-exact; experimental parity, cache and generation checks pass |
| [Attention Windows ABBA](gpt2_attention_windows_abba.json) | Trained quality passes; observed prefill 7.95% lower, baseline decode unstable; rejected |
| [Attention Linux BEFORE](gpt2_attention_before_linux.json), [ABBA](gpt2_attention_linux_abba.json) | Trained quality passes; observed prefill 2.22% lower, decode 5.84% higher and unstable; rejected |
| [Attention Windows phases](gpt2_attention_phase_windows.json) | Diagnostic warmup-inclusive means: packed prefill linear 85–86%, attention 9–10%, activation about 1% |
| [Row reuse single-matrix Windows](gpt2_row_reuse_shapes_windows.json), [rotating weights](gpt2_row_reuse_rotating_windows.json) | Warm-matrix evidence mixed; rotating active projection shapes stable and 2.08–4.45% lower time |
| [Row reuse Windows BEFORE](gpt2_row_reuse_before_windows.json), [ABBA](gpt2_row_reuse_windows_abba.json) | All native experiment gates pass: 4.07% lower prefill, no decode regression, stable timing and trained quality |
| [Row reuse fresh Windows framework comparison](gpt2_row_reuse_matched_windows.json) | Leaf and eager stable; Leaf remains 4.77% slower prefill / 11.89% slower decode than eager; SDPA unstable |
| [Row reuse default regression](row_reuse_default_regression.json), [experimental architectures](row_reuse_architectures.json) | Eight cases each pass; default bit-exact; scalar/two-thread/cache/generation checks |
| [Row reuse WSL2 shapes](gpt2_row_reuse_rotating_wsl.json), [BEFORE](gpt2_row_reuse_before_wsl.json), [ABBA](gpt2_row_reuse_wsl_abba.json) | MLP shapes improve; whole-model quality passes and prefill observes 4.53% lower time, but decode instability rejects the gate |

The vector GELU build is off by default. Its final binary emits an experimental
build marker that disallows automatic precision selection. These ABBA runs use
frozen quality references and do not constitute fresh PyTorch qualification.
The attention build is also off by default and excluded from automatic selection;
see the [follow-up and decision diagram](../../docs/linear-attention-followup.md).
The [row-reuse investigation](../../docs/linear-row-reuse.md) records a passing
incremental Windows native experiment. Its build option remains off by default;
the measured configuration does not qualify broader model/hardware performance.

The full-command record uses cached artifacts, no export/build/download, and
alternating case order. Its integrity preflight warms the OS file cache. External
timers include interpreter startup; native constructor and prompt forward
timings do not. Startup, hashing, tokenizer, IPC, and rendering share the remaining
interval, which is not an isolated Python timer. Historically auto-selected tokens are
checked against the selected quantized reference rather than declared identical
to FP32/PyTorch.

`LEAF_DECODER_PROFILE` phase diagnostics aggregate all calls, including warmups.
They are not additive independent forward components and are excluded from
timed experiment children. Profiling is not a precision/performance qualification.

## Paired-row GEMV Windows experiment (2026-10-06)

The [engineering report](../../docs/decode-gemv-results.md) explains the
accepted native decode gain, prefill tradeoff and framework-order limitation.

| File | Evidence |
|---|---|
| [gpt2_decode_gemv_before_windows.json](gpt2_decode_gemv_before_windows.json) | Fresh before-build trained quality baseline |
| [gpt2_gemv_specialization_windows.json](gpt2_gemv_specialization_windows.json) | Existing specialization control; unstable |
| [gpt2_gemv_pair_shapes_windows.json](gpt2_gemv_pair_shapes_windows.json) | Paired shape timings; no qualified isolated speed gain |
| [gpt2_gemv_pair_windows_abba.json](gpt2_gemv_pair_windows_abba.json) | Native decode -10.52%, prefill +0.91%; all gates pass |
| [gpt2_gemv_pair_matched_windows.json](gpt2_gemv_pair_matched_windows.json) | Sequential eager, SDPA, Leaf comparison; not order-balanced |
| [gpt2_gemv_pair_cached_logits_windows.json](gpt2_gemv_pair_cached_logits_windows.json) | All held-out tokens decoded individually; bit-exact before/after |
| [gemv_pair_default_regression.json](gemv_pair_default_regression.json) | Eight default architecture cases |
| [gemv_pair_architectures.json](gemv_pair_architectures.json) | Eight experimental architecture cases |
| [gemv_pair_validation_windows.json](gemv_pair_validation_windows.json) | Python/native checks, hashes and gate replay |

## MLP prefill Windows experiment (2026-10-07)

See the [MLP report](../../docs/mlp-prefill-experiment.md). No whole-model
improvement is accepted from this experiment.

| File | Evidence |
|---|---|
| [gpt2_mlp_panel_rotating_windows.json](gpt2_mlp_panel_rotating_windows.json) | Panel-blocking candidate; all shape comparisons unstable; not dispatched |
| [gpt2_mlp_pack_rotating_windows.json](gpt2_mlp_pack_rotating_windows.json) | Vector packing: stable MLP-up/down reductions of 1.09% / 2.85% |
| [gpt2_mlp_pack_windows_abba.json](gpt2_mlp_pack_windows_abba.json) | Quality passes; observed prefill -2.46%, but stability fails |
| [gpt2_mlp_pack_above_normal_windows_abba.json](gpt2_mlp_pack_above_normal_windows_abba.json) | Controlled priority: all stability checks pass, prefill -1.46%, below 2% target |
| [mlp_cost_diagnostic_windows.json](mlp_cost_diagnostic_windows.json) | Thread counters distinguish major off-CPU stalls; packing saves about 0.85 ms |
| [mlp_pack_environment_windows.json](mlp_pack_environment_windows.json) | Aggregate host counters; HighQoS has no demonstrated advantage |
| [gpt2_mlp_wide_reuse_rotating_windows.json](gpt2_mlp_wide_reuse_rotating_windows.json) | Three-row/wide-panel reuse; MLP shapes unstable, not integrated |
| [mlp_pack_default_regression.json](mlp_pack_default_regression.json) | Eight paired bit-exact default architecture cases |
| [mlp_pack_architectures.json](mlp_pack_architectures.json) | Eight experimental architecture cases |
| [mlp_prefill_validation_windows.json](mlp_prefill_validation_windows.json) | Tests, hashes and raw-sample replay |
| [mlp_prefill_final_validation_windows.json](mlp_prefill_final_validation_windows.json) | 813 tests, both gate replays, and byte-identical normal-build PE sections after diagnostic additions |

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
