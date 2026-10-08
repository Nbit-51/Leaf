# Prefill investigation checkpoint — 2026-10-07 (Windows, GPT-2 FP32)

Durable handoff note. Baseline is the shipped Windows FP32 default
(`build/fp32-release/default/leaf_decoder.exe`, commit `5163075`).

## CI

`5163075` failed only on macOS-14: `weight-panel vector output is not bit-exact
against full-K`. Cause (confirmed via AArch64 assembly): Clang vectorizes the
contiguous weight-panel portable loop with strict in-order, *unfused*
reductions, while contracting the strided token-panel portable loop to
`fmadd`. Same order, different rounding. Fix `18b5bcc` compiles the
token-panel test with `-ffp-contract=off`; production numerics are unchanged.
CI is green on Linux, Windows and macOS for `18b5bcc`.

## Step 1–2: is there MLP GEMM headroom? (Hypothesis contradicted)

Hot kernel: `gemm_token_panels_f32_row_reuse` → `token_panel_detail::tile<6>`.
GCC 15.2 inner loop per K step: 2 contiguous input loads, 6 weight
broadcasts, 12 FMAs, 15 YMM registers, no spills.

Measured on CPU 2, Above Normal, one thread (records in
`benchmark/results/mlp-headroom/`):

| Shape (M=63) | Leaf row-reuse, 12 rotating weights | PyTorch 2.12 `F.linear`, 12 rotating |
|---|---:|---:|
| MLP up N3072 K768 | 3.38 / 2.98 ms (two runs) | 4.00 ms |
| MLP down N768 K3072 | 3.37 / 3.55 ms | 3.36 ms (passes 4.01, 3.24) |

`gemm_ceiling_benchmark.cpp` measures independent register-only 256-bit FMA
throughput adjacent to each case. The shipped kernel reaches 0.90–1.09 of
that measured ceiling on both MLP shapes, hot or rotating (the ceiling itself
varies about ±10% with frequency state). No nominal clock is assumed.

Conclusion: these measurements are consistent with little FMA-throughput
headroom at this instruction count; they do not prove a hardware ceiling for
every clock state or shape. Isolated MKL did not show a consistent advantage.
Persistent output-channel weight packing keeps
the same 2-load/6-broadcast/12-FMA mix and would add ~216 MiB, so it was
**not implemented**.

## Profile of the shipped default (`LEAF_DECODER_PROFILE=1`, 25 forwards)

Per prefill forward: linear 145.6 ms, attention 19.1 ms, embedding 3.3 ms,
layer norm 2.8 ms, activation 2.2 ms. Attention is ~12 MFLOP/layer but costs
~1.6 ms/layer — far below GEMM throughput, so it is the better-supported
next target.

## Predeclared candidate (written before measurement)

Candidate `attn-ilp`: changes only instruction scheduling, never FP order.

1. Hoist the position-embedding tensor lookup out of the per-element
   embedding loop (it currently builds a `std::string` and hashes it 63×768
   times per prefill).
2. Vector prefill attention: compute Q·K for four unmasked keys at once
   (each key keeps its own accumulator, chunk order and in-order horizontal
   sum), and keep P·V output accumulators in registers across keys in
   64-dimension blocks (same key order, separate multiply then add).

Gates:
- Native test: new kernel output `memcmp`-equal to the previous kernel
  (kept in the test as a reference) across dims, GQA, masks, polarity, NaN.
- Trained GPT-2: logits bit-identical to the shipped default on all held-out
  targets, chunked cache and greedy generation.
- Whole model, fresh serial balanced AB/BA, one thread on CPU 2, Above
  Normal, 10 warmups, 31 samples: prefill ratio ≤ 0.98, decode ratio ≤ 1.02,
  existing stability checks pass. Otherwise not shipped.

## Resumed validation and second candidate

The rebuilt ILP candidate is bit-identical on full logits, chunked cache and
generation at one and two threads. Six balanced pairs measured 159.7478 →
156.5862 ms prefill (1.979% lower) and 28.43845 → 28.54515 ms decode.
Three passes failed decode spread; the prefill ratio also narrowly missed
0.98. **Not qualified.** All samples and the candidate patch are retained in
`benchmark/results/attention_ilp_v1_windows.json` and
`benchmark/experiments/attention-ilp-v1.patch`.

Paired profiles isolate embedding at 3.59–3.90 → 0.11–0.12 ms and attention
at 18.39–19.43 → 17.17–17.42 ms. Instrumenting attention stages finds about
13.2 ms in exponentials/normalization, versus 2.9 ms in score calculation.
Only 66 probabilities are subnormal, all in one layer: this does not explain
the cost across all layers. MinGW disassembly shows `expf` calls double
`exp`, whose main path uses x87 and changes/restores its rounding control.

Next candidate, predeclared before measurement: vectorize four softmax
exponentials using double-precision range reduction and a degree-13 Taylor
polynomial on the reduced interval. Use scalar `std::exp` outside [-80, 0]
and for exceptional inputs, retaining underflow and NaN semantics. Preserve
score subtraction and sequential denominator accumulation. No global fast-math,
flush-to-zero, or artifact changes. Primitive gate: at most one FP32 ULP from
scalar exp on a broad deterministic sweep, plus exceptional/tail checks.
Trained-model gates remain FP32 reference allclose (atol/rtol 2e-3), >=99.9%
next-token agreement, perplexity ratio <=1.001, cached/generation parity.
Report bit identity when observed, but do not assume a polynomial is universally
bit-identical to every platform's libm. Whole-model timing gates stay unchanged.

## End-of-session handoff (local work, do not restart experiments)

User requested stopping after one final confirmation and resuming in the
morning. We agreed to keep work local tonight; do not push or merge this session.
Branch remains `diag/core-scaling`; last committed checkpoint is `249c353`.
The runtime/test/tool edits below are local and uncommitted. The final
confirmation outcome is recorded below when complete.

Completed for the softmax candidate:

- Normal build: `build/attn-softmax/candidate/leaf_decoder.exe`.
- Native attention test: 1,052,672 sampled exp arguments, maximum observed
  FP32 ULP error **0**. GQA, broadcast masks, mask polarity, key tails,
  vector output tails, scalar fallback and nonfinite cases pass.
- Trained GPT-2: bit-identical full logits and chunked logits against the
  shipped baseline at threads 1 and 2; greedy tokens match both baseline
  and saved PyTorch generation. Record: `build/attn-softmax/identity.json`.
  Exact equality is to Leaf's baseline, not to PyTorch floating-point logits.
- Eight reduced architecture cases, including scalar and quantized paths,
  pass exact paired regression: `build/attn-softmax/architectures.json`.
- Full suite: **831 passed, zero skipped, four existing ONNX warnings**,
  50.86 seconds. Log: `build/attn-softmax/pytest.log`.
- First six-pair softmax run: prefill 158.7654 → 144.0173 ms (**9.29% lower**),
  decode 27.98995 → 28.08685 ms (+0.35%). All six prefill pairs improve.
  Both pooled phases are stable, but candidate pair 3 fails decode spread.
  Strict gate **fails**; retain this run, do not replace it with confirmation.
  Record: `build/attn-softmax/acceptance.json`.

Final confirmation was declared before execution after the user confirmed
plugged-in/no-heavy-work conditions. It uses 101 samples rather than 31,
six balanced pairs, ten warmups, CPU 2, one thread and Above Normal priority.
No builds, correctness tests or PyTorch runs overlap its timing window.
Thresholds remain prefill <=0.98, decode <=1.02, and every pass plus both
pooled phases stable at p90/p10 <=1.25. Every sample is retained.

```powershell
python tools/benchmark_default_abba.py --before build/fp32-release/default/leaf_decoder.exe --after build/attn-softmax/candidate/leaf_decoder.exe --workdir build/gpt2_prefill_followup --output build/attn-softmax/confirmation-101.json --pairs 6 --runs 101 --warmup 10
```

Resume order:

1. Read this checkpoint, `git status`, and the final result. Preserve local
   changes and completed validation; do not repeat the old GEMM packing or
   96-row scheduling experiments.
2. Use the final outcome to decide qualification, retaining the earlier
   failure in any claim. If qualified, run the fresh balanced PyTorch eager/
   SDPA comparison on Windows with `--default-kernels`. No new stable PyTorch
   parity/speed claim has been established for this candidate yet.
3. Complete wheel build/install smoke checks for the changed runtime. The
   old installed wheel does not contain this candidate. Confirm primitive
   CI coverage on all three OSes after pushing; current local changes have
   not run on remote CI.
4. Update README results and the existing Mermaid diagram with the accepted
   execution path. Commit/push `diag/core-scaling`, review required CI, then
   merge into main as requested earlier, once the work is complete.

The diagnostic-only stage-counter override is preserved locally at
`build/attn-ilp/diagnostic-include/leaf/kernels/attention_vector.h`; outputs are
`build/attn-ilp/attention-stages.txt` and `probability-counts.txt`. It is not
compiled into the candidate. The original ILP-only patch and failed result
are retained under `benchmark/experiments/` and `benchmark/results/`.

## Final outcome — stop here tonight

The 101-sample confirmation PASSED the unchanged gate. All twelve passes and
both pooled phases are stable. Pooled medians across 606 samples per binary:

| Phase | Shipped baseline | Softmax candidate | Change |
|---|---:|---:|---:|
| Prefill | 161.1558 ms | 145.4869 ms | 9.72% lower |
| Decode | 28.64235 ms | 28.48115 ms | 0.56% lower |

Median paired ratios: prefill 0.90221171; decode 0.99025637.
Retain the earlier 31-sample run's failed individual decode spread alongside
this passing confirmation. This qualifies the local incremental native
comparison on this workload; it does not establish a fresh PyTorch speed win.

Durable records are in benchmark/results/attention_softmax_{initial_windows,
confirmation_windows,identity,architectures}.json. All code changes remain
local/uncommitted on diag/core-scaling; nothing was pushed or merged tonight.
No benchmark processes remain running. README/diagram updates, fresh PyTorch
comparison, changed-runtime wheel verification, remote CI and merge remain
for the morning. Do not repeat completed full tests or exact-parity checks
unless source changes warrant it.

## Resumed 2026-10-08

The user resumed work and confirmed plugged-in/no-heavy-work conditions.
The overnight pause is over. New wheel build/install and offline first/cached
generation checks pass, with the default optimized FP32 runtime active.
Installed/source full logits, chunked logits and generation are bit-identical
at one and two threads.

Fresh installed-wheel/PyTorch forward-reverse comparison (101 samples and ten
warmups per pass, CPU 2, one thread, Above Normal): Leaf 144.30735/28.11020 ms,
eager 153.51550/28.68440 ms, SDPA 151.66305/28.57335 ms (prefill/decode).
All three runtimes pass individual, pooled and between-pass stability.
The **4.85% prefill reduction versus the fastest measured PyTorch** qualifies;
decode is 1.62% lower, passing non-regression but not a separate 2% improvement
threshold. Evidence, exact hashes and scope are in
`benchmark/results/attention_softmax_validation.json` and the new report
`docs/windows-attention-softmax.md`. README and Mermaid now reflect the path.
Next delivery steps: commit/push the current branch, inspect CI, create/review
the PR and merge to main under the user's earlier instructions.
