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

Conclusion: the MLP GEMM is FMA-throughput bound at this instruction count;
isolated MKL is not faster. Persistent output-channel weight packing keeps
the same 2-load/6-broadcast/12-FMA mix and would add ~216 MiB, so it was
**not implemented**.

## Profile of the shipped default (`LEAF_DECODER_PROFILE=1`, 25 forwards)

Per prefill forward: linear 145.6 ms, attention 19.1 ms, embedding 3.3 ms,
layer norm 2.8 ms, activation 2.2 ms. Attention is ~12 MFLOP/layer but costs
~1.6 ms/layer — far below GEMM throughput, so it is the better-supported
next target.

## Next action

Inspect multi-token AVX2 attention in `engine/src/kernels/transformer.cpp`
and its dispatch in `decoder.cpp`; implement one change, verify exactness /
tolerance, then whole-model ABBA.
