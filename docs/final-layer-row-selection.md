# Final-layer FP32 row selection

For last-token-logit requests, Leaf now skips earlier final-layer feed-forward
rows after appending every input token's K/V entries. This is a shared causal
decoder optimization, selected by precision, requested output and sequence
length rather than model name. Full-logit requests, single-token decode and
quantized/mixed artifacts keep their existing execution.

The retained suffix contains 16–31 rows and starts on an original 16-token
panel boundary. Keeping at least one full panel preserves the existing packed
GEMM and activation dispatch. Selecting only one row would instead switch to
GEMV and could change reduction order. Prefixes shorter than 32 tokens keep
their original path.

Every layer's attention and K/V append still run for all input tokens. Earlier
final-layer MLP results have no consumer when only the last output is requested;
future tokens use the stored keys/values, which precede that MLP. The final
normalization operates on the retained suffix and the output projection reads
its final row. This does not prune model weights or alter activation precision.

## Initial MLP release — October 14, 2026

### Correctness

The paired native regression covers eleven reduced configurations: GPT-2,
GPT-NeoX, OPT, Llama, Qwen2 and Mistral, including odd dimensions, bias/tied-weight
variants, ReLU and an eight-token sliding window. It checks lengths around
8/16/32/48/64-token boundaries, last/full logits, resets, single-token decode,
multi-token cache appends and a full-logit call after an optimized append.

All **121 precision/thread/scalar policies**, totaling **10,285 forward calls
per binary**, produce byte-identical before/after logit files. INT8/INT4 and
W8A8 are negative controls. A separate full-prefix FP32 PyTorch oracle passes
at the row-selection boundaries in all eleven configurations; that check is
included in Windows/Linux/macOS CI.

Trained GPT-2, Pythia-70M and Pythia-160M also produce byte-identical outputs
for the tested boundary/cache sequences at one/two threads and scalar execution.
Saved trained inputs additionally check full logits and eight generated tokens
against the before binary. Existing larger Pythia reference-tolerance failures
remain open; this change preserves their outputs rather than resolving them.

An operation-count profile of Pythia-70M at 63-token prefill confirms activation
rows decrease from **378 to 346** and FP32 linear rows from **2269 to 2205**.
Decode activation rows stay at 6. This profile is separate from acceptance timing.

### Windows incremental timing

Both 63-token FP32 prefill experiments pass the incremental gate:

| Model | Prefill before → after | Reduction | Decode before → after |
|---|---:|---:|---:|
| Pythia-70M | 56.37785 → 53.44780 ms | **5.20%** | 10.2687 → 10.2673 ms |
| GPT-2 | 144.05315 → 140.40020 ms | **2.54%** | 28.22635 → 28.47850 ms |

Median paired prefill ratios are 0.94995 and 0.97445, respectively. Pythia decode
is effectively unchanged; GPT-2 decode is 0.89% higher, within non-regression.

Both binaries use the same FP32 artifact and token IDs, one thread on logical
CPU 2 and Above Normal child priority. Six alternating AB/BA pairs each use ten
warmups and 31 measured samples: 186 samples per binary per phase. Every pass
and both pooled sample sets pass p90/p10 ≤1.25. The existing predeclared gate
requires prefill ratio ≤0.98 and decode ratio ≤1.02. No samples are removed.

These are improvements over the previous Leaf binary on the tested Windows
workloads. They are not fresh PyTorch or equal-quality llama.cpp speed comparisons.
The earlier [llama.cpp diagnostic](pythia-runtime-comparison.md) has different
timing policy and fidelity limitations; do not combine its medians with this run.

Evidence:
[Pythia timing](../benchmark/results/final-layer-rows/pythia70-abba.json),
[GPT-2 timing](../benchmark/results/final-layer-rows/gpt2-abba.json),
[paired reduced regression](../benchmark/results/final-layer-rows/reduced-parity.json),
[independent reduced reference](../benchmark/results/final-layer-rows/reference-parity.json),
[trained regression and operation counts](../benchmark/results/final-layer-rows/trained-parity.json).

The actual built wheel passes first preparation and cached execution without
compiler paths or Torch during inference. Long-prefix generation matches the
before binary, and its operation profile confirms the optimized path is present.
[Installed package smoke](../benchmark/results/final-layer-rows/package-runtime.json),
[installed dispatch check](../benchmark/results/final-layer-rows/installed-final-rows.json).

## Query and attention extension — October 15, 2026

**Unshipped candidate.** The [patch](../benchmark/experiments/final-attention-rows.patch)
and all results are retained. No Git push or release was made. The default
decoder retains the accepted MLP-only selection described above.

The extension moves the same row selection ahead of the final query projection,
query RoPE/reorder, attention and attention output projection. K/V projections,
K RoPE, K/V append and the final input normalization still cover every input row.
Single-token decode, full-logit requests and quantized/mixed paths retain their
previous work. The retained suffix starts on the original GEMM panel boundary.

The paired regression again passes all **121 policies / 10,285 forward calls per
binary**, with byte-identical output files. The same run also checks the FP32
results against independent full-prefix PyTorch references. All **15 trained
checks** pass for GPT-2, Pythia-70M and Pythia-160M, including one/two threads,
scalar execution, cached continuation, full logits and generation. These checks
do not resolve the existing larger Pythia strict-reference failures.

The separate Pythia-70M 63-token operation profile confirms this extension changes
attention query rows **378 → 346**, FP32 linear rows **2205 → 2141**, and RoPE
rows **819 → 787**. K/V append rows stay **378**, activation rows stay **346**,
and decode row counts are unchanged. Counts sum across layers and projections;
they are not percentages of total elapsed time.

The built wheel passes first preparation and cached execution with compiler
paths removed and no Torch during inference. Its 128-token generation matches
the shipped baseline exactly, and the installed operation profile confirms
the query/attention selection is present.

Acceptance timing uses the same gate, CPU, priority, pairing and sample counts
as the initial release. The predeclared primary workloads are **127-token prefill
+ one cached decode token** for Pythia-70M and GPT-2; the earlier **63-token**
workloads are secondary checks. All raw samples and any failures are retained.

| Workload | Prefill before → candidate | Observed reduction | Incremental gate |
|---|---:|---:|---|
| Pythia-70M, 127 tokens | 103.02450 → 99.42025 ms | 3.50% | Failed per-pass stability |
| GPT-2, 127 tokens | 276.21700 → 269.15230 ms | 2.56% | Failed per-pass stability |
| Pythia-70M, 63 tokens | 60.80025 → 60.09575 ms | 1.16% | Failed stability and 2% prefill threshold |
| GPT-2, 63 tokens | 139.29445 → 138.04605 ms | 0.90% | Failed per-pass stability and 2% prefill threshold |

All four pooled decode ratios are below 1.02. Both 127-token pooled prefill
ratios pass 0.98, but each workload has unstable individual passes. Pythia's
63-token pooled decode also fails stability. These observations do not qualify
a new model speedup. In Pythia's fourth 127-token pair both binaries show much
faster low-percentile prefill samples (62.82/66.53 ms) alongside ordinary samples
around 100 ms; the data do not identify the cause. No outliers were removed and
no unchanged acceptance run was repeated.

[Complete timing, paired/reference checks, profiles and installed-wheel evidence](../benchmark/results/final-attention-rows).
Next: investigate shared exact-erf GELU, which remains scalar and accounts for
about 14% of the separate Pythia profile. This is an operator-level opportunity;
the existing vector GELU-tanh is a different activation and cannot substitute for it.

## Reproduction

`tests/native/test_decoder_final_rows.cpp` links against the actual decoder and
emits every test call's logits while checking cache lengths and output shapes.
Compile the same driver against the before and after decoder revisions, then run:

```powershell
python tools/verify_final_layer_rows.py --before BEFORE_DRIVER --after AFTER_DRIVER --output build/final-rows-paired
python tools/verify_final_layer_rows.py --after AFTER_DRIVER --output build/final-rows-reference
python tools/benchmark_default_abba.py --before BEFORE_DECODER --after AFTER_DECODER --workdir MODEL_FIXTURES --output build/final-rows-timing.json --cpu 2 --pairs 6 --runs 31 --warmup 10
```

The CMake target is `leaf_decoder_final_rows_tests`. The CI mode requires no
before binary and uses independent PyTorch full-prefix references. Timing needs
`decoder-32.leaf` and `tokens.json` with `benchmark_ids` in `MODEL_FIXTURES`, a
quiet plugged-in Windows host and the appropriate physical performance core.
Use fresh output paths. No model files are committed to Git.
