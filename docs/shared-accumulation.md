# Shared FP32 accumulation accuracy — October 18, 2026

## Native accuracy option — October 26, 2026

The qualified K=256 balanced reduction is now a shared native kernel selected
by `--fp32-math accurate`, rather than a source-patched build. The existing
`fast` policy remains the default; the older timing failures below are retained.
This integration establishes output fidelity, not a new latency gain.

```sh
leaf run HuggingFaceTB/SmolLM2-360M-Instruct --bits 32 --source-bf16-head --fp32-math accurate --prompt "Explain CPU inference."
leaf evaluate MODEL_ID --fp32-math accurate --dataset heldout.jsonl --dataset-source "name/test split" --blocks 8 --output build/accurate-quality
```

The accurate policy shortens each AVX2 projection sum to 256 columns and merges
partials with a balanced FP32 tree. RMSNorm keeps FP32 squares and scale
semantics but sums the squares in FP64. Forced scalar execution uses FP64 dot
reductions. Explicit unchanged-input promises reuse packing within Q/K/V and
MLP up/gate groups. Model names do not choose this policy. Quantized requests
are rejected; original BF16 head/tied-embedding storage remains compatible.
`--bits auto` cannot reuse quantized speed profiles under this math policy.

Final Windows validation rebinds **all sixteen complete trained logit blocks**
and **twelve reference continuations** to the earlier qualified accurate build.
All match exactly. Ten scalar/cache/thread/storage controls also match, and
eight trained fast-policy regressions preserve the preceding ordinary outputs.
This uses retained PyTorch qualification; it is not a fresh independent oracle
run or a claim of task/reasoning accuracy. Original tolerances are unchanged.

The native kernel passes **43,456 output checks**, including exact AVX agreement
with the historical K=256 kernel, an independent FP64 oracle, portable fallback
and output padding checks. **1,204 unit tests pass** (two optional cases skip),
and ten reduced architecture configurations retain bit-exact default outputs.
An isolated installed wheel runs accurate math with compact BF16 storage and
two assigned workers, offline, with compiler paths and binary overrides removed
and without importing Torch/Transformers. Its `.text`, `.rdata` and `.data`
sections match the fully qualified native binary. Historical accumulation builds
remain reconstructible from the pre-integration engine revision `a78fa51`.

[Checkpoint and reproduction](checkpoint-2026-10-26-fp32-math.md) ·
[Retained migration evidence](../benchmark/validation/fp32-math/20261026/manifest.json)

The records below describe the earlier isolated experiments. Their performance
verdicts remain unchanged. Floating-point summation order can change results
even for mathematically equivalent operations; see
[PyTorch's numerical-accuracy guidance](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).

## Original isolated experiment

The isolated accumulation candidate passes the previously failing trained
SmolLM2 and Qwen2.5 fidelity checks without relaxing tolerances. Its complete
Smol timing comparison is stable but slower, so **it is not enabled in the
released runtime**. Both complete timing comparisons are stable and fail the
prefill improvement gate.

## What changed

The candidate keeps the six-row × sixteen-token AVX2 tile. It restarts FP32
FMA sums every 128 columns, combines partials in FP64, then rounds once to
FP32 and adds bias. Pure-FP32 RMSNorm also sums FP32 squares in FP64, retaining
the existing FP32 scale calculation. The scalar control exposed another long
FP32 reduction; pure-FP32 scalar projections now use FP64 dot sums. Quantized
and mixed-precision execution retain their prior arithmetic.

This is a shared operator/workload policy, not a model-ID branch. The compiled
candidate is marked `experimental_accumulation_build`, and production source
dispatch remains unchanged. Source, builder and inspection tools live in
[`benchmark/experiments/linear-accumulation`](../benchmark/experiments/linear-accumulation).

## Verified accuracy

Same pinned snapshots, WikiText-2 test prefix, eight independent 128-token
blocks per model, 1,016 scored next-token targets each. All logits must satisfy
`abs(error) <= 2e-4 + 2e-4*abs(reference)`; agreement >=99.9%, perplexity ratio
<=1.001, and all six raw greedy continuations must match the PyTorch oracle.
These measure runtime fidelity, not factual or conversational task accuracy.

| Model | Released-runtime logit violations | Candidate violations | Agreement | Perplexity ratio |
|---|---:|---:|---:|---:|
| SmolLM2-360M-Instruct | 154 | **0** | 100% | 1.0000000584 |
| Qwen2.5-0.5B-Instruct | 1,000 | **0** | 100% | 1.0000001423 |

All twelve generated continuations match exactly. Maximum absolute logit
errors are 0.0001745224 and 0.0001931190 respectively. The final candidate's
normal AVX path is bound to these results by comparing **every block**
byte-for-byte with the fully evaluated candidate and rerunning every
continuation against saved reference tokens. The records identify this
rebinding explicitly; a two-block result was not used as a complete gate.

The final candidate additionally passes:

- Twelve retained-reference/exclusion cases: full/chunked prefixes, tokenwise
  execution on two threads, forced scalar, trained GPT-2/TinyLlama prefixes,
  and exact W8A8 tokenwise/generation exclusions.
- Ten reduced decoder architecture configurations, with cache, generation,
  scalar, thread and precision-dispatch checks.
- **1,048 unit tests**, including guards against timing incomplete quality,
  relaxed tolerances, changed binary/weights, mismatched prefixes or incomplete
  quality rebinding.

The scalar controls are retained: the released path has 147/481 violations
on the tested Smol/Qwen prefixes; accurate RMS alone leaves 121/3. The scalar
dot correction brings both to zero. Earlier trained-model checks outside
Smol/Qwen cover prefixes, not a new full-corpus qualification.

## Controlled performance

Windows, one thread pinned to logical CPU 2, batch one, warmed 63-token
prefill plus one cached decode token and last-token logits. Six balanced
AB/BA pairs, 31 measured samples and ten warmups per pass. The unchanged
gate requires prefill ratio <=0.98, decode <=1.02, and all individual/pooled
stability checks. Loading, tokenization and command startup are excluded.

| Model | Before / after prefill | Before / after decode | Outcome |
|---|---:|---:|---|
| SmolLM2 | 492.9177 / 514.36175 ms | 80.5039 / 80.10565 ms | **4.35% higher prefill; gate fails**, all twelve passes stable |
| Qwen2.5 | 565.62355 / 589.518 ms | 110.15875 / 109.54915 ms | **4.22% higher prefill; gate fails**, all twelve passes stable |

The before binary is the released FP32 default from PR #8. These are
incremental Leaf comparisons; no competitor win is claimed. Failed gates stay
recorded and are not relabeled as passes because accuracy improved.

The median paired prefill ratios are 1.04620 and 1.04271. Decode passes
non-regression; its sub-1% reductions are not claimed as improvements.
[Raw reports and validation manifest](../benchmark/results/shared-accumulation-20261018/validation.json)
retain all samples, exact binary/input identities, full quality, scalar
controls, earlier failed checks and the interrupted no-sample launch. No
quality or stability threshold was changed to obtain these results.

## What the bounded experiments ruled out

The original partial helper repeatedly saves/restores nine SIMD registers
and executes `vzeroupper`. Inlining removes those boundaries and preserves
the retained logits, but its short micro probe does not show a speed win.
A second variant reuses each 8-KiB K chunk across the existing 96-row block,
with bounded 12-KiB FP64 totals. It passes row/K boundary and exact traversal
checks, but also has no demonstrated micro speed win. Both variants are
retained without changing the selected kernel or the acceptance thresholds.
Inspection finds no stack-vector accesses inside the identified six-row
FMA loops; eliminating calls alone did not remove the measured cost.

Next linear work should target the partial-reduction and output-conversion
overhead while preserving this candidate as a numerical reference. Any new
candidate must retain these strict checks and pass full-model timing before
becoming default. Larger contexts, independent CPUs and native Linux timing
remain outside this evaluation. Dataset inference throughput and individual
network benchmarks follow a deliberate runtime release decision; passing
these checks does not establish faster dataset I/O, augmentation or training.
