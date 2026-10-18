# Shared FP32 accumulation accuracy — October 18, 2026

The isolated accumulation candidate passes the previously failing trained
SmolLM2 and Qwen2.5 fidelity checks without relaxing tolerances. Its complete
Smol timing comparison is stable but slower, so **it is not enabled in the
released runtime**. Qwen timing is still in progress at this checkpoint.

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
| Qwen2.5 | Pending | Pending | Running serially after Smol |

The before binary is the released FP32 default from PR #8. These are
incremental Leaf comparisons; no competitor win is claimed. Failed gates stay
recorded and are not relabeled as passes because accuracy improved.

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
