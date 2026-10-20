# October 20: source-preserving vocabulary storage

Branch: `codex/source-bf16`, based on merged main `073dac0` (PR #9).
The previous `codex/linear-accumulation` branch was merged and deleted yesterday.
This checkpoint concerns new work; nothing here changes the default accumulation
policy or reclassifies the earlier Ollama probability-gate failures.

## Implemented

- Artifact version 5 retains original BF16 vocabulary weights and tied embedding
  payloads. FP32 source checkpoints keep their existing artifact bytes. There is
  no new weight rounding and computation stays FP32.
- Direct AVX2 paired GEMV and scalar/tail paths preserve their FP32 arithmetic;
  multi-token evaluation expands bounded row blocks for the existing GEMM.
- `leaf run --source-bf16-head` uses a separate compact cache, validates the runtime
  before weight downloads and rejects incompatible integer computation requests.
- Native storage counters, explicit zero/single/multiple EOS metadata and malformed
  artifact rejection are covered. The source package includes the new header.

## Checks completed

- **1,125 Python unit tests pass**, including real native scalar/vector, one/two
  thread, width/tile tails, cache, generation, sliding-window and CLI cases.
- Native exhaustive BF16 expansion plus GEMV/guard checks pass.
- Ten reduced architecture configurations pass with the ordinary Windows binary.
- All 290 tensors/model are audited against existing FP32 artifacts. Compact
  artifacts: Smol 1,447,312,000 -> 1,352,940,160 bytes; Qwen 1,976,158,464 ->
  1,703,889,152 bytes. This is artifact size, not measured resident memory.
- Eight trained full-logit blocks and six reference continuations/model pass exact
  rebinding to the prior complete accurate K=256 quality qualification. Trained
  scalar, chunked-cache and two-thread storage controls also pass.
- A Windows wheel builds; its isolated installed `doctor` and offline generation
  checks pass without a compiler or decoder override, or Torch/Transformers imports.

## Interrupted timing and fix

`build/source-bf16-20261020/acceptance.json` is complete but **unqualified**.
Windows logged Modern Standby entry at 14:49:15 and exit at 14:54:45 IST.
The entire attempt is invalid for release, including its apparently passing Smol
summary. Keep all samples. The original runner is preserved as
`integration-first-attempt.py`; `first-attempt-sleep-audit.json` records the events.

The runner now uses a scoped system/display-awake request, pass timestamps and a
fail-closed Windows sleep-event audit. The normal runtime does not keep the display
awake or change permanent power settings. A fresh full repeat writes
`build/source-bf16-20261020/acceptance-awake.json`. Its predeclared gate is decode
paired ratio <=0.98, prefill <=1.02, and every pass/pooled p90/p10 <=1.25. No samples
are removed. Both policies use AC monitoring; the new binary checks each native
sample as well. Timing is one thread on CPU 2, six balanced pairs/model, 31 samples
and ten warmups/pass. The precise accumulation policy is experimental on both sides.

## Fresh controlled gate: pass

`acceptance-awake.json` is complete and qualified for the tested configuration.
No Windows sleep events or AC failures occurred. Every individual and pooled
stability check passes, with no removed samples.

| Model | Prefill before / after | Decode before / after | Paired decode reduction |
|---|---:|---:|---:|
| SmolLM2 | 493.38740 / 491.26880 ms | 80.24445 / 76.20240 ms | 5.12% |
| Qwen2.5 | 568.44785 / 557.07875 ms | 110.50400 / 97.50815 ms | 11.65% |

These are incremental Leaf results on the accurate accumulation candidate, not
new competitor comparisons or measurements of the installed default policy.
The storage option is available through the normal CLI without changing that
default arithmetic. The accurate accumulation candidate is still experimental.

The conservative build compiles and passes 97 focused native-backed storage and
generation tests. The include-only portability fix produces **identical Windows
preprocessed source**, bound to the original build's source hash. The final wheel
was rebuilt from the corrected source and passes isolated installed startup and
fresh offline artifact preparation/generation without binary/compiler setup.

## Local resume locations

- `build/source-bf16-20261020/prepared.json`, `quality.json`, `screen.json`.
- Candidate `accurate-final/leaf_decoder.exe`: SHA256
  `d6f1ac75f27a156b429fc8030943a0f6c74bbac1b20aed87bf928c0493a51140`.
- Before binary: `build/linear-accumulation-20261019-pipeline/leaf_decoder.exe`,
  SHA256 `af65268c755564834ae9c1d25d591d1480f4120c04ea8a4d1a8138110dcca7be`.
- Ordinary runtime and local wheel are under `default/`, `wheels/`, `installed/`.
- `final-unit-results.xml`, `architectures.json`, `installed-check.json` retain checks.

Do not overwrite these records or rerun the old failed timing as a fresh attempt.
Compact evidence, including source provenance and the rejected standby-overlap
attempt, is retained under
[`source-bf16/results/20261020`](../benchmark/experiments/source-bf16/results/20261020).
Large weights, binaries and logits remain under ignored `build/`. No benchmark
or local model server remains running after the completed validation.

Next performance target remains the shared MLP projection cost (about 67% of Smol
and 77% of Qwen prefill in the retained profile). The vocabulary-storage change
targets decode traffic; it does not solve that compute bottleneck or establish a
universal runtime advantage.
