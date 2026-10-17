# Release checkpoint — October 15–17, 2026

The latest user instruction authorizes a checkpoint push to main now, before
the long Ollama rerun, because the session limit is near. Base was `8f279a3`.
Do not discard the user's untracked root PNG.

## October 17 continuation

- The default release passes all ten reduced configurations, including FP32
  and W8A8 SiLU dispatch, custom plans, scalar/INT4 exclusions, cache and
  generation checks. The validator now reads exported weight precision rather
  than inferring it from a test label; the earlier custom-plan failure was
  a validator error. Saved evidence:
  `benchmark/results/fp32-silu-followup/release-architectures.json`.
- Today's focused Python checks: **86 passed**. The complete **989-test**
  suite and strict incremental timing below already passed; do not repeat
  those long runs without a relevant source change.
- Fresh wheel validation passes: offline GPT-2 preparation and cached reuse
  match the reference, with no compiler on PATH or PyTorch loaded. Installed
  TinyLlama SiLU dispatch passes and all 512,000 checked logits exactly match
  the accepted release. All 19 source-bundle files and both runtimes are present.
- Fresh TinyLlama/Ollama comparison is COMPLETE: Leaf/Ollama prefill
  1391.4163/1530.6415 ms and cached forward 222.90765/251.448 ms. All twelve
  passes and pooled distributions are stable; reductions 9.10% and 11.35%.
  The unchanged Unicode logprob miss (0.0002179475 > 0.0002) keeps the
  overall qualification false. Do not repeat this run or relax the limit.
- Both SmolLM2-360M and Qwen2.5-0.5B-Instruct are downloaded via Leaf. General
  multi-EOS support preserves Qwen's two declared stop tokens, adds version 4
  artifacts, and retains versions 2/3. Final full unit suite: **1,029 passed**,
  one pytest cache-permission warning. A failed system-temp fixture attempt
  was stopped; the fresh workspace `--basetemp` run is the completed result.
- Both models complete eight length-128 quality blocks (1,016 targets), with
  100% next-token agreement and six exact continuations. Strict all-logit
  gates fail (154/1,000 violations), so neither is fully fidelity-qualified.
  Current-source installed-wheel GPT-2 and Qwen Hub-ID checks pass without
  compiler paths, binary overrides or heavy model frameworks.
- Publication uses [PR #7](https://github.com/Nbit-51/Leaf/pull/7), following
  the protected-main workflow. Commit `0b44e71` contains the earlier release;
  the continuation adds the completed comparison and multi-EOS work. Merge
  only after checks on the final head pass, then delete its temporary branch.
- Next: diagnose shared remaining numerical differences, then audited matched
  comparisons for the two new models. Do not restart rejected attention/GELU
  experiments or silently reuse old binaries for version-4 artifacts.

## Current release checkpoint — supersedes pending notes below

- Mains returned. The strict FP32 SiLU native gate COMPLETE/PASSED: prefill
  1897.61205 → 1391.1965 ms (26.68699% lower), decode 226.0822 → 225.3048 ms.
  All twelve passes and pooled distributions stable, thresholds unchanged.
  `build/fp32-silu-20261015/timing-32.json` is retained; **do not repeat it**.
- Integrated Windows pure-FP32 SiLU default using `LEAF_OPTIMIZED_SILU_FP32`.
  Conservative option available. Other precisions/OSes/activations retain
  their policies; scalar and one-token calls stay excluded.
- Release binary `build/fp32-silu-20261015/release/leaf_decoder.exe` exactly
  matches the accepted candidate on trained full logits (one 128-token
  sequence), chunk16, tokenwise, generation and forced scalar controls.
  This binds the release to the candidate's complete 1,016-target quality
  record; no release latency is silently inferred from these parity checks.
- Full Python suite: **989 passed**, four existing ONNX deprecation warnings.
  Source and Mermaid diagram updated. Experimental GELU remains off.
- Fresh power-aware generation release adapter:
  `build/fp32-silu-20261015/leaf-generation-release.exe`.
  The default comparison with Ollama completed on October 17, in a new output directory,
  no inherited profile/experiment env vars, AC on, no concurrent heavy work.
  No old preflight was reused: adapter/helper hashes changed. Keep the 2e-4
  selected-logprob check and explicit diagnostic timing if it still fails.
- The interrupted power-mixed Ollama run is invalidated; original diagnostic-b
  describes the older scalar FP32 default, not the newly shipped default.
- Historical Windows spread is not completely solved. Dispatch counters
  capture interruption-related spikes; do not claim the OS culprit is known.

## Latest continuation: acceptance spread and FP32 SiLU

- Added version-4 thread dispatch counters and `tools/diagnose_acceptance_spread.py`.
  Six-pair old-launch replay reproduces failures on CPU 2. Representative
  decode 16.6991 ms versus 10.5587 ms median, cycles only +4.47%, 54 switches
  versus median 12, zero faults. Some other spikes have elevated cycle counts.
  Scheduling interference is supported; the driver/process and every older
  cause remain unidentified. Do not claim all timing spread is fixed.
- Balanced identical-binary CPU 2/14 control passes both cores. It does not
  prove core switching cures the problem. Plain A/A then fails pair 3 BEFORE
  at prefill 1.31855 / decode 1.36044 p90/p10. All results retained under
  `benchmark/results/acceptance-spread-20261015/` and local build counterpart.
- WPR cannot enable profiling in this non-admin session; no recording active.
  User initially offered to run the prepared `scripts/trace_acceptance_spread.ps1`,
  then redirected to the compute gap; no admin trace was produced. Do not wait
  indefinitely for one or start it concurrently with timing.
- Fresh Ollama run `build/ollama-20261015/diagnostic-c/` stopped on reported
  power outage. OS power-source event: 16:18:17 IST. Own native/server processes
  cleaned up; partial data explicitly invalidated. Never report its mixed-power
  medians as a new comparison. Original completed diagnostic-b remains intact.
- Fresh pinned battery-only TinyLlama profile finds default activation 27.36%
  of prefill. Existing FP32 vector SiLU candidate reduces its profile share to
  1.43%, with 22 prefill gate calls and zero cached-decode gate calls; generated
  tokens and selected probabilities exactly match baseline on that diagnostic.
  No qualified speedup is claimed from battery profiling.
- `build/fp32-silu-20261015/leaf_decoder.exe` is freshly built with the existing
  `LEAF_EXPERIMENTAL_SILU_GATE=1` route; normal FP32 matrix/attention stack stays
  enabled. `trained-quality.json` is COMPLETE/PASSED for FP32: all 1,016 targets,
  100% agreement, perplexity ratio 0.9999998867, chunk16 and tokenwise cache,
  reference generation. `tools/verify_cached_decoder.py --bits 32` now avoids
  repeating unrelated W8A8 checks. Source-reference limits unchanged; separate
  Ollama probability miss is not resolved by these checks.
- Power-aware native generation adapters built as `leaf-generation-before.exe`
  and `leaf-generation-candidate.exe` in that directory. Runtime-info capability
  and per-sample AC marker are required by the updated Ollama runner. Actual
  battery guard refusal verified. Quality-only runs remain possible on battery.
  Parent/server cleanup now covers failures applying initial affinity/priority.
- **374 focused Python tests pass**, and rerun SiLU primitive test checks
  4,907,400 values, max ULP zero, AVX2 exercised. No timing runs after outage.
- Historical next command was in the [FP32 SiLU follow-up](silu-gate-experiment.md#fp32-follow-up--october-15-2026):
  plugged-in native before/after gate, then fresh candidate/Ollama comparison,
  then release integration only if qualified, final checks/docs and push main.
  Those native acceptance and integration steps are now complete. Do not
  restart failed attention or GELU experiments during release preparation.

## Completed; do not repeat

- Final-layer attention row experiment: byte-identical reduced/trained outputs
  and installed-package checks, but all four model timing runs fail stability.
  Decoder change restored; patch retained in
  `benchmark/experiments/final-attention-rows.patch`. See
  [report](final-layer-row-selection.md) and
  `benchmark/results/final-attention-rows/`.
- Exact-erf GELU AVX2 candidate: native 2,810,330-value parity checks, 132
  reduced policies / 11,220 calls per binary and eleven trained checks pass.
  Four synthetic distributions pass the 5% microbenchmark gain criterion;
  the wide uniform case does not. Pythia-70M/160M whole-model timing runs fail
  stability despite lower observed prefill medians. **Off by default**, enabled
  only with `LEAF_EXPERIMENTAL_GELU_ERF`. See
  [report](gemm-activation-followup.md#exact-erf-gelu-experiment--october-15-2026),
  `benchmark/results/gelu-erf/`, local `build/gelu-erf-20261015/`.
- TinyLlama FP32 / Ollama v0.40.4: six balanced pairs, all timing spread checks
  pass. Leaf 1,872.24 / 218.85 ms versus Ollama 1,516.17 / 251.99 ms prefill /
  one cached forward. All generated text matches, but one selected-logprob
  error 0.0002179475 exceeds 0.0002. **No overall qualified win**. See
  [report](tinyllama-ollama-comparison.md),
  `benchmark/results/tinyllama-ollama-fp32/`. No need to repeat import/conversion
  or this comparison. Ollama server/worker were cleaned up.
- Timing diagnostic startup race reproduced with actual CPU counters: old
  launch inherited monitor E-core 27 before pinning to P-core 2. Measured
  prefill/decode on CPU 27 took 135.24 / 30.08 ms, versus about 52 / 10 ms on
  CPU 2; next prefill migrated 27 to 2. The corrected runner inherits CPU 2
  at creation and gates execution until monitoring/policies are ready.
  All four corrected passes are stable, all 976 CPU boundary records match.
  Version-3 diagnostic build matches 3,219,456 logits byte for byte and its
  start gate passes integration checks. See
  `benchmark/results/timing-spread-20261015/startup-affinity-check.json` for
  88 passing focused tests, unchanged release runtime info, release ignoring
  the diagnostic gate, byte-identical release logits and rejection of the
  contaminated replay by the corrected validator. See also the
  [diagnosis](windows-timing-diagnosis.md#october-15-2026-diagnostic-startup-race).
  This defect is in the monitored diagnostic runner; the acceptance runner
  already pins before spawning, so older failed gates remain unresolved.
- Modern Standby 15:34–15:42 IST occurred after the Ollama comparison and
  preceding Pythia diagnostic. Neither overlapped sleep. Do not blame those
  samples or older awake runs on the user's grocery trip.

## Local assets

- Diagnostic source/helper changes: `engine/src/main_decoder.cpp`,
  `tests/native/decoder_diagnostic.h`, `tools/diagnose_windows_benchmark.py`,
  `tools/summarize_windows_diagnostics.py`, associated unit tests. Ordinary
  release builds have no diagnostic gate/counters.
- `build/timing-spread-20261015/`: old/new binaries, raw traces,
  `legacy-post-spawn.py` (saved reproducer, not a production runner), power
  events, `verify-v3.py`, parity records. Published summaries omit process PIDs
  and names; preserve local originals.
- `build/ollama-20261015/runtime/ollama.exe`, private `models/`, audited
  `tinyllama-f32.gguf`, `leaf-generation-benchmark.exe`, full local comparison
  directories. Do not commit binaries, model weights or user process lists.
- TinyLlama source snapshot:
  `C:/Users/navaneeth/Documents/Vs code projketcs/Imporved_Hydra/Hydra_Engine/TinyLlama-1.1B-Chat-v1.0-git`.
  Leaf FP32 artifact: `build/tinyllama_windows_fair/decoder-32.leaf`.
- Pythia-70M artifact:
  `build/manual-model-cache/models/0ba2eff1d3343234ba895c0d/decoder-32.leaf`;
  tokens: `build/gelu-erf-20261015/pythia70/tokens.json`.

## Next actions

1. Use the corrected diagnostic build/runner; never qualify instrumented
   results as release timing. No more HighQoS retries are justified by the
   current evidence. Do not change power plans, Defender settings or WSL.
2. If the separate uninstrumented acceptance runner fails again, capture
   scheduler/power evidence around that failure rather than repeatedly
   rerunning for a pass. Verify affinity before model work and record actual
   CPUs where possible. The historical root causes are not fully established.
3. Investigate TinyLlama's small selected-logprob discrepancy without changing
   the declared tolerance, then profile its slower prefill on the same matched
   workload. Prefer shared operators and test reduced/trained regressions.
4. Do not claim every model is supported or faster. Candidate next models are
   Qwen2.5-0.5B-Instruct and SmolLM2-360M-Instruct; no download started. Dataset
   work means embedding/classification job throughput at matched quality,
   not faster data loading or training. Keep voice work out of scope for now.

The user prefers bounded, decisive tests and durable checkpoints; they may run
commands themselves to save usage. Prior confirmation was plugged in with no
heavy work. Do not repeatedly ask the same readiness question.
