# October 22: explicit worker placement

Continue on `codex/mlp-schedule`, starting from `c88595c`. Dataset work is deferred.
Keep the pre-existing `docs/embedding.md` edit and unrelated image unstaged.

## Implemented

- `worker_affinity.h`: validated Windows/Linux thread scopes, allowed-mask checks,
  nested binding reuse and caller restoration. No process mask/power setting changes.
- `Workers`: optional CPU list, verified startup, owned-worker lifetime binding,
  exception propagation and safe reuse after failed tasks.
- Decoder/CLI: optional `--worker-cpus` list, native capability and metrics;
  incompatible older overrides fail before weight downloads. Existing auto-speed
  profiles cannot select quantization for a different placement policy.
- Default placement and model arithmetic are unchanged. Earlier unqualified
  worker-grain code stays isolated. Its builder now reads historical engine
  revision `880139d`, preserving reproduction after current engine changes.

## Completed correctness

- Final ordinary runtime passes **1,151 Python unit tests**.
- Native worker test: **130** range, participation, binding/restoration, nested
  scope, task-exception/reuse and invalid-policy checks.
- All **16 full trained logit blocks**, **12 reference continuations** and
  **10 scalar/cache/thread/default controls** match the prior qualified accurate
  build exactly. Strict tolerance thresholds are unchanged.
- Six ordinary FP32 trained regression cases match the preceding default exactly;
  this is not new strict PyTorch qualification.
- Wheel builds; isolated installed startup and offline source-BF16 generation
  with two explicit worker CPUs pass without compiler/decoder overrides or
  Torch/Transformers imports.
- Historical GEMM-schedule reconstruction builds and passes its 240 native cases.

## Timing in progress

The placement stage is now complete: assigned workers pass every individual
and pooled spread check on both models. The unassigned side fails several
spread checks, so the joint percentage-speed gate remains failed. Pooled
assigned p90/p10: Smol prefill 1.04648 / decode 1.14235; Qwen prefill 1.04018 /
decode 1.10580. All samples remain retained. This is a repeatability result,
not a qualified claim that placement makes every workload faster.

The one-core default non-regression stage is active. After it finishes, compare
the isolated GEMM job-size candidate with the same assigned workers on both sides.

`benchmark/experiments/worker-placement/validate.py` owns the predeclared gates.
Outputs are under `build/worker-placement-20261022/model/`:

- `quality.json`: complete/passing, bound to the new accurate binary.
- `placement.json`: same new binary, four allowed P-cores 2/4/6/8, off/on ABBA.
- `default.json`: old/new unbound, one P-core 2; run after placement finishes.

Each stage uses six balanced pairs/model, 31 samples and ten warmups/pass. No
filtering. All individual and pooled p90/p10 <=1.25. Placement requires paired
and pooled prefill <=0.98, decode <=1.02 for a speed claim. Default requires both
phases <=1.02 for non-regression. Every sample has native AC checks; parent AC
monitoring, scoped display/system-awake requests and sleep audits are required.
No builds, conversions or other tests may overlap either timing stage.

Do not treat an ON-only stability pass as a joint performance pass. If the OFF
stage is unstable, report that separately and do not claim a percentage speedup.
Placement remains optional regardless of the result. Read the complete JSON
verdict before publication; preserve partial/failed attempts rather than overwriting.

## Local resume paths

- `build/worker-placement-20261022/accurate-final/leaf_decoder.exe` and `build.json`.
- `default-runtime.txt`: final ordinary binary path.
- `units-final.xml`, `ordinary-parity.json`, `worker_affinity_test_final.exe`.
- `wheels/`, `installed/`, `installed-check.json`, `verify_installed.py`.
- `historical-schedule/`: historical private rebuild; no current engine edits.

Finish timing, retain compact evidence, update the usage/README/Mermaid and this
checkpoint, then push through a PR. Merge only after all checks pass and remove
the merged branch. The earlier performance failures remain failed. Next work
compares job sizes using the same verified placement on both sides, before any
new model/download or dataset expansion.
