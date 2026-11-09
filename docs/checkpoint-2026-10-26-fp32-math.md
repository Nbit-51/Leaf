# October 26: native accurate FP32 policy

Continue on `codex/mlp-schedule`. Preserve the unrelated `docs/embedding.md`
edit and untracked October 8 image. No power setting was changed, and no timing
acceptance run was attempted today. The October 22 failed/interrupted timing
records remain failed/incomplete. Do not rerun those unchanged.

## Implemented

- Shared native `FP32Math::Accurate` and `--fp32-math accurate` in `leaf run`
  and `leaf evaluate`. `fast` remains the default. Native metadata reports the
  selected policy; the CLI rejects an unsupported override before model weights.
- `accurate_fp32.h`: K=256 FP32 partials with a balanced reduction, wider sum of
  FP32 RMS squares, precise scalar projection fallback. No model-ID dispatch.
- Explicit packing reuse within unchanged Q/K/V and up/gate groups. Pointer
  identity alone never permits reuse, and calibration-transformed input does
  not qualify. Quantized/mixed-weight requests cannot silently select accurate.
- Accurate `--bits auto` selects unquantized FP32 and bypasses quantized speed
  profiles. Existing BF16 head/tied-embedding storage composes with this policy;
  its scope is unchanged. No new artifact version or interior BF16 storage.
- Old accumulation experiments reconstruct historical `a78fa51` engine sources,
  rather than patching the newly integrated decoder into a duplicate policy.
- README usage, native policy diagram and focused accuracy/storage docs updated.

## Verified

All local outputs are under `build/fp32-math-20261026/`:

- `trained/quality.json`: complete/passing exact rebinding to the prior qualified
  accurate binary. Sixteen complete 128-token blocks, twelve saved continuations,
  ten scalar/cache/thread/storage controls and eight ordinary trained regressions
  pass. This is not a new independent PyTorch oracle run. Original strict
  tolerances and reference qualification are retained.
- `test_accurate_fp32.exe`: 43,456 output checks, AVX/private-reference exact,
  independent FP64 error checks, portable fallback and output padding.
- `all-units.xml`: **1,204 passed, two optional cases skipped**. Earlier focused
  passes: 244 plus one skip, then 180 including evaluation and CLI policy checks.
- `architectures.json`: ten reduced random-weight architecture configurations,
  with paired bit-exact default regression against the preceding ordinary binary.
- `installed-check.json`: isolated installed doctor and offline generation with
  accurate math + source BF16 storage + two explicit workers. No compiler path,
  binary override or Torch/Transformers import is needed.
- `installed-code-binding.json`: installed `.text`, `.rdata`, `.data` hashes
  match the fully trained-qualified cached decoder. PE timestamps may differ.
- `historical-reconstruction/build.json`: historical K=256 candidate rebuilds.

The first focused attempt failed because Python's Windows sandbox temporary
folder denied file writes/cleanup (75 direct permission failures and one
downstream missing-metrics failure). It remains in `unit-focused.xml`. Reruns
use workspace-local TEMP/TMP, fresh `--basetemp`, and disabled pytest cache.
Do not delete or relabel that attempt. It is an environment/setup failure,
not evidence of a numerical kernel failure.

## Reproduce locally

```powershell
python tools/validate_fp32_math.py --binary (Get-Content build/fp32-math-20261026/runtime-final.txt -Raw).Trim() --directory build/fp32-math-20261026/new-trained-run
```

Use a new directory; do not overwrite the retained trained report. The developer
migration audit requires the earlier immutable model artifacts and qualified
binaries, and performs no downloads or speed qualification. End users can run
an independent quality evaluation via `leaf evaluate --fp32-math accurate`.

The built wheel is
`build/fp32-math-20261026/wheels/leaf_cpu-0.2.0-py3-none-win_amd64.whl`, SHA-256
`d788ec4711bf5380a470f8d285f28d63277516df6f4efa03f7c4903770c35eb9`.
It includes the new kernel header and a native binary with the policy capability.
The isolated installation is under `installed/`; normal source CLI use does not
need a decoder environment variable.

## Remaining

The accurate path is a usable opt-in; the ordinary fast default's previous
strict tolerance misses are not silently declared fixed. Default promotion and
new latency claims require controlled performance evidence. Before timing,
attribute per-pass CPU work/clock changes under the user's chosen efficiency
mode instead of repeating unchanged settling experiments.

Broader original-BF16 interior storage is a separate future format/reader/export
change requiring exact source-value, full-model, cache and installed-path checks.
Dataset and individual-network expansion remain deferred until the model runtime
milestone is settled. Avoid a new model download to substitute for these checks.
