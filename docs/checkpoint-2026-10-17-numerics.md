# October 17 numerical and decode follow-up

Base: merged PR #7, `1e0e57f`. Production arithmetic is unchanged; FP32
single-token SiLU dispatch is now integrated and validated. Preserve the unrelated root PNG. The user authorized
parallel agents and confirmed AC/no heavy work for upcoming acceptance timing.

## Completed evidence; do not repeat

- `build/numerics-20261017/{smol,qwen}-profile.json`: two fresh default profiles
  each, three runs/one warmup, CPU 2. Linear operations take 94–95% of prefill
  and 89–91% of decode; scalar SiLU takes 7–8% of decode. Profiles include
  first use/warmups and are not acceptance timings.
- New generic `tools/trace_decoder_logits.py` traces Llama/Qwen2/NeoX.
  Smol block 1 and Qwen block 1 reproduce the saved failures exactly;
  diagnostic and ordinary native logits are byte-identical. Native/expected
  arrays are in `smol-block1-trace-v2/` and `qwen-block1-trace/` under that build.
- Identical-input probes reproduce current native RMSNorm bit-for-bit.
  Summing FP32 squares in FP64 cuts norm RMS error roughly fivefold in both
  families; multiplication association matches Torch. Reports:
  `smol-norm-trace-report.json`, `qwen-norm-trace-report.json`, `norm-findings.md`.
- The temporary pure-FP32 RMS candidate reduces failing block violations
  Smol 154→68 and Qwen 451→5, but still fails. Do not call it fixed or ship it
  based only on the isolated norm result. See `rms-block-check.json`.
- One split-K candidate uses 3 rows × 16 tokens and two interleaved K banks:
  12 accumulator registers, no inner-loop spills, safe tails/strides/bias.
  Synthetic errors improve about 29%. Combined with RMS it reduces saved
  block failures to Smol 2 / Qwen 0. Still not a complete quality pass.
  All files are isolated under `build/numerics-20261017/splitk2/`.
  Its micro timing completed before acceptance: observed higher cost on both
  shapes, with no qualified pinning/stability claim. No parameter search.

## Concrete performance candidate

`build/numerics-20261017/decode-silu/leaf_decoder.exe` changes only the shared
SiLU dispatch to allow pure FP32 single-token calls. Existing vector math is
reused. Quantized and forced-scalar paths remain excluded; no new engine flag.
Baseline is `build/family-models-20261017/runtime/leaf_decoder.exe`.

`check_decode_silu.py` completed exact full/chunked/tokenwise/threads2/scalar
outputs and generation for Smol/Qwen/TinyLlama plus W8A8 exclusion. Raw case
results persist to `decode-silu/parity.json`. **Complete/passed true: all twenty
cases pass**, including both W8A8 exclusions. Do not repeat these checks.
Gate work directories and frozen token fixtures are prepared. The isolated
matrix timing is saved to `splitk2/timing.json`; no trained speed claim follows.

Smol acceptance is **COMPLETE/PASSED**: decode 83.4384→78.74105 ms, **5.63% lower**;
prefill 490.5763→489.08675 ms, non-regression passes. All twelve individual
passes and pooled distributions stable. Median paired decode ratio 0.94847;
pooled 0.94370. Qwen acceptance is **COMPLETE/PASSED**: decode
115.6098→109.27475 ms, **5.48% lower**; prefill 563.0904→564.99775 ms passes
non-regression. All twelve passes stable. Paired decode ratio 0.94540;
pooled 0.94520. **Do not repeat either acceptance run.**

Final default source build `release/leaf_decoder.exe` SHA256
`83db825956f5657c068a5699efbc35e088527fc34ac24a37122e7805f2ea2900`
passes ten reduced architecture configurations with exact paired outputs,
eight trained release-binding cases, and **1,040 unit tests**. Added dispatch
coverage excludes FP32 weights with 8-bit activations. An initial sandbox run
hit system-temp PermissionError; the full run with workspace TEMP/TMP passed.
Fresh installed wheel passes GPT-2 preparation/repeat and Qwen Hub-ID offline
with no compiler/Torch; Qwen confirms vector calls and exact prior tokens.
The final source guard preserves historical research policy. Numerical
candidates remain isolated and unshipped. See `docs/fp32-decode-silu.md`.

`tools/benchmark_default_abba.py --target-phase decode` predeclares decode
ratio <=0.98 and prefill non-regression <=1.02. The original default prefill
criteria, six balanced pairs, 31 samples/ten warmups and all stability limits
are unchanged. Fourteen focused gate/quality-binding tests pass.

Prepare hard links and frozen 64-token fixtures with:

```powershell
python build/numerics-20261017/prepare_decode_gates.py
```

Then, **serially**, after all builds/probes/model loads finish:

```powershell
python tools/benchmark_default_abba.py --before build/family-models-20261017/runtime/leaf_decoder.exe --after build/numerics-20261017/decode-silu/leaf_decoder.exe --workdir build/numerics-20261017/decode-silu/smol-work --target-phase decode --cpu 2 --pairs 6 --runs 31 --warmup 10 --output build/numerics-20261017/decode-silu/smol-acceptance.json
python tools/benchmark_default_abba.py --before build/family-models-20261017/runtime/leaf_decoder.exe --after build/numerics-20261017/decode-silu/leaf_decoder.exe --workdir build/numerics-20261017/decode-silu/qwen-work --target-phase decode --cpu 2 --pairs 6 --runs 31 --warmup 10 --output build/numerics-20261017/decode-silu/qwen-acceptance.json
```

The above commands are retained for provenance, not a request to repeat them.
Acceptance, dispatch audits, tests, docs and installed parity are complete.
Remaining release step: commit reviewed evidence and publish through a PR;
wait for every head CI check, merge, then delete that temporary branch.
Keep the numerical candidates separate until their full quality gates pass.
Future publication must use a PR because main is protected; delete its branch
only after a confirmed merge. Do not start unattended acceptance on battery or
concurrently with other workloads.
