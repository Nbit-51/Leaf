# Windows FP32 release

The release task is to make the established FP32 kernel stack available through
ordinary Windows builds, including the CLI source cache and native wheel. This
is separate from beating PyTorch or selecting a quantized precision automatically.

## Implementation

- Windows defaults to packed token panels, six-row reuse, vector GELU-new,
  vector prefill attention and paired-row GEMV for entirely FP32 artifacts.
- AVX2 and FMA are checked at runtime. Forced scalar execution and unsupported
  CPUs retain the portable path. Quantized/mixed artifacts retain their previous
  dispatch; their profiles still bind the exact executable hash.
- `scripts/build_decoder.ps1 -ConservativeFp32` or CMake
  `-DLEAF_CONSERVATIVE_FP32=ON` builds the previous conservative implementation.
- Existing explicit experimental builds retain their historical opt-in policy.
  The MLP vector-pack and 96-row scheduling experiments are not promoted.
- Other operating systems retain their existing default pending validation.

## Release checks

Status: qualified and enabled in the normal Windows build.

- [x] Default and conservative Windows builds compile.
- [x] Default logits exactly match the previously validated experimental stack
  on all 1,016 held-out GPT-2 prediction targets, at one and two threads.
- [x] Trained quality, cached chunks and greedy generation match their references.
- [x] Complete ten alternating conservative/default pairs: release gate passes;
  every individual pass also meets the original spread limits.
- [x] Check eight reduced architectures, scalar fallback and unchanged quantized paths.
- [x] Rerun Python tests with an explicit `--basetemp`: **831 passed, zero
  skipped**, four existing ONNX deprecation warnings. The packaging fixture no
  longer requires the legacy `wheel` package when modern setuptools supplies
  its own builder.
- [x] Run native kernel and dynamic KV-cache tests, including reset/reuse isolation.
- [x] Build/install a wheel and verify offline first/cached generation without a compiler.
- [x] Update the README and restore its earlier Mermaid diagram style.

| Source build, GPT-2 FP32 | Prefill median | Cached decode median |
|---|---:|---:|
| Conservative | 389.4917 ms | 31.1941 ms |
| Windows default | 158.8321 ms | 28.27235 ms |

These medians pool all 310 retained samples per build. The primary statistic,
the median of ten paired ratios, gives **59.34% less prefill time and 9.30%
less decode time**. Its bootstrap ratio intervals are 0.40372–0.41027 for
prefill and 0.89439–0.92158 for decode. No pair or sample was discarded.

Evidence: [release comparison](../benchmark/results/windows_fp32_release.json),
[architecture audit](../benchmark/results/windows_fp32_release_architectures.json),
[installed package](../benchmark/results/windows_fp32_release_package.json), and
[installed/native full-logit parity](../benchmark/results/windows_fp32_release_installed_parity.json).
The [final validation record](../benchmark/results/windows_fp32_release_validation.json)
binds the source files, result records and final test log by hash.
The wheel and source-build executables have different binary hashes; their
51,463,168 held-out logits are exactly equal. Both installed first/cached runs
report `optimized_fp32_active: true` with no experimental environment flags.

The predeclared release comparison uses ten AB/BA pairs, 31 retained samples and
ten warmups per process, one thread on CPU 2, and Above Normal priority for both
binaries. Qualification requires at least eight prefill pairs improving by 2%,
plus paired-bootstrap 95% upper bounds of 0.98 for prefill and 1.02 for decode.
Every individual spread check is also reported. This does not replace or relax
the existing automatic precision-selection gates. The evidence is limited to
this host, workload and FP32 quality sample.

Reproduce from the repository root after building both binaries:

```powershell
python tools/qualify_fp32_release.py --before build/fp32-release/conservative/leaf_decoder.exe --after build/fp32-release/default/leaf_decoder.exe --stack build/decode-gemv/candidate/leaf_decoder.exe --workdir build/gpt2_prefill_followup --output build/fp32-release/new-qualification.json
```

The local record is `build/fp32-release/qualification-v2.json`. The initial
`qualification.json` contains no timing passes: sandbox temporary-file access
failed before native validation began. The completed result is also checked in
as `benchmark/results/windows_fp32_release.json`.

## Fresh installed-wheel comparison with PyTorch

This is a separate serial forward/reverse framework comparison on the same
63-token prefix, one cached decode token, last-token logits, one thread on CPU
2, ten warmups and 31 samples per pass. Both PyTorch attention implementations
are measured in fresh processes. It is not a comparison with the old
conservative Leaf build.

| Runtime | First prefill / decode | Reverse-order prefill / decode | All stability checks |
|---|---:|---:|---|
| Installed Leaf | 132.3031 / 25.3510 ms | 157.5968 / 27.3650 ms | Fail |
| PyTorch eager | 159.9427 / 31.0837 ms | 151.9454 / 28.7635 ms | Fail |
| PyTorch SDPA | 161.2784 / 30.4770 ms | 150.9762 / 29.0536 ms | Pass |

The 132 ms observation is real, but the unchanged Leaf binary did not repeat
that level. Pooled Leaf p90/p10 is 1.5608 for prefill and 1.3453 for decode.
Therefore **no stable PyTorch speed win is claimed** from this run. This does
not undo the independent, stable conservative-to-default release comparison.
All samples remain in the [framework record](../benchmark/results/windows_fp32_release_pytorch.json).

## Core-scaling and cache audit

Commit `9837481` already retains the core-scaling work. The 96-row candidate
failed qualification and did not change the runtime. The one/two/four-core
sets were `[2]`, `[2,4]` and `[2,4,6,8]`, each using distinct physical P-cores.
The monitoring-only worker runs observed about 157% process CPU with two cores
and 210–243% with four (100% means one logical CPU fully busy). These include
loading, warmup and serial work; they do not isolate scheduler wait time.

`KVCache` already grows geometrically, preserves previous keys/values when
capacity grows, resets length while retaining allocated capacity, and belongs
to one decoder session. It supports virtual GQA repetition. Dynamic allocation
does not mean paged attention, cross-request prefix caching or a serving
scheduler; those are separate future features.

Next work after this checkpoint: resolve the remaining framework-comparison
variability and qualify complete-command latency, followed by a narrowly
measured remaining bottleneck.
Do not restart the rejected 96-row experiment or use unrelated roadmap features
as prerequisites for this release.
