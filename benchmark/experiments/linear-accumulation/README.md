# Shared linear accumulation candidate

**October 18 continuation:** the precise-scalar candidate now passes complete
Smol/Qwen eight-block quality, cache/thread/scalar checks, ten reduced
architectures and 1,048 unit tests. Smol's controlled timing is stable but
prefill is 4.35% slower; Qwen timing is in progress. Released defaults remain
unchanged. See [current findings](../../../docs/shared-accumulation.md) and
[live checkpoint](../../../docs/checkpoint-2026-10-18-accumulation.md).
Build this revision with `build.py --precise-scalar --output FRESH_DIRECTORY`.
The October 17 commands/results below are retained historical evidence.

Windows experiment based on `main` at `107cb98`. Release dispatch is unchanged.
The generated candidate marks its native metrics with
`experimental_accumulation_build: true`; do not package or advertise it as a
validated release.

The previous two-bank GEMM reduced row reuse from six to three. This candidate
keeps the six-row × sixteen-token tile, restarts FP32 FMA partial sums every
128 columns, and combines those partials in FP64 before one FP32 rounding and
the final bias addition. Explicit partial buffers occupy at most 1,152 bytes
per tile. Existing input packing and weight storage are reused. Pure-FP32
RMSNorm also sums FP32 squares in FP64, retaining its FP32 scale calculation.
LayerNorm and quantized arithmetic remain on their original paths. Scalar
GEMM fallback is unchanged; pure-FP32 RMSNorm also changes under forced scalar.

## Completed small checks — do not repeat

Current local build: `build/linear-accumulation-20261017-v2/`.
The `results/20261017/` copies preserve its build identities and small results;
large logits, binaries and model weights stay in the ignored build/cache.

- Thirteen GEMM shapes pass FP64-reference tolerance checks, strided padding,
  input immutability, bias and scalar-fallback equality. K boundaries include
  127/128/129/255/257. A high-dynamic-range RMS case recovers small squares lost
  by serial FP32 summation.
- Synthetic projection RMSE decreases from 1.7904e-7 to 6.5925e-8 on
  63×960→2560 and from 1.6759e-7 to 6.4318e-8 on 63×896→4864.
- Frozen failing block **1 only**, 128 tokens each, unchanged
  `abs(error) <= 2e-4 + 2e-4*abs(reference)`:

| Model | Original violations | Previous norm + two-bank GEMM | This candidate |
|---|---:|---:|---:|
| SmolLM2-360M | 154 | 2 | **0** |
| Qwen2.5-0.5B | 451 | 0 | **0** |

Maximum tolerance ratios are 0.45556 and 0.37885; next-token agreement is 100%
on each block. This is not an eight-block, cached-decode or generation pass.
The candidate changes reduction order intentionally; exact old FP32 logits
are not expected. Strict quality limits have not changed.

One short packing-inclusive ABBA micro probe, observed child affinity `[2]`,
has pooled baseline/candidate medians 3.36265/3.49745 ms and
5.8193/6.0978 ms. These observations contain spikes and establish **no stable
speed win or whole-model regression**. Full acceptance was not run. Avoid
starting long acceptance before addressing the observed kernel overhead.

The first build exposed a missing explicit default argument after selecting
a function pointer; v2 fixes it. An initial timing wrapper used a nonexistent
`Popen.check_returncode`; its samples were lost, and the corrected short probe
was rerun once with raw stdout retained. Neither failure changed the kernel
math or quality tolerance.

## Build or run small checks on a new revision

Use a **fresh output directory**; scripts reject overwriting retained results.

```powershell
python benchmark/experiments/linear-accumulation/build.py --output build/linear-accumulation-next
python benchmark/experiments/linear-accumulation/run_probe.py --build build/linear-accumulation-next
python benchmark/experiments/linear-accumulation/check_saved_blocks.py --build build/linear-accumulation-next
```

The saved-block check uses retained references under
`build/numerics-20261017/` and model/cache records under
`build/family-models-20261017/`; paths can be overridden with `--fixtures` and
`--quality-root`. It never downloads models or regenerates reference outputs.
To collect another explicitly diagnostic micro probe after a source revision:

```powershell
python benchmark/experiments/linear-accumulation/run_probe.py --build build/linear-accumulation-next --timing
```

## Deferred larger verification

These commands are queued, **not executed**. They use the existing v2 build;
do not rebuild it or rerun completed small checks unless sources change.

```powershell
cd C:\Users\navaneeth\leaf
$env:LEAF_DECODER_BIN = (Resolve-Path build/linear-accumulation-20261017-v2/leaf_decoder.exe).Path
$env:LEAF_CACHE_DIR = "$PWD\build\family-models-20261017\cache"
$env:PYTHONPATH = "$PWD\build\python-deps"
$env:TEMP = "$PWD\build\linear-accumulation-20261017-v2\tmp"
$env:TMP = $env:TEMP
$models = Get-Content build/numerics-20261017/model-paths.json -Raw | ConvertFrom-Json
python -m leaf.cli evaluate $models.smol.snapshot --offline --dataset benchmark/data/wikitext2-test.parquet --dataset-source "WikiText-2 test split; eight length-128 blocks" --blocks 8 --sequence-length 128 --generate 16 --threads 1 --output build/linear-accumulation-20261017-v2/full-quality-smol
python -m leaf.cli evaluate $models.qwen.snapshot --offline --dataset benchmark/data/wikitext2-test.parquet --dataset-source "WikiText-2 test split; eight length-128 blocks" --blocks 8 --sequence-length 128 --generate 16 --threads 1 --output build/linear-accumulation-20261017-v2/full-quality-qwen
python tools/verify_decoder_architectures.py --executable $env:LEAF_DECODER_BIN --expect-silu-defaults --output build/linear-accumulation-20261017-v2/architecture-quality.json
```

Inspect each completed quality report's `passed` field; process success alone
does not qualify quality. Confirm frozen artifact/token hashes match the
earlier records. Do not use `--before-executable` for an exact FP32 regression:
this candidate deliberately changes FP32 arithmetic. Separately bind unchanged
quantized execution and validate forced scalar, two threads, chunked cache and
generation against the reference before promotion. Include trained GPT-2 and
TinyLlama regression checks before a release.

After quality and kernel overhead are resolved, run serial full-model ABBA
against `build/numerics-20261017/release/leaf_decoder.exe` on AC and an idle PC.
Use frozen existing workdirs, six pairs, 31 samples and ten warmups, retaining
the existing >=2% target improvement, <=2% other-phase regression and stability
limits. No current speed or full-family quality claim is authorized by these
small results. Clear the developer `LEAF_DECODER_BIN` override before normal use.
