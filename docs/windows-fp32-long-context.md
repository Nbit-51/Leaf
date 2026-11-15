# Four-core FP32 inference on Windows

Leaf can now split long FP32 prefill projections by token rows and divide attention
by head across its existing persistent workers. Leaf assigns complete outputs to
workers, preserving the per-output floating-point accumulation order. The shared
AVX2 attention kernel also skips whole groups of masked positions during softmax.
The path uses the converted FP32 weights directly and leaves the model's math
policy in place. Select four threads with `leaf run ... --threads 4` on a CPU with
AVX2 and FMA. The projection path starts at 768 prompt tokens. Parallel attention
also runs for single-token decode with at least 512 cached positions including
the new token.

## Decoder validation

Windows, Core i7-14700HX, four physical P-cores (`0, 2, 4, 6`), plugged in under
the existing Better Battery policy. Each comparison used the same full FP32
artifact on both sides and measured six order-balanced pairs with five samples
and two warmups per pass. The workload prefilled 768 tokens, then timed a cached
single-token forward pass. The reductions use the median of the paired ratios.
Every per-pass and pooled p90/p10 spread stayed within 1.25× for the table below.

| Model and math policy | Prefill p50, old → current | Paired reduction | Cached decode p50, old → current | Paired reduction |
| --- | ---: | ---: | ---: | ---: |
| OPT-125M, accurate FP32 | 1,302.81 → 616.82 ms | 52.80% | 26.88 → 18.07 ms | 31.32% |
| SmolLM2-360M, fast FP32 | 4,558.64 → 2,140.51 ms | 53.19% | 77.76 → 54.00 ms | 30.51% |

Both models' prefill and next-step cached logits matched the previous Leaf binary
bit-for-bit at 64 and 768 tokens. Each tested 16-token greedy continuation also
matched. In a separate held-out OPT evaluation at four threads, Leaf and PyTorch
agreed on all 1,016 next-token predictions and six greedy continuations. Maximum
absolute logit error was `7.82e-5`, and every logit passed the existing tolerance.
SmolLM2's full-FP32 checks here measure parity with the prior Leaf build.

In the direct four-core comparison against PyTorch 2.12 CPU SDPA, Leaf's pooled
medians were 637.62 vs 969.42 ms for prefill and 18.82 vs 29.42 ms for cached
decode. The median paired ratios were 0.877 and 0.605. All boundary logits passed
tolerance, with maximum absolute error `1.29e-5`. Both runtimes had passes that
exceeded the spread limit, so this comparison remains diagnostic for both phases.

The masked-softmax change alone had standalone paired prefill estimates of 6.52%
on one-core OPT and 8.08% on source-BF16 SmolLM2. Those two timing runs missed the
spread gate and remain diagnostic. The qualified percentages above apply to the
combined release with four cores on the tested FP32 workloads.
[Raw paired samples and parity records](../benchmark/validation/windows-fp32-20261115/decoder/manifest.json)

## CPUs with fewer than four physical cores

Four workers are a performance setting. `leaf run` defaults to one worker.
Use `--threads 1` on a single-core system or start with `--threads 2` on a
dual-core system. The existing decoder kernels handle those settings and retain
the requested FP32 math policy. Runtime feature checks select eligible AVX2/FMA
kernels. A CPU without those instructions uses the portable path. Individual
model weights and the KV cache still have to fit in available memory.

`--threads` sets the worker count. Windows chooses where the workers run unless
you pass `--worker-cpus` for a controlled experiment. The performance tests on
the i7-14700HX used four distinct P-cores mapped to logical CPUs `0, 2, 4, 6` on
that laptop. Other machines need their own affinity mapping. This processor
has eight P-cores and twelve E-cores, while an i7-1365U laptop processor has two
P-cores and eight E-cores. Four P-cores are a benchmark configuration.
[14700HX specifications](https://www.intel.com/content/www/us/en/products/sku/235997/intel-core-i7-processor-14700hx-33m-cache-up-to-5-50-ghz/specifications.html)
· [1365U specifications](https://www.intel.com/content/www/us/en/products/sku/232141/intel-core-i71365u-processor-12m-cache-up-to-5-20-ghz/specifications.html)
· [Windows CPU topology](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_cpu_set_information)

We checked the fallback with the same OPT-125M FP32 artifact on the review laptop.
One and two workers were bit-identical to four in the 64-token prompt plus one
cached-step check. A forced-scalar one-worker run had a maximum absolute logit
error of `2.48e-5` and passed the FP32 tolerance. All three modes matched the
same 16-token greedy continuation. These are checks of the executable paths on
the current i7. Absolute latency on a 2016 processor still needs a physical run.

We also restricted the installed wheel process to one or two physical P-cores
and let Windows place its workers within those CPUs. Both cases matched all
38,659,168 OPT logits from the four-P-core build at 768 prompt tokens plus one
cached step. A one-P-core run with the scalar backend forced passed the shorter
logit check, and every case matched the greedy continuation.
[Restricted-CPU scheduler check](../benchmark/validation/windows-fp32-20261115/pcore-diagnostics/manifest.json)

A clean Windows wheel install generated the same pinned OPT continuation using
`--offline` and the bundled runtime with the compiler removed from `PATH`. Long-context
logits in the installed wheel matched the previous build bit-for-bit on FP32 OPT
and source-BF16 Qwen3 at 768 tokens plus one cached step. The local Python suite
ran with the bundled decoder, embedding and graph binaries, with 1,557 passes
and 21 skips.
[Fallback and wheel evidence](../benchmark/validation/windows-fp32-20261115/fallback/manifest.json)

In a separate 768-token observation on that i7, two pinned physical cores
measured 1,655.53 ms prefill and 29.29 ms cached decode. One pinned core measured
2,378.15 ms prefill and 38.08 ms cached decode. Each result pooled seven samples
after two warmups. One-core decode missed the spread limit, and the observations
were not interleaved, so this is a sizing reference rather than a speed claim.

An older two-core machine can start with a small checkpoint and a short prompt.
For example, the FP32 Pythia-70M weights alone take roughly 0.28 GB, with
additional memory for the model state, KV cache and temporary buffers.

```sh
leaf run EleutherAI/pythia-70m --bits 32 --threads 2 --fp32-math accurate --prompt "Once upon a time" --max-tokens 16
```

An Intel Core i3-6100 exposes four hardware threads across two physical cores
and supports AVX2. Its threads in each core share execution resources. An Intel
Pentium G4400 or Celeron G3900E exposes two cores and two hardware threads, and
Intel lists SSE4.2 as their highest SIMD extension. CPU tier and model are more
useful for choosing the test matrix than model year alone.
[Core i3-6100](https://www.intel.com/content/www/us/en/products/sku/90729/intel-core-i36100-processor-3m-cache-3-70-ghz/specifications.html)
· [Pentium G4400](https://www.intel.com/content/www/us/en/products/sku/88179/intel-pentium-processor-g4400-3m-cache-3-30-ghz/specifications.html)
· [Celeron G3900E](https://www.intel.com/content/www/us/en/products/sku/90713/intel-celeron-processor-g3900e-2m-cache-2-40-ghz/specifications.html)

Valve's October 2026 survey reports 2.86% with fewer than four physical cores
among its Windows respondents, summing the one-, two- and three-core rows.
This is a sample of Steam users. For a university deployment, inventory the lab
machines directly before estimating how many use these settings.
[Steam Hardware Survey](https://store.steampowered.com/hwsurvey/cpus)

## Two-worker performance experiment

With the i7 process restricted to two physical P-cores, an experimental build
enabled token-panel and attention-head scheduling for two workers. Both models
ran the same FP32 artifacts and accurate math policy as the installed Leaf
baseline. At 768 prompt tokens, both models passed the prefill spread gate.

| Model | Warm prefill p50, previous → candidate | Paired reduction |
| --- | ---: | ---: |
| OPT-125M | 1,645.55 → 1,139.05 ms | 31.09% |
| Pythia-70M | 807.41 → 565.72 ms | 29.56% |

Both models matched all reference Leaf logits on the tested chunks and the same
16-token greedy continuation. Cached decode medians were lower, but both decode
runs missed the spread limit. This experiment measures a two-core budget on the
review machine. An older physical CPU will have its own latency and memory
behavior. [Candidate source and complete raw samples](../benchmark/validation/windows-fp32-20261115/two-worker-pilot/manifest.json)

A follow-up limited the two-worker change to prefill and retained the existing
decode dispatch. Both models again passed prefill stability, with paired
reductions of 30.56% on OPT and 29.69% on Pythia and bit-exact tested outputs.
Paired decode latency stayed within the 2% non-regression margin on both, and
decode spread again exceeded 1.25×. The two-worker optimization remains a
candidate under the existing release gate.
[Prefill-only candidate and raw timing](../benchmark/validation/windows-fp32-20261115/two-worker-prefill/manifest.json)

## Run the budget-laptop check

Install the Leaf platform wheel and `psutil` in the same environment. From the
checkout, prepare the pinned OPT snapshot once, then run the benchmark offline.
On a single-core machine pass `--threads 1`. On a dual-core machine collect both
one- and two-worker results. A four-core machine can include `--threads 1 2 4`.
Use the operating system's default scheduling for a normal-use result.

```powershell
$model = 'facebook/opt-125m'
$revision = 'e3bf7e1695cb08c7963df13dbf144e0d1812717c'
leaf run $model --revision $revision --bits 32 --fp32-math accurate --raw --prompt "Once upon a time, a small robot found" --max-tokens 16
python -m tools.benchmark_budget_cpu $model --revision $revision --threads 1 2 --output build/older-laptop-opt
```

Each observation starts a fresh CLI and decoder process with cached weights.
The report includes first visible text from process launch, native prefill,
median cached decode step, decoder peak resident memory, and raw samples. The
peak excludes Python and the rest of the system. The initial file checks and
warmup touch the model on disk. Use the same prompt and revision on other
machines. Pass `--prompt-file` for a fixed longer workload, including prompts
with at least 768 tokens when evaluating the four-worker long-context path.
The actual token count can be read from `prompt_token_ids` in the report.
[Benchmark runner](../tools/benchmark_budget_cpu.py)

### Constrained i7 reference run

The installed Windows wheel ran six fresh processes per setting on an
i7-14700HX with 16 GB RAM. Each one- and two-worker run could use the same two
physical P-cores. The four-worker run used a separate mask with four physical
P-cores. Windows scheduled workers inside the masks. All long-prompt runs used
the same 970 input tokens and generated the same 16 output tokens.

| Prompt | Workers | First visible text p50 | Prefill p50 | Decode step p50 | Max decoder peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 tokens | 1 | 1.06 s | 227 ms | 28.8 ms | 480 MiB |
| 10 tokens | 2 | 1.04 s | 213 ms | 30.6 ms | 480 MiB |
| 970 tokens | 1 | 4.39 s | 3.51 s | 40.0 ms | 609 MiB |
| 970 tokens | 2 | 4.45 s | 3.62 s | 42.7 ms | 609 MiB |
| 970 tokens | 4 | 2.59 s | 1.77 s | 26.6 ms | 609 MiB |

The short two-worker prefill and decode timings exceeded the spread limit.
The three long-prompt configurations passed it for first visible text, prefill,
and decode. The four-worker run occurred separately and had a larger allowed
core budget. These results describe the current machine. Use the same workload
on an older physical laptop to establish its actual speed.
[Raw installed-wheel results](../benchmark/validation/windows-fp32-20261115/budget-cpu/manifest.json)

On the same two-core mask, a separate warmed native test forced the scalar
backend for Pythia-70M. At 64 input tokens and 16 output tokens, median
decode was 32.85 ms per step with one worker and 16.97 ms with two. Both
decode configurations passed the spread check and generated the same tokens
as the AVX2 backend. Scalar prefill medians were 829 ms and 426 ms, with
the two-worker prefill exceeding the spread limit. Forcing scalar on this i7
exercises the fallback code for CPUs without AVX2. An actual older CPU will
have its own timings.
[Raw forced-scalar results](../benchmark/validation/windows-fp32-20261115/budget-cpu/manifest.json)

On one allowed P-core, the 10-token OPT prompt reached first visible text in
0.56 s when Leaf opened the cached snapshot by local path and 1.09 s when
Leaf resolved the pinned Hub ID offline. These were two separate six-run
observations with matching prompt IDs, weights, decoder and output tokens.
To use the local path, run `leaf inspect` with the cached model and revision,
then pass its `snapshot` path to `leaf run --offline`.
[Raw startup results](../benchmark/validation/windows-fp32-20261115/budget-cpu/manifest.json)
