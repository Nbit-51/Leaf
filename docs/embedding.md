# Native document embeddings

Leaf runs the original `google/embeddinggemma-300m` FP32 snapshot through its
native C++ encoder. This is implemented inference with numerical reference
validation. Retrieval-dataset quality remains unqualified; the first CPU-engine
comparison is described below.

## Finite masked-span attention and product workflow — October 13, 2026

The shared vector attention kernel now omits leading/trailing masked value
products when values are finite and rounding is nearest. It retains key order,
interior mask holes and the previous path for NaN/Inf or other rounding modes.
Native comparisons with the previous kernel are bit-exact. The independent
[attention microbenchmark](../benchmark/results/embeddinggemma/attention-span-micro-20261012.json)
has 12 stable cases; the 669-token, radius-256 case takes **15.55% less time**.
This is a kernel result, not a whole-model speedup.

The [combined tail-plus-attention model experiment](../benchmark/results/embeddinggemma/attention-span-combined-ab-20261012.json)
completed six alternating pairs (three warmups, eleven samples per pass). Its
long-input paired reduction was 6.47%, but one candidate pass had p90/p10 of
1.3055, failing stability. All six input qualifications failed. No samples were
removed. The maintainer explicitly accepted enabling attention with only the
verified kernel claim. The separate GEMM narrow-tail experiment remains disabled
by default; its [earlier model gate](../benchmark/results/embeddinggemma/tail-ab-20261011.json)
also failed. The [tail microbenchmark](../benchmark/results/embeddinggemma/tail-micro-20261011.json)
uses vector packing, whereas the encoder uses its existing packing policy;
isolated gains must not be treated as model gains.

The final attention-only [llama.cpp diagnostic](../benchmark/results/embeddinggemma/attention-span-llama-diagnostic-20261012.json)
passed numerical parity but qualified no timing wins. It used just two
alternating pairs and five samples per pass. Leaf's long-input pass medians were
5676.79 and 2306.31 ms; llama.cpp's were 5544.53 and 5349.79 ms. This large
variation prevents a reliable performance conclusion. The user reported light
Twitter browsing during the run; that is a possible confounder, not a measured
cause or a reason to discard samples. These are repeated embedding calls, not
successive generated tokens. Normal desktop activity is allowed while doing
functional checks; performance acceptance needs a separate controlled run.

The [operation profile](../benchmark/results/embeddinggemma/operation-profile-20261012.json)
of the earlier tail candidate measured about 2770 ms for the long input: MLP
GEMM 1013 ms, other projection GEMM 576 ms, attention 1017 ms, and packing 53 ms.
Attention children include values 484 ms, scores 356 ms and softmax 163 ms.
Parent/child timers overlap and profiling adds overhead, including an extra
maximum reduction; these numbers explain where work goes, not release speed.

The product workflow now accepts a Hub model ID for `leaf embed`, downloads its
supporting files automatically, validates metadata before weights, pins the
weight download to the inspected revision, and supports offline cached reuse.
The download contracts and existing CLI regressions pass **171 tests**. An
[installed-wheel check](../benchmark/results/embeddinggemma/attention-span-package-20261013.json)
passes first local preparation, local reuse and a real Hub offline-cache lookup
by model ID, all without a compiler or heavy framework imports. Maximum error
over its five reference embeddings is 2.39e-7. The offline Hub cache was built
from the existing local snapshot using a synthetic revision; online transport
and access errors were tested with mocks, not a new live gated download.

The final bundled decoder passes [ten bit-exact architecture regressions](../benchmark/results/embeddinggemma/attention-span-decoder-20261013.json).
The [installed encoder](../benchmark/results/embeddinggemma/attention-span-release-parity-20261013.json)
also passes short, long and repeated-input reference checks at one thread, two
threads and forced scalar execution (maximum error 3.80e-7 across variants).
The next product step is a small supported-model onboarding matrix: public
decoder, gated embedder, cached offline run and explicit unsupported-model error.
Broader model coverage should extend shared operators and import validation,
with correctness evidence before making performance claims.

## Usage

Supply a model ID and Leaf obtains the supporting files automatically:

```sh
leaf embed google/embeddinggemma-300m --text "Find documents about CPU inference" --task query --output query.npy
leaf embed google/embeddinggemma-300m --offline --dataset ./data/documents.jsonl --output documents.npy
```

For gated Gemma access, accept the model's terms on Hugging Face and sign in once
with `hf auth login`. Leaf reports this step when access is denied. The first
online run checks configuration and module support before downloading weights,
then fetches weights and tokenizer files at the same snapshot revision.
Subsequent offline runs need no network. Both commands also accept registered
aliases. Unsupported configurations fail before a large weight download; this
does not add support for arbitrary embedding architectures.

An existing local snapshot is also supported. Keep its layout, including `config.json`,
`model.safetensors`, tokenizer files, `modules.json`, and the original module
folders `1_Pooling`, `2_Dense`, `3_Dense`. Each dense folder needs its own
`config.json` and `model.safetensors`. Leaf validates module semantics and tensor
shapes before exporting a memory-mapped artifact. Obtain weights under the
[model's access terms](https://huggingface.co/google/embeddinggemma-300m).

```sh
leaf embed ./models/embeddinggemma-300m --text "Find documents about CPU inference" --task query --output query.npy
leaf embed ./models/embeddinggemma-300m --dataset ./data/documents.jsonl --text-column text --task document --threads 1 --output documents.npy --metrics documents.metrics.json
```

JSONL contains one object per line, for example `{"text":"Leaf runs models on CPUs."}`.
The output is a NumPy FP32 array with shape `(documents, 768)`, in input order.
Blank lines are skipped; invalid/missing text is rejected. Output files must be
new. `query` and `document` apply the model's retrieval prefixes; `raw` accepts
already formatted text. Prompt tokens participate in mean pooling.

The first invocation prepares a cached native artifact. Later invocations check
and reuse it. Each invocation loads the native model once and processes documents
sequentially, with projection workers controlled by `--threads`. The launcher
uses NumPy and `tokenizers`; inference needs neither PyTorch nor Transformers.
Platform wheels bundle `leaf_embed`; source installs need a C++17 compiler.
`LEAF_CACHE_DIR` chooses the cache and `LEAF_EMBED_BIN` can select a test binary.

Current limits: supported FP32 snapshot only, 768 output dimensions, at most 2048
tokens per document including prefix/special tokens, and one million documents
per invocation. Overlong text is rejected rather than silently truncated; split
documents explicitly. Quantized embeddings, model-server APIs, parallel document
batching and Matryoshka dimension truncation are not implemented. Use ASCII
native artifact paths on Windows. Treat native artifacts as trusted local files.

## Implementation and evidence — October 10, 2026

The encoder implements embedding scaling, zero-centered RMSNorm (including
Q/K), alternating bidirectional local/full attention, layer-specific RoPE,
GELU-tanh gated feed-forward blocks and four block norms. Output passes through
mean pooling, the two trained identity-activation dense layers, and L2
normalization. The configured local window 512 maps to inclusive radius 256.
Head geometry is independent of hidden width. Native projections reuse Leaf's
tiled GEMM; AVX2/FMA dispatch has a scalar fallback.

| Check | Result |
|---|---|
| Trained snapshot versus frozen FP32 reference | Maximum absolute embedding error **4.77e-7**, minimum cosine **0.999999999994** |
| Native execution variants | One thread, two threads and forced scalar all pass |
| Trained inputs | Five short inputs, a 669-token input crossing the local window, then a repeated short input |
| Reduced architecture | Ten sequences around window boundaries; independent head width, nonzero norm weights, local/full attention and state reuse |
| Invalid input/artifact checks | Empty/overlong inputs, invalid token, bad magic/version and truncated artifact rejected |
| Installed wheel | Offline first preparation and cached JSONL reuse; bundled binary, no compiler or heavy framework imports |
| Existing decoder regression | All ten reduced decoder cases remain bit-identical to the previous shipped decoder |
| Python regression | 901 tests passed; 42 targeted embedding/packaging tests passed after final formatting and output-path guards |

Predeclared embedding gates: max absolute error ≤2e-4, cosine ≥0.99999 and
unit-norm error ≤1e-5. Tests compare final embedding coordinates, not just cosine
or toy retrieval rankings. The trained checks are deliberately small correctness
fixtures; they do not establish retrieval accuracy over a real dataset.

The saved external reference was checked against Sentence Transformers, padding
equivalence and independently calculated pooling/projections. Downloaded files
are identified by SHA256; their upstream revision is unknown. No weight files
are committed. Records:

- [Reference preparation](../benchmark/results/embeddinggemma_reference_20261010.json)
- [Native trained parity](../benchmark/results/embeddinggemma/native-parity-20261010.json)
- [Reduced architecture and rejection tests](../benchmark/results/embeddinggemma/architectures-20261010.json)
- [Installed package](../benchmark/results/embeddinggemma/package-20261010.json)
- [Decoder regression](../benchmark/results/embeddinggemma/decoder-regression-20261010.json)

CLI metrics separate preparation, tokenization, native-process time and output
writing. They are single diagnostic observations. Native-process time includes
startup and artifact loading; it is not warmed forward latency or a qualified
speed comparison. Existing GPT-2/TinyLlama performance claims are unchanged.

## Reproduce correctness

From a source checkout with validation dependencies and a built encoder:

```sh
python tools/export_embedding.py --model ./models/embeddinggemma-300m --output build/embedding/model.leaf
python tools/verify_embedding_architectures.py --executable ./build/native/leaf_embed --output build/embedding/reduced.json
python tools/verify_embedding_native.py --executable ./build/native/leaf_embed --artifact build/embedding/model.leaf --reference ./saved-reference --output build/embedding/parity.json
python tools/verify_embedding_package.py --installed ./installed-wheel --model ./models/embeddinggemma-300m --reference ./saved-reference --output build/embedding/package-check
```

Use fresh output paths. Windows builds can use `scripts/build_embedding.ps1`;
CMake exposes target `leaf_embed`. The reference directory is produced separately
by `tools/verify_embeddinggemma_reference.py` with Sentence Transformers installed;
it contains hash-bound token IDs and FP32 vectors. Reduced checks run in portable
CI on all three operating systems; CI is not a performance benchmark.

## First llama.cpp baseline — October 10, 2026

The original Leaf embedding implementation (`10a59b7`) and llama.cpp both passed
numerical parity. Six alternating pairs completed, with 66 measured samples per
engine/input. Observed pooled medians were:

| Fixture | Tokens | Leaf FP32 | llama.cpp FP32 |
|---|---:|---:|---:|
| Query1 | 18 | 93.53 ms | 70.77 ms |
| Query2 | 18 | 93.15 ms | 70.39 ms |
| Document1 | 23 | 99.07 ms | 83.58 ms |
| Document2 | 25 | 104.39 ms | 89.33 ms |
| Document3 | 17 | 92.40 ms | 68.32 ms |
| Long input | 669 | 3067.86 ms | 2594.69 ms |

**No stable speed ratio is qualified.** Every input failed the combined
per-pass/pooled stability requirement. Both engines had unstable passes; Leaf's
last pass additionally shifted to about twice the earlier latency on several
inputs. The cause was not recorded, so it cannot be attributed to the kernel,
power policy or another application. All measurements, including that pass,
remain in the [baseline record](../benchmark/results/embeddinggemma/llama-baseline-20261010.json).
The observed medians and paired ratios favor llama.cpp throughout; this is not
evidence of a Leaf speed advantage.

Sampled peak process RSS across measured passes was 437.61–437.66 MiB for Leaf
and 1156.14–1157.00 MiB for llama.cpp. This includes initialization and mapped
weight residency under these adapters; it is not private memory, a server's
total memory budget, or a general memory-reduction guarantee. Output parity
errors were at most 4.77e-7 for Leaf and 6.79e-5 for llama.cpp. See
[source/conversion/runtime provenance](../benchmark/results/embeddinggemma/llama-provenance-20261010.json).

A separate [diagnostic profile](../benchmark/results/embeddinggemma/profile-20261010.json)
of original Leaf measured 3090.86ms for the long input: MLP linear1095.30ms,
attention930.96ms, attention linear629.06ms, GELU/gating254.45ms and
RoPE-table construction105.41ms. Norms and other work account for the remainder.
The first optimization targets scalar GELU and repeated RoPE construction;
the larger attention and projection costs require separate experiments.

## GELU and RoPE optimization

The encoder now reuses Leaf's existing AVX2/FMA GELU kernel before gate
multiplication, with scalar fallback. It builds one local and one global RoPE
table per input and shares them across layers. These changes apply by operation
and shape, with no checkpoint-name dispatch or altered weights.

Trained-reference error is at most **3.43e-7** on the vector path. One/two-thread
and forced-scalar checks pass, together with the reduced geometry, attention
boundary and malformed-input cases. Six alternating before/after pairs retained
66 samples per input and binary:

| Fixture | Previous Leaf median | Optimized median | Median paired time reduction | Full speed gate |
|---|---:|---:|---:|---|
| Query1, 18 tokens | 102.46 ms | 90.62 ms | 11.82% | Unstable passes |
| Query2, 18 tokens | 97.59 ms | 88.92 ms | **10.32%** | **Passed** |
| Document1, 23 tokens | 101.89 ms | 92.78 ms | 13.51% | Unstable passes |
| Document2, 25 tokens | 109.40 ms | 89.79 ms | 15.55% | Unstable passes/pool |
| Document3, 17 tokens | 100.93 ms | 89.38 ms | 12.03% | Unstable passes |
| Long input, 669 tokens | 3159.59 ms | 3049.73 ms | 6.59% | Unstable pass |

Paired ratios compare each candidate pass with its corresponding baseline pass;
they differ from the ratio of pooled medians. The implementation is enabled
with a **performance claim limited to Query2**. This is not an all-input timing
acceptance or a qualified win over another engine. Other medians are encouraging
observations, not stable speedups; no samples were dropped or gates relaxed.

- [Before/after samples and gates](../benchmark/results/embeddinggemma/gelu-rope-ab-20261010.json)
- [Trained output parity](../benchmark/results/embeddinggemma/gelu-rope-parity-20261010.json)
- [Reduced architecture checks](../benchmark/results/embeddinggemma/gelu-rope-architectures-20261010.json)

A fresh [candidate/llama.cpp diagnostic](../benchmark/results/embeddinggemma/gelu-rope-llama-diagnostic-20261010.json)
retained two alternating pairs with five samples per input/pass. The long-input
pooled medians were 2799.64ms for Leaf and 2646.75ms for llama.cpp (paired ratio
1.0605). This smaller diagnostic does **not** qualify a speed claim; it locates
the remaining gap without repeating a full acceptance run. All output checks
passed. The short fixtures likewise establish no qualified competitor win.

The Windows wheel also passed an [offline installed-package check](../benchmark/results/embeddinggemma/gelu-rope-package-20261011.json):
first export and cached reuse both matched saved vectors, with maximum absolute
error 2.39e-7. It selected the bundled native executable with no compiler on PATH
and no heavy framework imports. These are packaging/correctness checks, not
latency qualification. The focused Python suite passed 49 tests.

Before another timing campaign, diagnose input length and execution time:
compare different texts at the same token count, then a small set of lengths
around actual kernel/window boundaries. Record wall time and thread CPU time
per sample, alongside available page-fault/context-switch telemetry. Equal-length
queries receiving different stability outcomes does not establish content as
the cause; fixed input order and host scheduling remain confounders. Use the
results to target shared attention/projection kernels, retaining all samples.

Further work should share operator contracts between decoder and encoder plans,
especially normalization and position/attention policies. Decoder plans already
keep model identities in import adapters. Generality should mean adding compatible
checkpoints through plans with little new native code; it does not mean promising
every architecture will work or run faster than every competing engine.

## Reproduce the CPU-engine comparison

### Input and core diagnostics (2026-10-11)

The [completed diagnostic record](../benchmark/results/embeddinggemma/input-cpu-diagnostic-20261011.json)
contains eight instrumented passes: six frozen inputs, two warmups and seven
measurements per input, one thread, physical P-cores at logical CPUs 2 and 4.
Input, engine and core order were reversed for the second half. All passes
preserve reference parity (Leaf max error 3.43e-7; llama.cpp 6.79e-5).
No release kernel changed in this diagnostic.

Pooled medians for the 669-token input (14 samples per engine/core):

| Logical CPU | Leaf wall / thread CPU | llama.cpp wall / thread CPU |
|---|---:|---:|
| 2 | 2712.20 / 2562.50 ms | 2505.02 / 2375.00 ms |
| 4 | 2820.12 / 2554.69 ms | 2660.30 / 2398.44 ms |

These instrumented observations are not acceptance benchmarks. They show a
remaining calling-thread compute gap as well as time not accounted for by that
thread's CPU counter. CPU4 added wall time to both engines without a comparable
increase in thread CPU time; this does not establish a universal best core.

The clearest short-input stall was Leaf Document1 in the reverse CPU2 pass:
**224.445ms wall, 78.125ms thread CPU, 192.61 million cycles, zero page faults**.
Its pass medians were 94.56ms wall, 78.125ms CPU and 189.32 million cycles.
This reproduces delay without a proportional increase in arithmetic execution
or measured page faults. Scheduling/waiting is a plausible explanation; these
counters cannot identify preemption, a waiting cause, clock changes or thermals.
GetThreadTimes is coarse (observed 15.625ms increments), so short-sample wall/CPU
differences are approximate and must not be treated as exact descheduling time.
Cycles are not converted into elapsed time or clock frequency.

Leaf had zero faults in all measured long-input calls; its 271 recorded faults
were confined to one short-input call. llama.cpp recorded 54 faults in total.
These process counters combine soft and hard faults. They do not support blaming
all variation on paging. The two 18-token queries had similar median cycles
within each pass (within about 1.3% for Leaf); their wall-time differences do not
establish a content-dependent kernel problem. Three Leaf input/pass spreads
still exceeded 1.25. All raw samples remain available.

Next implementation experiment: measure a smaller GEMM tail tile. The shared
FP32 token-panel kernel currently computes 16 lanes even for a partially filled
last panel. Compare lengths 16/17, 24/25 and 32/33, preserve increasing-K FP32
accumulation and verify parity before whole-model acceptance. This is a
shape-based candidate for both decoder and encoder, not a measured improvement
yet. Reuse CPU2 as the established baseline, retain CPU4 as a diagnostic, and
reserve scheduler tracing for attributing remaining stalls rather than repeatedly
rerunning unchanged acceptance benchmarks.

Reproduce with `scripts/build_embedding_comparison.ps1 -Diagnostics` and
`tools/diagnose_embedding_inputs.py`; pass the same artifact/reference paths as
below and a fresh output directory. Diagnostic executables have separate names
and never overwrite the saved acceptance binaries. Thirteen diagnostic and
comparison harness tests passed. No system power plan or process priority was
changed; a scoped keep-awake request was restored when the run completed.

### Uninstrumented comparison

The first comparator is llama.cpp b11540, source commit
`1e6f04a75e91da4bb66b5248f3fd4bdeff169d41`, using the official Windows x64
CPU release. Both adapters compile from `benchmark/embedding_runtime_benchmark.cpp`.
The comparison uses original FP32 weights, including both trained dense output
layers. It does not substitute Ollama's BF16 default for an FP32 baseline.

llama.cpp conversion additionally needs the original `tokenizer.model`; the
JSON tokenizer alone did not work with this pinned converter. Preserve the
snapshot and use the unmodified upstream converter. Its dependencies were
isolated from Leaf: Transformers4.57.6, tokenizers0.22.2 and Hub0.36.0.

```powershell
# Place the pinned source in $llama/source and the official CPU DLLs in $llama/bin.
$llama = 'build/competitors/llama-b11540'
$env:PYTHONPATH = (Resolve-Path build/competitors/conversion-deps).Path
$env:HF_HUB_OFFLINE = '1'
python "$llama/source/convert_hf_to_gguf.py" build/models/embeddinggemma-300m --outfile "$llama/embeddinggemma-f32-complete.gguf" --outtype f32 --sentence-transformers-dense-modules
Remove-Item Env:PYTHONPATH
./scripts/build_embedding_comparison.ps1
python tools/benchmark_embedding_competitor.py --leaf-executable build/competitors/leaf-embedding-benchmark.exe --llama-executable "$llama/bin/llama-embedding-benchmark.exe" --leaf-artifact PATH_TO_PREPARED_MODEL.leaf --gguf "$llama/embeddinggemma-f32-complete.gguf" --reference build/embeddinggemma/reference-20261010-final --output build/competitors/fresh-comparison --cpu 2
```

Select a physical P-core appropriate to your machine; logical CPU2 was used on
this host. Timing covers token IDs to normalized vectors, including per-request
reset, pooling, both projections, normalization and output copy. Loading,
tokenization, file I/O and HTTP are excluded from both sides. Each model remains
loaded within its pass. There is no reused prefix cache. Both engines use one
thread pinned to the same CPU; llama.cpp GPU offload is disabled and K/V storage
is requested as FP32. Its automatic attention kernel is retained. FP32 weights
do not imply identical intermediate arithmetic across engines.

Both adapters must pass saved-vector parity before six alternating pairs of
processes run. Each input gets three warmups and eleven measured samples per
pass. The predeclared stability check is p90/p10 ≤1.25 in every pass and pooled
samples. A Leaf win additionally requires median paired Leaf/llama.cpp ≤0.98.
This experiment's gates are explicit and separate from earlier decoder gates.
All samples remain in the report. Sampled process peak RSS includes model loading
and is not a measure of private allocations or total application memory.

These six numerical fixtures establish a small, reproducible engine baseline;
they do not establish retrieval-dataset throughput or general model leadership.
llama.cpp is the first comparator; Ollama service overhead and other engines
still need their own measured runs.

## Next acceptance work

1. Freeze a public retrieval dataset and split, exact queries/documents and
   tokenization. Compare native and reference embeddings plus nDCG@10/Recall@10;
   preserve per-query results and long-document chunking policy.
2. On Windows, measure warmed encoding and complete JSONL-to-vector job time,
   documents/tokens per second and peak process memory against PyTorch. Use the
   same FP32 weights, prefixes, lengths, core/thread policy and output work;
   alternate execution order and retain all samples, including unstable runs.
3. Add CPU-only llama.cpp/Ollama where the same embedding model is supported,
   separating quantized quality differences and service overhead. Add supported
   ONNX Runtime/OpenVINO paths. Defer vLLM's Intel/AMD Linux CPU comparison to the
   planned native Linux phase; do not rank Windows against WSL timings.
4. Expand toward 10–15 distinct trained checkpoints across supported decoder,
   embedding and graph workloads. Prioritize compatible small Qwen/Llama,
   Mistral and Granite configurations after operator/memory checks. Each needs
   real weights, held-out task quality, matched timings and memory measurements.
   Random-weight architecture tests and multiple precision variants do not count
   as additional validated models. Unsupported cases remain visible in the matrix.

Dataset acceleration means completing useful inference jobs sooner: indexing a
document collection, classifying images or scoring records on existing CPUs.
Quality, throughput, peak memory and complete job time define success. Data
loading, augmentation and training are separate operations; no acceleration of
those is claimed by the embedding implementation.
