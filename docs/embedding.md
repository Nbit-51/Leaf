# Native document embeddings

Leaf runs the original `google/embeddinggemma-300m` FP32 snapshot through its
native C++ encoder. This is implemented inference with numerical reference
validation. Retrieval-dataset quality and comparative speed are not yet qualified.

## Usage

Keep the complete local snapshot layout, including `config.json`,
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

Current limits: local FP32 snapshot only, 768 output dimensions, at most 2048
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
