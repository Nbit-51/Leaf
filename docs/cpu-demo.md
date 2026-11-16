# Leaf CPU demo

The three examples use separate input types and run inference locally on CPU.
Prepare the model files before an offline demo.

## Install

From the checkout, build and install a platform wheel with a C++17 compiler.
On the demo machine, the wheel bundles all three native executables.

```powershell
python -m pip wheel . --no-deps --wheel-dir dist
$wheel = (Get-ChildItem dist/leaf_cpu-*-win_amd64.whl | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
python -m pip install "${wheel}[graph]"
leaf doctor
```

## Generate with a small decoder

Run this once online to fill Leaf's cache. Add `--offline` for subsequent runs.
Use the same model ID, prompt and precision when comparing two installations.

```powershell
leaf run HuggingFaceTB/SmolLM2-360M-Instruct --bits 32 --prompt "Give three reasons to run a small language model locally." --max-tokens 32 --metrics build/smol-demo.json
```

The metrics report separates native prefill and the cached decode steps.
The [matched-runtime protocol and current results](model-onboarding.md#decoder-costs-by-prompt-length)
describe which measurements can support a latency claim.

## Index a document collection

Use a downloaded EmbeddingGemma 300M snapshot or authenticate with Hugging
Face before the first download. Supply JSONL rows with a `text` field. Include
a unique string `_id` in each row if you want to score retrieval against
judgments. Removing `--offline` allows the first download.

```powershell
leaf index google/embeddinggemma-300m --offline --dataset documents.jsonl --threads 4 --chunk-size 16 --directory build/demo-index --max-new-chunks 1
leaf index google/embeddinggemma-300m --offline --dataset documents.jsonl --threads 4 --chunk-size 16 --directory build/demo-index
```

The second command reuses the verified saved batch and writes
`build/demo-index/embeddings.npy`. Start a new index directory after changing
the input data, model or runtime. To evaluate a real retrieval workload, use
the [SciFact reproduction commands](model-onboarding.md#reproduce-the-scifact-test-split).
The retained result covers 5,183 documents and 300 queries, with 0.7921 nDCG@10
and 0.9219 Recall@10.

## Run a trained individual network

Use a supported ONNX export with one float32 input and one float32 output.
For the trained ResNet-20 example, use its exported ONNX model and a float32
array shaped `(1, 3, 32, 32)` with the CIFAR-10 normalization applied. Set
`--output` to a fresh path.

```powershell
leaf infer resnet20.onnx --input image.npy --output logits.npy
python -c "import numpy as np; x=np.load('logits.npy'); print('Class index', x.argmax(axis=1).tolist())"
```

The [trained CIFAR-10 benchmark](project-reference.md#73-trained-resnet-20-complete-cifar-10-test-split)
provides the dataset, checkpoint and export procedure. The one-command example
includes graph import, optimization and process startup, so use a dedicated
benchmark when comparing warmed native forward times with another runtime.

## Verified Windows wheel

The November 15 wheel was built from the source distribution and installed in
a separate environment with a compiler-free PATH. It generated 32 SmolLM2
tokens offline and resumed a 32-document SciFact job, reusing the first batch
and reproducing the saved rankings for all four pilot queries. On one
CIFAR-10 test image, the trained ResNet-20 command returned class 3 like the
PyTorch reference, with a maximum absolute logit difference of `7.63e-6`.
The optimized wheel also reproduced the accepted 64-token response through
the public `leaf run --source-bf16 --threads 4` command.
[Wheel verification and retained results](../benchmark/validation/cpu-demo/20261115/manifest.json)
