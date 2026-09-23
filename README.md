# Leaf

Leaf is a CPU inference engine and model optimizer. It takes an already trained
model, imports an ONNX graph, removes redundant work, and runs the supported
graph in a small C++ runtime. The development machine may use Python, PyTorch,
ONNX, and calibration data; the intended deployment machine only needs the
exported model and native runtime.

The project is actively developed. Its native path runs FP32 CNN graphs and
selected calibrated INT8 CNN/linear graphs. ResNet-18, CIFAR-10, and
Qwen2.5-0.5B are validation workloads, not model-specific design targets.
Full decoder-only Transformer execution is still pending.

## 1. Why build Leaf?

Large inference frameworks can add installation size and Python overhead to
CPU deployments. Leaf's goal is to turn a trained network into a compact,
predictable CPU artifact
while measuring every change against its numerical reference and prior
latency. It is an inference project, not a training framework.

The concrete targets are:

1. Import supported ONNX graphs through a shared IR and expand operator
   coverage across model families.
2. Remove constant work, fuse operators, and reuse activation memory.
3. Reduce weight and activation cost with calibrated INT8; evaluate
   weight-only formats after FP32 correctness and model-quality gates.
4. Move hot inference loops into C++, with a portable scalar build and
   architecture-specific acceleration where available.
5. Reproduce artifacts across representative CPU machines and measure latency,
   memory, and model quality without assuming a particular model or capacity.

The speed rule is evidence driven: a new implementation must preserve outputs
within a stated tolerance and beat its previous Leaf baseline under the same
shape and machine conditions before it is called a performance improvement.
PyTorch is an external comparison, not a substitute for this regression gate.

## 2. What runs today?

| Component | Current state | Boundary |
|---|---|---|
| ONNX importer and graph IR | Implemented | Shape metadata, graph validation, producer/consumer maps |
| Constant folding | Implemented | Small constant-only subgraphs; 16 MiB materialization guard |
| CNN fusion | Implemented | Conv+BatchNorm folding, then Conv+ReLU |
| Transformer rewrites | Graph passes implemented | MatMul+bias and Gemm+activation; Qwen-pattern RMSNorm, RoPE table, RepeatKV, Attention, SwiGLU MLP |
| INT8 calibration and conversion | Implemented in Python | Per-tensor activation scales, per-output-channel Conv/Gemm/MatMul weight scales; NumPy execution simulates quantized values |
| Memory planning | JSON export implemented | 64-byte-aligned liveness plan; C++ still uses its runtime buffer pool rather than the exported offsets |
| `.leaf` artifact and native executor | FP32 and selected INT8 paths implemented | Version 2 stores aligned typed weights and per-channel scales; native Conv/Gemm/MatMul INT8 dispatch is opt-in |
| Native kernels | FP32 and INT8 kernels implemented | Scalar build is available; AVX2/FMA build is optional and faster on the measured host |
| Full-model Qwen in Leaf | Pending | Qwen benchmark below provides a PyTorch CPU reference |
| Structured pruning and trained-model quality gates | Pending | No CIFAR accuracy or Qwen perplexity claim yet |

The Qwen-pattern fusion passes identify and rewrite graph motifs, but their
custom nodes do not yet have a complete native execution path. Pattern
recognition alone is not a claim of Qwen inference speedup.

General-purpose describes the architecture and intended direction, not a
claim that every ONNX model runs today. The native executor currently accepts
one graph input and output, and Conv is limited to one NCHW image with one
group. Unsupported operators and layouts fail explicitly. Operator coverage,
dynamic shapes, batching, and CPU-specific dispatch are expanded against
independent model tests rather than hard-coded for the named benchmarks.

## 3. Architecture and artifact boundary

```mermaid
flowchart LR
    subgraph Dev[Development machine]
      PT[Trained PyTorch model] --> ONNX[ONNX export]
      ONNX --> IR[Leaf graph IR]
      IR --> CF[Constant folding]
      CF --> FU[CNN and Transformer rewrites]
      FU --> FP[FP32 .leaf artifact]
      FU --> CAL[Representative data calibration]
      CAL --> Q[INT8 .leaf artifact and error simulation]
      FU --> MP[Liveness memory plan JSON]
      PT -. numerical reference .-> CHECK[Parity and quality checks]
      Q --> CHECK
      FP --> CHECK
    end
    subgraph Target[CPU deployment]
      FP --> EX[C++ graph executor]
      Q --> EX
      EX --> POOL[Reusable activation buffers]
      EX --> KERNEL[Portable or AVX2 FP32/INT8 kernels]
      KERNEL --> OUT[Prediction]
    end
    MP -. arena integration pending .-> EX
```

There are two representations today. Version 2 `.leaf` carries FP32/INT8
initializers and calibrated scales for the native executor. The versioned
`leaf-memory-plan-v1` JSON carries planned activation offsets for inspection
and future runtime integration. They are not yet one artifact with a directly
applied arena plan.

## 4. Optimization pipeline, step by step

```mermaid
flowchart LR
    A[ONNX import] --> B[Validate IR]
    B --> C[Fold constants]
    C --> D[Graph rewrites]
    D --> E[Calibrate INT8]
    E --> F[Plan memory]
    D --> G[FP32 or INT8 binary export]
    G --> H[C++ whole-model execution]
    E --> H
    F -. planned arena pending .-> H
    H --> I[PyTorch parity and latency gate]
```

### Step 1: Import and validate

Export an eval-mode PyTorch model to ONNX, then load it with
`Graph.from_onnx`. Leaf stores nodes, initializers, shapes, dtypes, inputs,
outputs, and metadata. Validation checks missing values and cycles before a
rewrite can alter the graph. The ResNet test uses a full 20-Conv, eight
residual-Add, one-Gemm architecture at 32×32 for fast offline verification.

### Step 2: Fold constants

The optimizer evaluates constant-only arithmetic and shape subgraphs once at
export time. It can fold `Constant`, `Identity`, arithmetic, `MatMul`, `Gemm`,
`Reshape`, `Transpose`, `Concat`, `Squeeze`, `Unsqueeze`, `Gather`, `Shape`, and
`Cast`. Outputs larger than 16 MiB remain as nodes, preventing accidental
artifact inflation. Unused initializers are removed.

### Step 3: Rewrite the graph

For CNNs, BatchNorm's fixed eval statistics become Conv weights and bias;
then a single-consumer ReLU can become the Conv epilogue. For transformer
feed-forward blocks, `MatMul + constant bias` becomes `Gemm`, followed by
optional GELU, SiLU, or ReLU epilogue fusion. Separate Qwen ONNX pattern
passes recognize RMSNorm, RoPE table construction, grouped KV repetition,
attention, and SwiGLU MLP. Each pass preserves graph order and checks
consumer relationships so a shared intermediate is not removed incorrectly.

### Step 4: Calibrate and simulate INT8

Representative input samples collect min/max activation ranges. Leaf uses
symmetric INT8 with one scale per activation tensor and one scale per output
channel for Conv/Gemm/MatMul weights. The NumPy reference executor then
simulates quantize/dequantize at those nodes and compares the output with
PyTorch. Version 2 `.leaf` stores INT8 initializer bytes and scales directly;
the native executor quantizes activations and dispatches Conv/Gemm/MatMul.
When a weight is shared by consumers that need different channel axes or by
an FP32 consumer, export retains separate correctly typed copies.
The real CIFAR subset below uses 32 images for calibration and 100 different
images for evaluation.

### Step 5: Plan memory

`plan_memory` computes first and last uses of each statically shaped tensor.
It assigns 64-byte-aligned offsets, reusing a block only after its prior
tensor dies. Unknown shapes are listed explicitly as unplanned. A graph
fingerprint lets an exported plan be checked against the graph it describes.
The C++ executor currently reuses retired buffers via a thread-local pool;
applying the exported offsets directly is the next integration step.

### Step 6: Export and execute in C++

The exporter writes a versioned `.leaf` file with nodes, attributes, and
32-byte-aligned typed weights. `leaf_infer` loads it and executes the supported CNN path:
Conv, BatchNorm, ReLU, Add, MaxPool, GlobalAveragePool, Flatten, Gemm, and
Identity. `leaf_graph_bench` measures repeated full-graph inference. Native
scalar kernels are correctness baselines; AVX2/FMA GEMM, im2col Conv, packed
Conv weights, fused ReLU, and signed INT8 GEMM/Conv kernels are benchmarked
separately. Native INT8 graph execution is available for supported ops, but
remains opt-in because the measured small whole graphs below are slower.

### Step 7: Keep or reject a speed change

Run numerical parity first, then warmed latency tests on the same machine,
shape, dtype, and thread count. A microkernel speedup is recorded as a
microkernel result; it is not assumed to speed up the complete model. The
project's native benchmark command fails if an optimized kernel loses to its
scalar Leaf baseline.

## 5. Reproduce the development checks

From the repository root on Windows, install the development dependencies and
run the complete currently automated gate:

```powershell
python -m pip install -r requirements-dev.txt
./scripts/verify_all.ps1
```

That command runs Python tests, builds and runs native correctness tests,
enforces the native kernel speed gate, writes the deterministic PyTorch/NumPy
baseline JSON, and verifies the FP32 C++ ResNet output against PyTorch. The
latest run passed all 25 Python tests and the native/C++ checks. The default
native build uses GCC with `-O3 -mavx2 -mfma` on compatible x86 CPUs. A
portable scalar build is available with:

```powershell
./scripts/build_native.ps1 -Portable -BuildDirectory build/portable
./build/portable/leaf_native_tests.exe
```

The portable build passed the native kernel and INT8 graph parity checks.
CMake is an alternative:

```powershell
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake --config Release
ctest --test-dir build/cmake -C Release --output-on-failure
```

For a portable CMake build, configure with `-DLEAF_ENABLE_AVX2=OFF`.

The CMake route was also configured, built, and tested on Linux; all five
CTest targets passed.

Individual checks can be run with:

```powershell
python -m pytest -q
./scripts/build_native.ps1
./build/leaf_native_tests.exe
./build/leaf_kernel_bench.exe --enforce-speedup --output benchmark/results/native_latest.json
python -m benchmark.run_baselines
python tools/verify_cpp_runtime.py --leaf-infer ./build/leaf_infer.exe
python tools/verify_quantized_runtime.py --leaf-infer ./build/leaf_infer.exe --leaf-bench ./build/leaf_graph_bench.exe
```

The Python/NumPy executor is a numerical oracle. Its latency appears in the
results so Python overhead is visible, but it is not the deployment runtime.

## 6. Real CIFAR-10 benchmark setup

The test data is the University of Toronto CIFAR-10 test split served by its
[Hugging Face dataset mirror](https://huggingface.co/datasets/uoft-cs/cifar10/tree/main/plain_text).
The 23,940,850-byte Parquet file used for this run has SHA-256
`841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de`.
The first 132 original-order test images were prepared locally; the data files
are Git-ignored. Images 0–31 calibrate the small CNN optimizer graph, and
images 32–131 evaluate it. The full ResNet comparison uses the first 20
images. Download and prepare once:

```powershell
New-Item -ItemType Directory -Force benchmark/data | Out-Null
Invoke-WebRequest -Uri 'https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet' -OutFile benchmark/data/cifar10-test.parquet
python -m benchmarks.prepare_cifar10 --samples 132
```

The preparation command needs `pyarrow` and Pillow. Check the downloaded
file's hash if reproducing the exact rows. With the generated NPZ present,
the actual benchmarks run offline:

```powershell
python -m benchmarks.bench_cifar10 --calibration-samples 32 --evaluation-samples 100 --repeats 5 --threads 1
./scripts/build_native.ps1
python -m benchmarks.bench_cifar10_resnet18 --samples 20 --warmup 5 --runs 10 --threads 1
python -m benchmarks.bench_cifar10 --calibration-samples 32 --evaluation-samples 100 --repeats 5 --threads 1 --leaf-infer build/leaf_infer.exe --leaf-bench build/leaf_graph_bench.exe --native-evaluation-samples 20 --output benchmark/results/cifar10_native_int8.json
```

The small CNN graph is an untrained 3→8 Conv/BatchNorm/ReLU optimizer test at
16×16. The full ResNet-18 has deterministic random weights and runs at the
original 32×32 resolution. These runs establish numerical parity and latency
on real images; neither supplies a trained CIFAR classifier, so classification
accuracy is not reported. A trained model and separate held-out accuracy gate
are still needed before making an accuracy claim.

## 7. Qwen2.5-0.5B reference benchmark

The benchmark loads locally cached Qwen2.5-0.5B weights without network
access. The measured snapshot revision is
`060db6499f32faf8b98477b0a26969ef7d8b9987`. It runs in FP32 on one
CPU thread and uses the same fixed 64-token input for two next-token
computations:

1. Recompute all 64 tokens without a KV cache.
2. Prefill the first 63 tokens outside timing and decode the last token with
   the populated KV cache.

It checks maximum logit difference and next-token argmax agreement. Run the
following from the repository root in an environment containing the cached
model:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m benchmarks.bench_qwen25_cached --model Qwen/Qwen2.5-0.5B --threads 1 --sequence-length 64 --warmup 1 --runs 5
```

`--model` also accepts a complete local snapshot directory. The benchmark
does not download weights implicitly. Its output is a PyTorch KV-cache
reference for future Leaf Transformer execution, not a Leaf full-model
measurement.

## 8. Measured results and what each number means

All results below are development-machine observations, not target-machine
promises. Windows runs used an Intel Core i7-14700HX, Windows 11, one CPU
thread, PyTorch 2.12.0+cpu, and GCC 15.2.0 for native code. The Qwen run used
WSL2 on the same host, Python 3.12.3, Transformers 4.57.6, and CPU execution.
Source JSON/CSV files are under
[`benchmark/results`](benchmark/results/README.md).

### 8.1 Real CIFAR-10, full ResNet-18, C++ versus PyTorch

Each image had five warmups and ten measured executions per runtime. The table
reports the median of 20 per-image medians. Both runtimes used the same FP32
seeded model, input images, 32×32 shape, and one thread; model load and process
startup are outside the timed section. A second independent run checked
repeatability.

| Run | PyTorch CPU p50 | Leaf C++ FP32 p50 | Leaf / PyTorch latency | Max absolute logit difference |
|---|---:|---:|---:|---:|
| First | 10.322 ms | 9.483 ms | 0.919× (Leaf 1.09× faster) | 4.62e-7 |
| Repeat | 10.609 ms | 9.701 ms | 0.914× (Leaf 1.09× faster) | 4.62e-7 |

The earlier ten-image pilot varied with host load, so the twenty-image runs
above are the comparison to use. The raw per-image p50 values are preserved in
[`cifar10_resnet18_cpp.json`](benchmark/results/cifar10_resnet18_cpp.json) and
[`cifar10_resnet18_cpp_repeat.json`](benchmark/results/cifar10_resnet18_cpp_repeat.json).
There is no accuracy figure because these are untrained weights.

### 8.2 Real CIFAR-10, calibration and optimizer micrograph

With 32 calibration images and 100 distinct evaluation images, FP32 Leaf/NumPy
versus PyTorch had maximum absolute output error `4.77e-7`; calibrated INT8
simulation versus PyTorch had error `0.02667`. Five repeated sweeps gave the
following median per-image latency:

| Runtime | Latency per image |
|---|---:|
| PyTorch CPU FP32 | 0.098 ms |
| Leaf NumPy FP32 reference | 1.504 ms |
| Leaf NumPy INT8 simulation | 1.469 ms |

These Python reference timings expose interpreter overhead; they are not native
INT8 deployment speed. The complete record is
[`cifar10_cpu.json`](benchmark/results/cifar10_cpu.json).

### 8.3 Real CIFAR-10, native INT8 versus native FP32

Using the same 32 calibration images and a disjoint 100-image evaluation
subset, the optional native harness measured the first 20 evaluation images.
Each graph was warmed five times and timed ten times per image, then the
median of per-image p50 latencies was reported:

| Runtime | Median per-image p50 | Maximum absolute output difference vs PyTorch FP32 |
|---|---:|---:|
| Leaf C++ FP32 | 0.014 ms | 4.77e-7 |
| Leaf C++ INT8 | 0.062 ms | 0.02277 |

This is an untrained optimizer micrograph, not a classification accuracy
result. Native INT8 is currently slower, so FP32 remains the preferred
latency path. The complete environment and methodology are recorded in
[`cifar10_native_int8.json`](benchmark/results/cifar10_native_int8.json).

### 8.4 Native INT8 graph integration check

On this development host, version 2 artifacts matched the quantized Python
reference within `4.77e-7` for the synthetic CNN and `1.72e-7` for the fused
transformer FFN. Relative RMSE against PyTorch FP32 was `1.000%` and `1.076%`.
With ten warmups and 50 timed runs, native p50 latency was:

| Workload | Leaf FP32 | Leaf INT8 | Decision |
|---|---:|---:|---|
| CNN, 1×3×16×16 | 0.014 ms | 0.062 ms | Keep FP32 as latency default |
| FFN, 1×8×16 | 0.004 ms | 0.011 ms | Keep FP32 as latency default |

An 8-output-channel AVX2 INT8 lane improved the synthetic CNN from an
earlier `0.081 ms` p50 to `0.062 ms`; a direct scalar convolution candidate
was measured and rejected at `0.116 ms`. These tiny graphs remain sensitive
to activation quantization, packing, and workspace allocation overhead.
They establish functional INT8 integration,
not a speedup or trained-model quality result. Reproduce with
`tools/verify_quantized_runtime.py`; larger shapes need a separate
same-session whole-model gate before promotion. The record includes peak
process RSS for each native run:
[`native_int8_graph.json`](benchmark/results/native_int8_graph.json).

### 8.5 Qwen cached weights, PyTorch CPU baseline

One warmup and five measured runs on the cached Qwen2.5-0.5B revision produced:

| Computation | p50 latency |
|---|---:|
| Full 64-token recomputation without cache | 763.827 ms |
| Last-token decode with a populated KV cache | 91.862 ms |

The cache made this PyTorch workload `8.315×` faster. Cached and uncached
logits differed by at most `1.67e-5`, and the selected token matched. Model
parameters occupied 1,976,131,072 FP32 bytes; process RSS after loading was
2,686,046,208 bytes. Prefix prefill is deliberately excluded from the cached
single-token timing. See [`qwen25_cached_cpu.json`](benchmark/results/qwen25_cached_cpu.json).

### 8.6 Native kernel speed gate

The latest GCC `-O3 -mavx2 -mfma` run compared each optimized kernel against
Leaf's scalar implementation, after warmup. The GEMM shape was 64×384×384.

| Kernel | Scalar Leaf baseline | Optimized Leaf | Speedup |
|---|---:|---:|---:|
| FP32 GEMM | 6.512 ms | 0.823 ms AVX2 | 7.914× |
| INT8 GEMM | 4.246 ms | 1.123 ms AVX2 | 3.783× |
| FP32 Conv, 16→32, 3×3 | 3.200 ms direct | 0.406 ms im2col+AVX2 | 7.889× |

These are kernel measurements, not full-model speedups. The command uses
`--enforce-speedup` so a slower optimized path fails. Raw values are in
[`native_latest.json`](benchmark/results/native_latest.json).

### 8.7 Prior full ResNet runtime measurements

The 224×224, single-image FP32 ResNet-18 measurements show the effect of
temporary-buffer reuse. Their input shape and WSL2 environment differ from
the CIFAR 32×32 comparison.

| Stage | Mean latency | Throughput | Context |
|---|---:|---:|---|
| AVX2 4×8 GEMM | 140.904 ms | 7.097 images/s | One warmup, ten runs |
| Reuse im2col scratch buffer | 106.812 ms | 9.362 images/s | Three warmups, twenty runs; 24.2% below prior stage |
| Executor buffer pool | 126.314 ms | 7.917 images/s | Separate measurement session |

The last row should be compared with its *same-session* unpatched measurement,
`129.620 → 126.314 ms` (2.55% reduction), not with the earlier 106.812 ms
run. Final-logit parity versus PyTorch remained `3.93e-6`. The data and
measurement notes are in [`wsl_dev_machine.csv`](benchmark/results/wsl_dev_machine.csv)
and [`docs/decisions.md`](docs/decisions.md).

### 8.8 Deterministic synthetic checks

The offline CI workloads cover a small CNN and transformer FFN with fixed
synthetic data. They prove pipeline behavior when real datasets or model
weights are unavailable; they do not stand in for CIFAR classification or
Qwen generation.

| Workload | FP32 max error | INT8 simulated max error | PyTorch CPU | NumPy FP32 | NumPy INT8 simulation |
|---|---:|---:|---:|---:|---:|
| CNN | 5.96e-7 | 0.02034 | 0.0446 ms | 0.6168 ms | 0.6394 ms |
| Transformer FFN | 2.71e-7 | 0.09518 | 0.0154 ms | 0.0215 ms | 0.0329 ms |

The detailed timings and memory plans are in
[`latest.json`](benchmark/results/latest.json). The transformer row is a small
feed-forward graph, not full Qwen.

## 9. Memory-plan format and current runtime reuse

`optimize_graph(graph, calibration_samples, memory_plan_path=...)` exports
`leaf-memory-plan-v1`. A plan records the graph fingerprint, 64-byte
alignment, the arena size versus an allocate-everything size, exact tensor
offsets and live intervals, and any dynamic tensors that could not be planned.

```json
{
  "format": "leaf-memory-plan-v1",
  "graph_fingerprint": "...",
  "alignment": 64,
  "arena_size": 128,
  "naive_size": 192,
  "bytes_saved": 64,
  "allocations": [
    {"tensor": "a", "offset": 0, "size": 64, "first_node": 0, "last_node": 1},
    {"tensor": "c", "offset": 0, "size": 64, "first_node": 2, "last_node": 3}
  ],
  "unplanned_tensors": []
}
```

Two values may share an offset only when their live intervals do not overlap.
The current C++ executor instead has a grow-only, thread-local scratch buffer
for im2col and a best-fit pool of retired activation vectors. It reuses memory
without loading this JSON. Connecting verified plan offsets to a single arena
is an open implementation task. `leaf_graph_bench` now reports peak process
RSS, supplying a baseline for the required before/after arena test; the
synthetic FP32/INT8 process values are saved in
[`native_int8_graph.json`](benchmark/results/native_int8_graph.json). No RSS
improvement from an applied plan is claimed yet.

## 10. Repository map

```text
Leaf/
├── tools/graph_opt/       ONNX IR, folding, fusion, calibration, planner, exporter
├── engine/                C++ graph parser, executor, FP32 and INT8 kernels
├── benchmark/             Synthetic workloads, baseline harness, native gate
│   └── results/           Committed per-run JSON and prior runtime measurements
├── benchmarks/            Real CIFAR and cached Qwen reproduction scripts
├── tests/                 Graph, quantization, integration, and native tests
├── scripts/               Native build and complete verification command
├── docs/decisions.md      Performance decisions and same-session comparisons
└── README.md              Current architecture, methods, measurements, roadmap
```

`benchmark/data/`, exported ONNX files, generated `.leaf` artifacts, and model
weights are intentionally Git-ignored. The repository keeps scripts and
small result records so the large datasets and cached weights stay local.

## 11. Development loop and remaining goals

The next work is iterative implementation, not a final-report phase. For each
candidate: profile the full model, change one bottleneck, verify outputs,
measure same-session latency and RSS, and keep the change only when the result
supports it. The current CIFAR full-model benchmark and native speed gate are
the reference points for the next C++ optimization.

- [x] Load and validate ResNet-18 ONNX as Leaf IR.
- [x] Export and run a self-contained FP32 `.leaf` ResNet artifact in C++.
- [x] Match full ResNet outputs against PyTorch.
- [x] Add constant folding and CNN/Transformer graph rewrites.
- [x] Calibrate symmetric INT8 and convert Conv/Gemm/MatMul weights per output channel.
- [x] Export a versioned, aligned liveness memory plan.
- [x] Add AVX2 FP32/INT8 kernel tests and a no-slowdown native speed gate.
- [x] Measure real CIFAR images against PyTorch and Leaf C++ FP32.
- [x] Measure locally cached Qwen2.5-0.5B PyTorch CPU decode with and without KV cache.
- [x] Store INT8 tensors and scales in `.leaf`, then dispatch supported quantized graphs in C++.
- [x] Provide a portable scalar build alongside the AVX2/FMA build.
- [ ] Apply the exported memory plan directly in the C++ executor and measure peak RSS.
- [ ] Add native transformer norm, attention, KV-cache, and FFN execution; run full Qwen parity and latency.
- [ ] Add trained CIFAR-10 accuracy and Qwen perplexity/next-token quality gates.
- [ ] Evaluate structured pruning and weight-only INT8/INT4 only after those quality gates exist.
- [ ] Profile packing, tiling, threading, and cache behavior; accept only measured whole-model improvements.
- [ ] Reproduce representative final artifacts and benchmarks on additional CPU machines.

Training, GPU deployment, and unstructured sparsity are outside Leaf's scope.
