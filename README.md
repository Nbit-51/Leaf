# Leaf

**A CPU-native neural network inference optimization engine — making complex models runnable on hardware without a GPU.**


## 1. Problem Statement

Access to machine learning is gated by hardware. State-of-the-art models assume a GPU is available for both training and inference, which excludes students, hobbyists, and researchers on constrained laptops or low-resource machines. Leaf's goal is to close that gap on the **inference** side: take an already-trained network and transform it — through quantization, structured pruning, graph-level optimization, and hand-written CPU kernels — into a form that runs with acceptable latency and memory footprint on a CPU-only machine with limited RAM and storage.

This is not a training framework. Leaf assumes a model already exists (trained on any machine, GPU or not) and focuses entirely on making it efficient to *run*.

## 2. Goals

- Support both CNN (vision) and Transformer (language) architectures through a shared intermediate representation (IR).
- Reduce model size via INT8 (CNN) and INT4 weight-only (Transformer) quantization.
- Reduce compute and memory via structured pruning (channel pruning for CNNs, attention-head/FFN-neuron pruning for transformers).
- Reduce redundant computation via graph-level optimization: constant folding, operator fusion (Conv+BN+ReLU, Linear+Activation), and memory-reuse planning.
- Recover raw throughput lost to abstraction via custom AVX2 CPU kernels for the hot paths (INT8 GEMM, convolution, attention).
- Validate that the entire pipeline runs end-to-end on a **6 GB storage, no-GPU** machine — not just the primary dev machine.
- Produce a benchmark harness that isolates the contribution of each optimization stage (baseline → +quant → +pruning → +fusion → +custom kernels) on latency, peak memory, and accuracy/perplexity delta.

## 3. Non-Goals

- Training or fine-tuning models. Leaf consumes already-trained weights.
- GPU support of any kind. This is intentionally CPU-only, since that constraint *is* the project.
- Supporting arbitrary/unbounded model architectures — the shared IR targets the operator set needed for standard CNNs and decoder-only transformers, not every possible layer type.

## 4. Architecture

Two machines are involved, with a hard separation between them:

- **Dev machine** (has a GPU, more storage): exports trained models to ONNX, runs calibration for quantization, produces the final optimized artifacts.
- **Target machine** (6 GB storage, no GPU): only builds and runs the C++ runtime against the pre-exported artifacts. Never installs PyTorch or any training framework.

```
                    DEV MACHINE                                TARGET MACHINE
        ┌─────────────────────────────────┐          ┌──────────────────────────────┐
        │  Trained model (PyTorch)         │          │                              │
        │         │                        │          │                              │
        │         ▼                        │          │                              │
        │  ONNX Export                     │          │                              │
        │         │                        │          │                              │
        │         ▼                        │          │                              │
        │  Leaf Graph IR                   │          │                              │
        │         │                        │          │                              │
        │   ┌─────┼─────┬─────────┐        │          │                              │
        │   ▼     ▼     ▼         ▼        │          │                              │
        │ Fuse  Quant Prune  Mem-Plan       │  ships   │  Leaf Runtime (C++)           │
        │   └─────┴─────┴─────────┘   ─────┼─────────▶│  + compiled AVX2 kernels     │
        │         │                        │ artifact │         │                     │
        │         ▼                        │ (.leaf)  │         ▼                     │
        │  Serialized optimized model      │          │  Inference + Benchmark        │
        └─────────────────────────────────┘          └──────────────────────────────┘
```

### 4.1 Graph IR

A minimal node/edge graph loaded from ONNX — ops, tensor shapes, dtypes. Shared substrate for every optimization pass below, so passes compose instead of each reimplementing graph traversal.

### 4.2 Graph-Level Optimizer

- Constant folding
- Operator fusion: Conv + BatchNorm + ReLU (CNN), Linear + GELU/SiLU (Transformer FFN blocks)
- Liveness analysis for buffer reuse — this is the primary lever for peak memory reduction, independent of quantization.

### 4.3 Quantization

- CNN: post-training static INT8, calibrated on a small held-out batch.
- Transformer: weight-only INT4 (round-to-nearest baseline; GPTQ-style as a stretch goal).

### 4.4 Pruning

- Structured only (channel pruning for CNNs, head/neuron pruning for transformers). Unstructured sparsity is explicitly out of scope — it gives no real speedup on general-purpose CPUs without dedicated sparse GEMM kernels.

### 4.5 Custom CPU Kernels

- AVX2 intrinsics for INT8 GEMM
- Direct or im2col convolution kernel
- Cache-tiled fused attention kernel for the transformer path

### 4.6 Runtime

Lightweight C++ executor that loads a serialized optimized graph and runs it using the kernels above. No ML framework dependency — this is what keeps the target machine's footprint small.

## 5. Current Implementation Status

Leaf now has a verified FP32 vertical slice for ResNet-18:

- Python loads ONNX into Leaf IR, applies Conv/BatchNorm/ReLU fusion, and exports a self-contained `.leaf` artifact.
- The C++ runtime parses that artifact and executes the ResNet operator subset: Conv (including exported fused bias), ReLU, Add, MaxPool, GlobalAveragePool, Flatten, Gemm, and unfused BatchNorm.
- Runtime weights are referenced directly from the loaded artifact. Intermediate activations are reclaimed after their final consumer, avoiding an additional model-sized weight copy and retaining only live activation buffers.
- `leaf_infer` runs a raw float32 input; `leaf_bench` measures repeatable inference latency.

### Verification

- C++ unit tests cover AVX2 GEMM (including alpha/beta, vector tails, and tiled paths), im2col boundaries, and an end-to-end synthetic graph.
- A real eval-mode ResNet-18 exported from PyTorch matches Leaf's final 1,000 logits with a maximum absolute difference of `3.93e-06`.
- The ResNet stem Conv direct implementation, im2col path, and AVX2 GEMM agree with the Python reference within `1.91e-06`.

## 6. Test Models

| Domain | Model | Format | Approx. size |
|---|---|---|---|
| CNN | ResNet-18 (CIFAR-10) | FP32 → INT8 | ~44 MB → ~11 MB |
| Transformer | Qwen2.5-0.5B or SmolLM2-360M | GGUF-style Q4 | ~250–400 MB |

Datasets kept small deliberately: a few hundred CIFAR-10 images (not the full 170 MB set) and a fixed prompt set for the language model, both used only for calibration and benchmarking, not training.

## 7. File Structure

```
leaf/
├── README.md
├── LICENSE
├── CMakeLists.txt                 # top-level build
├── .gitignore
│
├── docs/
│   ├── architecture.md            # expanded version of section 4
│   ├── benchmarks.md              # results log, updated per milestone
│   └── decisions.md               # ADR-style log of design tradeoffs
│
├── engine/                        # C++ runtime — ships to the target machine
│   ├── include/leaf/
│   │   ├── ir/                    # graph IR headers
│   │   ├── kernels/                # kernel interfaces
│   │   └── runtime/                # executor, memory planner
│   ├── src/
│   │   ├── ir/
│   │   ├── kernels/
│   │   │   └── avx2/               # AVX2 intrinsic implementations
│   │   └── runtime/
│   └── CMakeLists.txt
│
├── tools/                          # dev-machine only — never needed on target
│   ├── export/                     # PyTorch -> ONNX export scripts
│   ├── quantize/                   # INT8 / INT4 calibration + conversion
│   ├── prune/                      # structured pruning scripts
│   └── graph_opt/                  # fusion / const-fold / mem-plan (Python prototype
│                                    # before porting logic into engine/)
│
├── models/
│   ├── configs/                    # per-model export/quant configs (yaml/json)
│   └── exported/                   # generated .leaf artifacts (gitignored)
│
├── benchmark/
│   ├── harness/                    # latency, peak-RSS, accuracy/perplexity measurement
│   ├── datasets/                   # small fixed CIFAR-10 subset + prompt set
│   └── results/                    # csv/json output per run, per machine
│
├── tests/
│   ├── unit/                       # per-pass and per-kernel correctness tests
│   └── integration/                # full pipeline, fp32 baseline vs optimized output
│
└── scripts/
    ├── build.sh / build.ps1        # single-command build for the target machine
    └── setup_dev_env.ps1           # dev-machine only environment setup
```

## 8. Target-Machine Footprint

The constraint driving the whole design: the target machine must be able to clone, build, and run the full benchmark suite within 6 GB total.

| Component | Approx. size | Needed on target? |
|---|---|---|
| Leaf runtime (compiled binary) | a few MB | Yes |
| Compiler toolchain (g++/clang) | usually preinstalled | Yes (build step) |
| ONNX Runtime (CPU, optional — reference cross-check only) | ~100–150 MB | Optional |
| ResNet-18 INT8 artifact | ~11 MB | Yes |
| Qwen2.5-0.5B / SmolLM2-360M Q4 artifact | ~250–400 MB | Yes |
| CIFAR-10 subset + prompt set | well under 100 MB | Yes |
| PyTorch / training framework | 800 MB – 3+ GB | **No — never installed on target** |

## 9. Roles (proposed split)

| Area | Owner | Covers |
|---|---|---|
| Graph IR + graph-level optimizer | TBD | Sections 4.1, 4.2 |
| Quantization + pruning | TBD | Sections 4.3, 4.4 |
| AVX2 kernels + runtime | TBD | Sections 4.5, 4.6 |

Fill in owners once the team divides the work — the three-way split above maps roughly to the three optimization pillars, so each person can own a vertical slice end-to-end (IR pass → kernel → benchmark) rather than everyone touching the same files.

## 10. Milestones

- [x] **M1a** — ResNet-18 ONNX → Leaf artifact round-trip and independent C++ parser verification.
- [x] **M1b** — FP32 C++ ResNet-18 executor matches PyTorch final logits (`max abs diff: 3.93e-06`).
- [x] **M2a** — Python Conv/BatchNorm/ReLU fusion passes numerical tests.
- [x] **M5a** — AVX2 FP32 GEMM and im2col Conv are in the runtime; the 4x8 GEMM micro-kernel improved the measured ResNet-18 mean latency from `397.428 ms` to `140.904 ms` on the development WSL machine.
- [ ] **M2b** — Add constant folding, memory-plan export, and Transformer graph rewrites.
- [ ] **M3a** — Add calibrated per-channel INT8 CNN weights/activations and an AVX2 INT8 Conv/GEMM path.
- [ ] **M3b** — Add Transformer weight-only INT8 first, then evaluate INT4; preserve FP32 for norms, softmax, residual paths, and the initial KV cache implementation.
- [ ] **M4** — Add structured pruning with accuracy/perplexity gates.
- [ ] **M5b** — Benchmark multi-threading, packed/tiled GEMM, streaming im2col, fusion, and memory reuse against the FP32 baseline.
- [ ] **M6** — Reproduce the complete pipeline on the 6 GB/no-GPU target machine.
- [ ] **M7** — Final report and benchmark write-up.

## 11. Benchmark Methodology

At each pipeline stage (baseline → +quant → +pruning → +fusion → +custom kernels), measure and log:

- Peak resident memory (RSS)
- Single-thread and multi-thread latency
- Accuracy delta (CNN) or perplexity delta (transformer) vs. fp32 baseline

Results are recorded per-machine in `benchmark/results/`, so dev-machine and target-machine numbers can be compared directly — this cross-machine comparison is the core evidence for the "removes the hardware constraint" claim.

### Current FP32 Result

On the development WSL machine, single-image ResNet-18 (`1x3x224x224`) with one warm-up and ten measured runs reports:

| Runtime | Mean latency | Throughput | Numerical check |
|---|---:|---:|---|
| Leaf FP32 AVX2 runtime | 140.904 ms | 7.097 images/s | PyTorch max abs diff `3.93e-06` |

This is a development baseline, not a cross-machine claim. Repeat the benchmark on an otherwise idle machine, use more warm-up/runs for published values, and record CPU model, compiler, thread count, and peak RSS.

### Precision-Safe Optimization Queue

The next performance work should preserve the verified FP32 result before quantization is introduced:

1. Pack and cache Conv/GEMM weights for the micro-kernel, avoiding repeated layout work.
2. Stream or tile im2col instead of materializing each full patch matrix.
3. Reuse preallocated activation/workspace buffers from a liveness-based memory plan.
4. Parallelize independent output-channel tiles, with deterministic per-output accumulation.
5. Fuse Conv+bias+ReLU and common linear activations to eliminate intermediate writes.
6. Add runtime CPU-feature dispatch so AVX2/FMA is selected when available and a scalar fallback remains correct.

Quantization is a separate, accuracy-gated phase. CNN INT8 should use calibration and per-channel scales. For LLMs, begin with weight-only INT8 while keeping activations and KV cache high precision; evaluate long prompts with perplexity, next-token agreement, and retrieval tests before attempting W8A8, KV-cache quantization, or INT4.

### 10.1 PyTorch CPU Baseline

PyTorch is not part of the Leaf runtime and never ships to the target machine — its CPU backend (oneDNN) is an industry-grade optimized engine, and pulling it into the runtime would defeat the purpose of building the optimization stack from scratch. Instead, it's used as an external reference point, run once on the dev machine:

- Run the same fp32 model through standard `torch` CPU inference (no `torch.compile`, to keep it a plain baseline) and log latency + peak RSS alongside the Leaf pipeline stages.
- Optionally add a second row with `torch.compile` enabled, since its fusion/graph-optimization approach is a useful point of comparison against Leaf's own graph-level optimizer (section 4.2).
- This gives the report an honest, industry-standard baseline rather than only comparing Leaf's stages against its own fp32 starting point — and if Leaf approaches PyTorch's latency with a fraction of the install footprint and no framework dependency, that's the headline result.

This comparison only needs to run on the dev machine (which already has PyTorch installed for export/training); it is not part of the target-machine test suite.

## 12. Building

```
# Target machine (runtime only, no PyTorch required)
git clone https://github.com/Nbit-51/Leaf.git
cd Leaf
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build -j
ctest --test-dir engine/build --output-on-failure

# FP32 artifact inference
./engine/build/leaf_infer model.leaf input.bin 1,3,224,224 output.bin

# FP32 latency measurement
./engine/build/leaf_bench model.leaf input.bin 1,3,224,224 3 20
```

Full dev-machine setup (model export, quantization, pruning) is documented separately in `docs/architecture.md` once the `tools/` scripts are in place.

What is AI inference?
AI inference is the "doing" part of artificial intelligence. It's the moment a trained model stops learning and starts working, turning its knowledge into real-world results.

AI Training
    ↓
Model learns weights
    ↓
Fine-tuning
    ↓
Adapt model for a specific task
    ↓
AI Inference
    ↓
Give input → model predicts
    ↓
Inference Serving
    ↓
Make that prediction available to users/apps

Our project is mainly here:
                 AI MODEL
                    ↓
              ┌───────────┐
              │   LEAF    │
              └───────────┘
                    ↓
             Optimize model
                    ↓
        ┌───────────┬───────────┐
        ↓           ↓           ↓
   Quantization   Pruning    Graph Opt.
        └───────────┬───────────┘
                    ↓
             Efficient Model
                    ↓
              CPU Inference
                    ↓
              Prediction
