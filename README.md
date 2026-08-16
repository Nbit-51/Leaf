# Kestrel

**A CPU-native neural network inference optimization engine — making complex models runnable on hardware without a GPU.**

> Name is a working title (easy to rename via find-replace across the repo if the team lands on something else — LEAN was the runner-up).

---

## 1. Problem Statement

Access to machine learning is gated by hardware. State-of-the-art models assume a GPU is available for both training and inference, which excludes students, hobbyists, and researchers on constrained laptops or low-resource machines. Kestrel's goal is to close that gap on the **inference** side: take an already-trained network and transform it — through quantization, structured pruning, graph-level optimization, and hand-written CPU kernels — into a form that runs with acceptable latency and memory footprint on a CPU-only machine with limited RAM and storage.

This is not a training framework. Kestrel assumes a model already exists (trained on any machine, GPU or not) and focuses entirely on making it efficient to *run*.

## 2. Goals

- Support both CNN (vision) and Transformer (language) architectures through a shared intermediate representation (IR).
- Reduce model size via INT8 (CNN) and INT4 weight-only (Transformer) quantization.
- Reduce compute and memory via structured pruning (channel pruning for CNNs, attention-head/FFN-neuron pruning for transformers).
- Reduce redundant computation via graph-level optimization: constant folding, operator fusion (Conv+BN+ReLU, Linear+Activation), and memory-reuse planning.
- Recover raw throughput lost to abstraction via custom AVX2 CPU kernels for the hot paths (INT8 GEMM, convolution, attention).
- Validate that the entire pipeline runs end-to-end on a **6 GB storage, no-GPU** machine — not just the primary dev machine.
- Produce a benchmark harness that isolates the contribution of each optimization stage (baseline → +quant → +pruning → +fusion → +custom kernels) on latency, peak memory, and accuracy/perplexity delta.

## 3. Non-Goals

- Training or fine-tuning models. Kestrel consumes already-trained weights.
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
        │  Kestrel Graph IR                │          │                              │
        │         │                        │          │                              │
        │   ┌─────┼─────┬─────────┐        │          │                              │
        │   ▼     ▼     ▼         ▼        │          │                              │
        │ Fuse  Quant Prune  Mem-Plan       │  ships   │  Kestrel Runtime (C++)       │
        │   └─────┴─────┴─────────┘   ─────┼─────────▶│  + compiled AVX2 kernels     │
        │         │                        │ artifact │         │                     │
        │         ▼                        │ (.kest)  │         ▼                     │
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

## 5. Test Models

| Domain | Model | Format | Approx. size |
|---|---|---|---|
| CNN | ResNet-18 (CIFAR-10) | FP32 → INT8 | ~44 MB → ~11 MB |
| Transformer | Qwen2.5-0.5B or SmolLM2-360M | GGUF-style Q4 | ~250–400 MB |

Datasets kept small deliberately: a few hundred CIFAR-10 images (not the full 170 MB set) and a fixed prompt set for the language model, both used only for calibration and benchmarking, not training.

## 6. File Structure

```
kestrel/
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
│   ├── include/kestrel/
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
│   └── exported/                   # generated .kest artifacts (gitignored)
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

## 7. Target-Machine Footprint

The constraint driving the whole design: the target machine must be able to clone, build, and run the full benchmark suite within 6 GB total.

| Component | Approx. size | Needed on target? |
|---|---|---|
| Kestrel runtime (compiled binary) | a few MB | Yes |
| Compiler toolchain (g++/clang) | usually preinstalled | Yes (build step) |
| ONNX Runtime (CPU, optional — reference cross-check only) | ~100–150 MB | Optional |
| ResNet-18 INT8 artifact | ~11 MB | Yes |
| Qwen2.5-0.5B / SmolLM2-360M Q4 artifact | ~250–400 MB | Yes |
| CIFAR-10 subset + prompt set | well under 100 MB | Yes |
| PyTorch / training framework | 800 MB – 3+ GB | **No — never installed on target** |

## 8. Roles (proposed split)

| Area | Owner | Covers |
|---|---|---|
| Graph IR + graph-level optimizer | TBD | Sections 4.1, 4.2 |
| Quantization + pruning | TBD | Sections 4.3, 4.4 |
| AVX2 kernels + runtime | TBD | Sections 4.5, 4.6 |

Fill in owners once the team divides the work — the three-way split above maps roughly to the three optimization pillars, so each person can own a vertical slice end-to-end (IR pass → kernel → benchmark) rather than everyone touching the same files.

## 9. Milestones

- [ ] **M1** — Graph IR loads ONNX for both ResNet-18 and the chosen transformer; round-trips without loss.
- [ ] **M2** — Constant folding + fusion passes pass unit tests on both model types.
- [ ] **M3** — INT8 CNN quantization + INT4 transformer weight quantization produce correct (accuracy-validated) outputs.
- [ ] **M4** — Structured pruning integrated; accuracy/perplexity degradation logged.
- [ ] **M5** — AVX2 kernels replace naive ops; benchmark harness shows measurable latency improvement over the fp32 baseline.
- [ ] **M6** — Full pipeline runs end-to-end on the target 6 GB / no-GPU machine; results reproduced independently by teammate.
- [ ] **M7** — Report + benchmark writeup finalized for submission.

## 10. Benchmark Methodology

At each pipeline stage (baseline → +quant → +pruning → +fusion → +custom kernels), measure and log:

- Peak resident memory (RSS)
- Single-thread and multi-thread latency
- Accuracy delta (CNN) or perplexity delta (transformer) vs. fp32 baseline

Results are recorded per-machine in `benchmark/results/`, so dev-machine and target-machine numbers can be compared directly — this cross-machine comparison is the core evidence for the "removes the hardware constraint" claim.

### 10.1 PyTorch CPU Baseline

PyTorch is not part of the Kestrel runtime and never ships to the target machine — its CPU backend (oneDNN) is an industry-grade optimized engine, and pulling it into the runtime would defeat the purpose of building the optimization stack from scratch. Instead, it's used as an external reference point, run once on the dev machine:

- Run the same fp32 model through standard `torch` CPU inference (no `torch.compile`, to keep it a plain baseline) and log latency + peak RSS alongside the Kestrel pipeline stages.
- Optionally add a second row with `torch.compile` enabled, since its fusion/graph-optimization approach is a useful point of comparison against Kestrel's own graph-level optimizer (section 4.2).
- This gives the report an honest, industry-standard baseline rather than only comparing Kestrel's stages against its own fp32 starting point — and if Kestrel approaches PyTorch's latency with a fraction of the install footprint and no framework dependency, that's the headline result.

This comparison only needs to run on the dev machine (which already has PyTorch installed for export/training); it is not part of the target-machine test suite.

## 11. Building

```
# Target machine (runtime only, no PyTorch required)
git clone <repo-url>
cd kestrel
cmake -B build engine/
cmake --build build
./build/kestrel_bench --model models/exported/resnet18_int8.kest
```

Full dev-machine setup (model export, quantization, pruning) is documented separately in `docs/architecture.md` once the `tools/` scripts are in place.
