# Benchmark result records

The result files are measurements from specific development environments.
Compare numbers only when the workload, shape, precision, thread count, and
machine match. The large source datasets and model weights remain local.

| File | What it records |
|---|---|
| `native_latest.json` | Scalar Leaf versus AVX2 Leaf FP32/INT8 GEMM and Conv kernel speed gate |
| `latest.json` | Deterministic synthetic CNN/FFN calibration, PyTorch parity, Python reference timings, and memory plans |
| `cifar10_cpu.json` | Real CIFAR-10 images on the small optimizer CNN graph; PyTorch versus NumPy FP32/INT8 simulation |
| `cifar10_resnet18_cpp.json` | First 20 real CIFAR-10 test images, full seeded FP32 ResNet-18, PyTorch versus C++ |
| `cifar10_resnet18_cpp_repeat.json` | Independent repeat of the same full-model comparison |
| `qwen25_cached_cpu.json` | Offline cached Qwen2.5-0.5B PyTorch CPU decode, full-context recomputation versus populated KV cache |
| `wsl_dev_machine.csv` | Historical 224×224 FP32 ResNet-18 runtime stages under WSL; different sessions are labeled |
| `memory_plans/` | JSON liveness plans for the deterministic CNN and FFN workloads |

`cifar10_resnet18_cpp*.json` uses untrained model weights and measures parity
and latency, not classifier accuracy. `qwen25_cached_cpu.json` is a PyTorch
reference for KV-cache behavior; Leaf does not yet execute the full model.

The README contains reproduction commands and a table of the important
numbers. Run those commands again on the deployment CPU before drawing
machine-independent performance conclusions.
