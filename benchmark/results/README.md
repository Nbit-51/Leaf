# Benchmark result records

The result files are measurements from specific development environments.
Compare numbers only when the workload, shape, precision, thread count, and
machine match. The large source datasets and model weights remain local.

| File | What it records |
|---|---|
| `native_latest.json` | Scalar Leaf versus AVX2 Leaf FP32/INT8 GEMM and Conv kernel speed gate |
| `latest.json` | Deterministic synthetic CNN/FFN calibration, PyTorch parity, Python reference timings, and memory plans |
| `cifar10_cpu.json` | Real CIFAR-10 images on the small optimizer CNN graph; PyTorch versus NumPy FP32/INT8 simulation |
| `cifar10_native_int8.json` | Same real-image subset with native FP32 and native INT8 graph parity and latency |
| `native_int8_graph.json` | Synthetic native INT8 graph parity, FP32 comparison, latency, and peak process RSS |
| `memory_plan_native.json` | Embedded version 3 arena versus version 2 buffer-pool ResNet parity, latency, and peak process RSS |
| `native_transformer_ops.json` | RMSNorm PyTorch parity and portable/AVX2 native latency |
| `native_attention.json` | Masked attention prefill/decode PyTorch parity and portable/AVX2 native latency |
| `native_swiglu.json` | Fused versus unfused native SwiGLU FFN parity and order-alternated latency |
| `cifar10_resnet18_cpp.json` | First 20 real CIFAR-10 test images, full seeded FP32 ResNet-18, PyTorch versus C++ |
| `cifar10_resnet18_cpp_repeat.json` | Independent repeat of the same full-model comparison |
| `cifar10_resnet18_cpp_transformer_update.json` | Full ResNet real-image PyTorch/C++ parity and latency after native Transformer additions |
| `cifar10_resnet18_transformer_no_regression.json` | Same-session alternating pre-change/current C++ ResNet latency check |
| `qwen25_cached_cpu.json` | Offline cached Qwen2.5-0.5B PyTorch CPU decode, full-context recomputation versus populated KV cache |
| `wsl_dev_machine.csv` | Historical 224×224 FP32 ResNet-18 runtime stages under WSL; different sessions are labeled |
| `memory_plans/` | JSON liveness plans for the deterministic CNN and FFN workloads |

`cifar10_resnet18_cpp*.json` uses untrained model weights and measures parity
and latency, not classifier accuracy. `qwen25_cached_cpu.json` is a PyTorch
reference for KV-cache behavior; Leaf does not yet execute the full model.

The README contains reproduction commands and a table of the important
numbers. Run those commands again on the deployment CPU before drawing
machine-independent performance conclusions. INT8 graph support is opt-in;
the current whole-graph measurements do not show a latency benefit.
