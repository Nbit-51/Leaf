# Trained CNNs against ONNX Runtime

Leaf and ONNX Runtime 1.31.0 ran the same trained FP32 ONNX graphs on one pinned
Core i7-14700HX P-core in Windows Better Battery mode. Leaf imported each graph,
folded constants and exported the native artifact used by its graph executor.
ONNX Runtime used `CPUExecutionProvider`, all graph optimizations, and one intra-op
and one inter-op thread.

| Network | Held-out images | Top-1 accuracy, both runtimes | Leaf warmed p50 | ONNX Runtime warmed p50 | Timing status |
| --- | ---: | ---: | ---: | ---: | --- |
| ResNet-20 | 10,000 | 92.60% | 18.6965 ms | 0.9683 ms | Diagnostic |
| VGG-11-BN | 512 | 92.77% | 58.7236 ms | 4.5377 ms | Diagnostic |

On both models, both runtimes exactly matched every PyTorch reference prediction
and every logit passed `atol=1e-3, rtol=1e-3`. The largest absolute Leaf errors
were `2.01e-5` for ResNet and `1.58e-5` for VGG. The ResNet evaluation used the
entire CIFAR-10 test split. VGG used a deterministic 512-image subset from that
split and a published CIFAR-10 checkpoint.
[VGG-11-BN checkpoint and architecture](https://github.com/chenyaofo/pytorch-cifar-models/blob/master/pytorch_cifar_models/vgg.py)

Timing used the same 101 fixed images for each paired pass, six alternating
Leaf-first and ONNX-Runtime-first pairs, and 20 warmups per pass. The timed
boundaries were Leaf's `Executor.run` and ONNX Runtime's `session.run`. Both
exclude input staging, loading and warmup. The raw report also records Leaf's
offline import/export time and ONNX Runtime's session setup time. These setup
measurements describe different operations. The spread exceeded 1.25× in both
comparisons, so the table records observed medians only.

The checked-in runner takes a fixed-batch FP32 ONNX file, raw FP32 images, FP32
reference logits, uint32 labels and the native `leaf_dataset` executable. Run
`python -m benchmark.bench_graph_onnxruntime --help` in an environment with
NumPy, ONNX and the pinned ONNX Runtime wheel installed.
[Runner](../benchmark/bench_graph_onnxruntime.py) ·
[Raw samples and provenance](../benchmark/validation/windows-fp32-20261115/graphs/manifest.json)
