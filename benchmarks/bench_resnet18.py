import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models

sys.path.insert(0, "tools/graph_opt")
from ir import Graph
from executor import run_graph

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

onnx_path = "/tmp/resnet18_bench.onnx"
torch.onnx.export(
    model, dummy_input, onnx_path,
    input_names=["input"], output_names=["output"],
    do_constant_folding=True, opset_version=13, dynamo=False,
)

graph = Graph.from_onnx(onnx_path)
input_np = dummy_input.numpy()

N_WARMUP = 2
N_RUNS = 10

# PyTorch baseline (eager mode, CPU)
for _ in range(N_WARMUP):
    with torch.no_grad():
        model(dummy_input)

torch_times = []
for _ in range(N_RUNS):
    start = time.perf_counter()
    with torch.no_grad():
        model(dummy_input)
    torch_times.append(time.perf_counter() - start)

# Leaf's Python/numpy reference executor
for _ in range(N_WARMUP):
    run_graph(graph, {"input": input_np})

leaf_times = []
for _ in range(N_RUNS):
    start = time.perf_counter()
    run_graph(graph, {"input": input_np})
    leaf_times.append(time.perf_counter() - start)

torch_mean = np.mean(torch_times) * 1000
torch_std = np.std(torch_times) * 1000
leaf_mean = np.mean(leaf_times) * 1000
leaf_std = np.std(leaf_times) * 1000

print(f"\n=== ResNet-18, single image (1, 3, 224, 224), {N_RUNS} runs after {N_WARMUP} warmup ===")
print(f"PyTorch (eager, CPU):        {torch_mean:8.2f} ms  (± {torch_std:.2f} ms)")
print(f"Leaf reference executor:     {leaf_mean:8.2f} ms  (± {leaf_std:.2f} ms)")
print(f"\nLeaf is {leaf_mean / torch_mean:.1f}x slower than PyTorch eager mode.")
print("(Expected -- Leaf's reference executor is pure Python/numpy loops, not optimized.")
print(" This number is the baseline the C++ AVX2 engine needs to beat, not PyTorch itself.)")
