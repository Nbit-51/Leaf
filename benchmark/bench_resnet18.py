"""Legacy Python/NumPy graph-executor baseline, not the native Leaf runtime.

Uses seeded random weights unless --pretrained explicitly requests torchvision's
ImageNet weights (which may download). See bench_cifar10_full for trained native
accuracy and bench_cifar10_resnet18 for native graph timing.
"""
import argparse
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import torchvision.models as models

from tools.graph_opt.ir import Graph
from tools.graph_opt.executor import run_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained', action='store_true', help='explicitly allow downloading ImageNet weights')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if min(args.size, args.runs, args.threads) < 1 or args.warmup < 0:
        parser.error('size, runs and threads must be positive; warmup must be nonnegative')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if args.pretrained else None).eval()
    inputs = torch.randn(1, 3, args.size, args.size)
    with tempfile.TemporaryDirectory(prefix='leaf_resnet_reference_') as directory:
        onnx_path = Path(directory) / 'resnet18.onnx'
        torch.onnx.export(model, inputs, str(onnx_path), input_names=['input'], output_names=['output'],
                          do_constant_folding=True, opset_version=13, dynamo=False)
        graph = Graph.from_onnx(onnx_path)
        input_np = inputs.numpy()
        with torch.inference_mode():
            expected = model(inputs).numpy()
            actual = run_graph(graph, {'input': input_np})['output']
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            timings = {}
            for name, invoke in (('PyTorch eager', lambda: model(inputs)),
                                 ('Leaf Python/NumPy reference', lambda: run_graph(graph, {'input': input_np}))):
                for _ in range(args.warmup):
                    invoke()
                samples = []
                for _ in range(args.runs):
                    start = time.perf_counter()
                    invoke()
                    samples.append((time.perf_counter() - start) * 1000)
                timings[name] = samples
    print('Output parity passed. Weights:', 'ImageNet' if args.pretrained else 'seeded random')
    for name, samples in timings.items():
        print(f'{name}: mean {np.mean(samples):.2f} ms; samples={samples}')
    print('Diagnostic reference-executor timing only; no native speed or classification-accuracy claim.')


if __name__ == '__main__':
    main()
