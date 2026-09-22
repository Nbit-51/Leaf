from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from tools.graph_opt.ir import Graph, Node, TensorInfo


def cnn_workload(seed: int = 300):
    rng = np.random.default_rng(seed)
    weight = rng.normal(0, 0.12, (8, 3, 3, 3)).astype(np.float32)
    bias = rng.normal(0, 0.03, (8,)).astype(np.float32)
    scale = rng.uniform(0.7, 1.2, (8,)).astype(np.float32)
    bn_bias = rng.normal(0, 0.04, (8,)).astype(np.float32)
    mean = rng.normal(0, 0.1, (8,)).astype(np.float32)
    variance = rng.uniform(0.7, 1.3, (8,)).astype(np.float32)
    graph = Graph()
    graph.inputs, graph.outputs = ["x"], ["y"]
    graph.initializers = {
        "weight": weight, "bias": bias, "scale": scale,
        "bn_bias": bn_bias, "mean": mean, "variance": variance,
    }
    graph.nodes = [
        Node("conv", "Conv", ["x", "weight", "bias"], ["conv_out"], {"pads": [1, 1, 1, 1]}),
        Node("batchnorm", "BatchNormalization",
             ["conv_out", "scale", "bn_bias", "mean", "variance"], ["bn_out"], {"epsilon": 1e-5}),
        Node("relu", "Relu", ["bn_out"], ["y"]),
    ]
    graph.value_info = {
        name: TensorInfo(name, shape, "float32")
        for name, shape in {
            "x": (1, 3, 16, 16), "conv_out": (1, 8, 16, 16),
            "bn_out": (1, 8, 16, 16), "y": (1, 8, 16, 16),
        }.items()
    }

    def torch_reference(value):
        output = functional.conv2d(value, torch.from_numpy(weight), torch.from_numpy(bias), padding=1)
        output = functional.batch_norm(
            output, torch.from_numpy(mean), torch.from_numpy(variance),
            torch.from_numpy(scale), torch.from_numpy(bn_bias), training=False, eps=1e-5,
        )
        return functional.relu(output)

    return graph, torch_reference


def transformer_workload(seed: int = 400):
    rng = np.random.default_rng(seed)
    weight = rng.normal(0, 0.14, (16, 32)).astype(np.float32)
    bias = rng.normal(0, 0.04, (32,)).astype(np.float32)
    graph = Graph()
    graph.inputs, graph.outputs = ["x"], ["y"]
    graph.initializers = {"weight": weight, "bias": bias}
    graph.nodes = [
        Node("linear", "MatMul", ["x", "weight"], ["linear_out"]),
        Node("bias", "Add", ["linear_out", "bias"], ["biased"]),
        Node("gelu", "Gelu", ["biased"], ["y"]),
    ]
    graph.value_info = {
        name: TensorInfo(name, shape, "float32")
        for name, shape in {
            "x": (1, 8, 16), "linear_out": (1, 8, 32),
            "biased": (1, 8, 32), "y": (1, 8, 32),
        }.items()
    }

    def torch_reference(value):
        return functional.gelu(
            torch.matmul(value, torch.from_numpy(weight)) + torch.from_numpy(bias), approximate="tanh"
        )

    return graph, torch_reference
