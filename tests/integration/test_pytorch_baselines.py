import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmark.datasets import transformer_samples, vision_samples
from benchmark.workloads import cnn_workload, transformer_workload
from tools.graph_opt.executor import run_graph
from tools.graph_opt.pipeline import optimize_graph


@pytest.mark.parametrize(
    "factory,samples",
    [(cnn_workload, vision_samples(3)), (transformer_workload, transformer_samples(3))],
)
def test_leaf_pipeline_against_pytorch(factory, samples):
    graph, torch_reference = factory()
    result = optimize_graph(graph, samples)
    for sample in samples:
        with torch.inference_mode():
            expected = torch_reference(torch.from_numpy(sample["x"])).numpy()
        fp32 = run_graph(graph, sample)["y"]
        int8 = run_graph(result.graph, sample)["y"]
        np.testing.assert_allclose(fp32, expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(int8, expected, rtol=0.12, atol=0.08)
