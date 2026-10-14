# Benchmarks and evidence

This is the single benchmark directory. Run Python entry points from the repository
root with `python -m benchmark.<name>`. Install the repository's validation/dev
dependencies first; datasets and model weights are not part of the Python wheel.

| Location / runner | Purpose |
|---|---|
| `bench_cifar10_full.py` | Trained ResNet-20: held-out classification accuracy, native parity and CPU timing |
| `bench_cifar10.py` | Small untrained CNN: calibration, graph transformations and optional native comparison |
| `bench_cifar10_resnet18.py` | Untrained ResNet-18 on CIFAR inputs: native graph parity and timing |
| `bench_qwen25_cached.py` | Offline PyTorch decoder baseline with/without KV cache; not a Leaf comparison |
| `bench_resnet18.py` | Legacy Python/NumPy graph executor; no native performance claim; optional explicit pretrained download |
| `prepare_cifar10.py` | Convert an existing CIFAR Parquet file to a small offline NPZ fixture |
| `run_baselines.py`, `workloads.py`, `datasets.py` | Shared deterministic synthetic workloads and Python graph baselines |
| `*.cpp` | Isolated native kernel and operator benchmark programs |
| `llama_decoder_benchmark.cpp` | Pinned llama.cpp adapter matching Leaf's last-logit prefill and one cached decode step; orchestrated by `tools/benchmark_decoder_competitor.py` |
| `data/` | Local downloaded datasets/checkpoints; ignored by Git |
| `results/` | Retained measurements, provenance and failures; consult the associated documentation before comparing |
| `experiments/` | Preserved experimental patches and implementations |

For example:

```sh
python -m benchmark.prepare_cifar10 --help
python -m benchmark.bench_cifar10_full --help
python -m benchmark.bench_qwen25_cached --help
```

Shared validation and experiment orchestration remain in `tools/`; build/staged
verification commands remain in `scripts/`; automated regression tests remain in
`tests/`. A benchmark measures a workload; a test checks an implementation contract.

The former `benchmarks/` runners moved here. Update old commands from
`python -m benchmarks.<runner>` to `python -m benchmark.<runner>`.
Existing `benchmark/data/` and result paths are unchanged. Historical JSON records
remain untouched even if they contain an older command or source path.

See the [documentation index](../docs/README.md) for qualified releases and
[model/output checks](../docs/model-onboarding.md) for correctness-only onboarding.
