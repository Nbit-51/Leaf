"""Compare warmed single-image FP32 classification on Leaf and ONNX Runtime."""

import argparse
import json
from pathlib import Path
from statistics import median
import subprocess
from time import perf_counter_ns

import numpy as np

from leaf.affinity import pin_cpu
from leaf.evaluation import write_record
from leaf.power import WINDOWS_POWER_MODE_NAMES, benchmark_power_mode_guard, keep_awake, require_benchmark_ac
from tools.decoder_validation import latency_stability
from tools.graph_opt.constant_folding import fold_constants
from tools.graph_opt.export_binary import export_graph
from tools.graph_opt.fusion import run_fusion_passes
from tools.graph_opt.ir import Graph
from tools.validate_decoder import digest


def ort_run(args, images, expected, labels, *, timing):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    start = perf_counter_ns()
    session = ort.InferenceSession(str(args.onnx), options, providers=["CPUExecutionProvider"])
    setup_ms = (perf_counter_ns() - start) / 1e6
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or len(outputs) != 1 or inputs[0].type != "tensor(float)" or
            outputs[0].type != "tensor(float)" or session.get_providers() != ["CPUExecutionProvider"]):
        raise ValueError("Expected one FP32 input and output on ONNX Runtime's CPU provider")
    buffer = np.empty_like(images[:1])
    feed = {inputs[0].name: buffer}
    output_names = [outputs[0].name]
    np.copyto(buffer[0], images[0])
    for _ in range(20):
        session.run(output_names, feed)
    samples = []
    logits = np.empty_like(expected)
    for i, image in enumerate(images):
        np.copyto(buffer[0], image)
        start = perf_counter_ns()
        result, = session.run(output_names, feed)
        elapsed = (perf_counter_ns() - start) / 1e6
        if result.shape != (1, expected.shape[1]) or result.dtype != np.float32:
            raise ValueError("ONNX Runtime output differs from the reference shape or precision")
        logits[i] = result[0]
        if timing:
            samples.append(elapsed)
    predictions = logits.argmax(axis=1)
    differences = np.abs(logits - expected)
    return dict(
        accuracy=float(np.mean(predictions == labels)),
        prediction_agreement=float(np.mean(predictions == expected.argmax(axis=1))),
        max_abs_error=float(differences.max()),
        parity_failures=int((~np.isclose(logits, expected, atol=1e-3, rtol=1e-3)).any(axis=1).sum()),
        session_setup_ms=setup_ms,
        latency_samples_ms=samples,
    )


def leaf_run(args, dataset, artifact, output):
    inputs, reference, labels, count = dataset
    subprocess.run([str(args.executable.resolve()), str(artifact), str(inputs), str(reference), str(labels),
                    "1," + ",".join(str(value) for value in args.shape), str(count), str(output)],
                   check=True, stdout=subprocess.DEVNULL)
    return json.loads(output.read_bytes())


def require_parity(report, count):
    if report["parity_failures"] or report["prediction_agreement"] != 1.0:
        raise ValueError(f"Logit or prediction parity failed for {count} held-out inputs")


def main():
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("onnx", "inputs", "reference", "labels", "executable", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=3, default=(3, 32, 32), metavar=("C", "H", "W"))
    parser.add_argument("--samples", type=int, default=101, help="Fixed held-out inputs per timed pass")
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--required-power-mode", required=True, choices=WINDOWS_POWER_MODE_NAMES)
    args = parser.parse_args()
    labels = np.fromfile(args.labels, dtype="<u4")
    if min(args.shape) < 1 or not 5 <= args.samples <= len(labels):
        parser.error("Require a positive image shape and at least five available timing samples")
    images = np.memmap(args.inputs, mode="r", dtype="<f4").reshape(len(labels), *args.shape)
    expected = np.fromfile(args.reference, dtype="<f4").reshape(len(labels), -1)
    if not np.isfinite(expected).all() or expected.shape[1] < 2 or np.any(labels >= expected.shape[1]):
        parser.error("Invalid reference logits or class labels")
    pin_cpu(args.cpu)
    args.output.mkdir(parents=True, exist_ok=False)
    start = perf_counter_ns()
    graph = Graph.from_onnx(args.onnx)
    artifact = args.output / "graph.leaf"
    export_graph(run_fusion_passes(fold_constants(graph)), str(artifact))
    export_ms = (perf_counter_ns() - start) / 1e6
    bound = (args.onnx, args.inputs, args.reference, args.labels, args.executable, artifact, Path(__file__))
    report = dict(
        complete=False, scope=__doc__, onnxruntime_version=ort.__version__, cpu=args.cpu, threads=1,
        shape=args.shape, quality_samples=len(labels), timing_samples=args.samples, warmup=20, pairs=[],
        precision="FP32", power_mode={}, leaf_offline_export_ms=export_ms,
        hashes={str(p): digest(p) for p in bound},
    )
    manifest = args.output / "comparison.json"
    try:
        report["quality"] = {
            "leaf": leaf_run(args, (args.inputs, args.reference, args.labels, len(labels)), artifact,
                             args.output / "leaf-quality.json"),
            "onnxruntime": ort_run(args, images, expected, labels, timing=False),
        }
        for quality in report["quality"].values():
            require_parity(quality, len(labels))
            quality.pop("latency_samples_ms", None)
        selected = np.linspace(0, len(labels) - 1, args.samples, dtype=int)
        timing_images, timing_expected, timing_labels = images[selected], expected[selected], labels[selected]
        staged = tuple(args.output / p for p in ("inputs.bin", "logits.bin", "labels.u32"))
        for path, data in zip(staged, (timing_images, timing_expected, timing_labels)):
            data.tofile(path)
        report["timing_indices"] = selected.tolist()
        with keep_awake(), benchmark_power_mode_guard(report["power_mode"], required_mode=args.required_power_mode):
            for index in range(6):
                order = ("leaf", "onnxruntime") if index % 2 == 0 else ("onnxruntime", "leaf")
                pair = dict(order=order)
                for runtime in order:
                    require_benchmark_ac()
                    if runtime == "leaf":
                        row = leaf_run(args, (*staged, args.samples), artifact, args.output / f"leaf-{index+1}.json")
                    else:
                        row = ort_run(args, timing_images, timing_expected, timing_labels, timing=True)
                    require_benchmark_ac()
                    require_parity(row, args.samples)
                    samples = row["latency_samples_ms"]
                    pair[runtime] = dict(samples_ms=samples, stability=latency_stability(samples))
                    if runtime == "onnxruntime":
                        pair[runtime]["session_setup_ms"] = row["session_setup_ms"]
                report["pairs"].append(pair)
                write_record(manifest, report)
        series = {side: [p[side]["samples_ms"] for p in report["pairs"]] for side in ("leaf", "onnxruntime")}
        pooled = {side: sum(rows, []) for side, rows in series.items()}
        ratio = median(median(leaf) / median(ort) for leaf, ort in zip(series["leaf"], series["onnxruntime"]))
        stable = all(latency_stability(row)["passed"] for rows in series.values() for row in rows)
        stable = stable and all(latency_stability(row)["passed"] for row in pooled.values())
        report["summary"] = dict(medians_ms={side: median(row) for side, row in pooled.items()},
                                 median_paired_ratio=ratio, stable=stable,
                                 leaf_speed_qualified=stable and ratio <= .98)
        if any(digest(path) != report["hashes"][str(path)] for path in bound):
            raise ValueError("Benchmark inputs changed during execution")
        report["complete"] = True
        print(json.dumps(report["summary"], indent=2), flush=True)
    finally:
        write_record(manifest, report)


if __name__ == "__main__":
    main()
