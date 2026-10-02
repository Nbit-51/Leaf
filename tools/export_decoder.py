"""Stream a supported Hugging Face decoder into a mapped native Leaf artifact.

No ONNX export or full FP32 model materialization is required. Linear weights
use row-major FP32, symmetric per-row INT8, or symmetric grouped INT4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.decoder_plan import config_values, make_plan, tensor_shapes, validate_plan, validate_snapshot_generation

MAGIC = b"LEAFDC02"
HEADER = struct.Struct("<8sIIQ10Iff8If")
DESCRIPTOR = struct.Struct("<5I4Q")


def tensor_index(filename: Path) -> tuple[int, dict]:
    with filename.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError("Truncated safetensors header")
        length = struct.unpack("<Q", length_bytes)[0]
        if length > 100_000_000 or length + 8 > filename.stat().st_size:
            raise ValueError("Invalid safetensors header length")
        entries = json.loads(stream.read(length))
    entries.pop("__metadata__", None)
    return 8 + length, entries


def read_tensor(filename: Path, base: int, entry: dict) -> np.ndarray:
    dtypes = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}
    if entry["dtype"] not in dtypes:
        raise ValueError(f"Unsupported source weight dtype {entry['dtype']}")
    shape = tuple(entry["shape"])
    start, end = entry["data_offsets"]
    dtype = np.dtype(dtypes[entry["dtype"]])
    if (any(not isinstance(d, int) or d <= 0 for d in shape) or
            start < 0 or end < start or end + base > filename.stat().st_size or
            end - start != int(np.prod(shape)) * dtype.itemsize):
        raise ValueError("Invalid safetensors tensor range")
    array = np.memmap(filename, dtype=dtype, mode="r", offset=base + start, shape=shape)
    if entry["dtype"] == "BF16":
        return (array.astype(np.uint32) << 16).view(np.float32)
    return array.astype(np.float32, copy=False)


def transform_array(array: np.ndarray, operations: list[dict]) -> np.ndarray:
    for operation in operations:
        if set(operation) == {"reshape"}:
            array = array.reshape(operation["reshape"])
        elif set(operation) == {"transpose"}:
            array = array.transpose(operation["transpose"])
        elif set(operation) == {"slice"}:
            axis, start, end = operation["slice"]
            selectors = [slice(None)] * array.ndim
            selectors[axis] = slice(start, end)
            array = array[tuple(selectors)]
        else:
            raise ValueError(f"Unsupported declarative tensor transform {operation}")
    return array


def quantize(array: np.ndarray, bits: int, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = array.shape
    if cols % group_size:
        raise ValueError(f"Input width {cols} is not divisible by group size {group_size}")
    grouped = array.reshape(rows, cols // group_size, group_size)
    limit = (1 << (bits - 1)) - 1
    scales = np.maximum(np.max(np.abs(grouped), axis=-1) / limit, 1e-12).astype("<f4")
    values = np.clip(np.rint(grouped / scales[..., None]), -limit, limit).astype(np.int8)
    values = values.reshape(rows, cols)
    if bits == 4:
        nibbles = values.astype(np.uint8) & 15
        values = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    return np.ascontiguousarray(values), np.ascontiguousarray(scales)


def channel_factors(weight: np.ndarray, activation_max: np.ndarray, alpha: float) -> np.ndarray:
    """Equivalent W*s and X/s rescaling before quantization (SmoothQuant)."""
    if not 0 <= alpha <= 1 or activation_max.shape != (weight.shape[1],):
        raise ValueError("Invalid smoothing alpha or calibration width")
    if not np.isfinite(activation_max).all() or (activation_max < 0).any():
        raise ValueError("Invalid activation calibration values")
    weight_max = np.maximum(np.max(np.abs(weight), axis=0), 1e-5)
    return np.clip(np.maximum(activation_max, 1e-5) ** alpha / weight_max ** (1 - alpha),
                   1e-4, 1e4).astype(np.float32)


def protected_int8_tensors(plan: dict) -> list[str]:
    """Protect decoder I/O weights by canonical semantics, never model name."""
    validate_plan(plan)
    return [name for name in ("model.embed_tokens.weight", "model.position_embeddings.weight", "lm_head.weight")
            if name in plan["tensors"]]


def validate_fp32_tensors(plan: dict, requested) -> set[str]:
    if requested is None:
        return set()
    if (not isinstance(requested, (list, tuple, set, frozenset)) or
            any(not isinstance(name, str) or not name for name in requested)):
        raise ValueError("FP32 tensor protection must be a collection of exact canonical names")
    names = set(requested)
    unknown = names - set(plan["tensors"])
    if unknown:
        raise ValueError(f"Unknown FP32 tensor protection names: {sorted(unknown)}")
    return names


def export_decoder(snapshot: Path, output: Path, bits: int = 32, group_size: int = 64,
                   plan_path: Path | None = None, calibration_path: Path | None = None,
                   alpha: float = 0.5, keep_fp32_tensors=None, int8_group_size: int = 0) -> dict:
    if bits not in (32, 8, 4):
        raise ValueError("Precision must be 32, 8, or 4 bits")
    if group_size <= 0 or group_size % 2:
        raise ValueError("Group size must be positive and even")
    if (not isinstance(int8_group_size, int) or isinstance(int8_group_size, bool) or
            int8_group_size < 0 or int8_group_size % 2):
        raise ValueError("INT8 group size must be zero (per row) or positive and even")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path else make_plan(config)
    validate_plan(plan)
    validate_snapshot_generation(snapshot, plan)
    kept_fp32 = validate_fp32_tensors(plan, keep_fp32_tensors)
    values = config_values(plan["config"])
    files = sorted(snapshot.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError("Snapshot has no safetensors weights")
    sources = {}
    for filename in files:
        base, entries = tensor_index(filename)
        for name, entry in entries.items():
            if name in sources:
                raise ValueError(f"Duplicate tensor {name}")
            sources[name] = (filename, base, entry)
    physical_sources = dict(sources)
    for specification in plan["tensors"].values():
        source = specification["source"]
        if source not in physical_sources:
            aliases = [alias for alias in specification.get("source_aliases", []) if alias in physical_sources]
            if len(aliases) > 1:
                raise ValueError(f"Ambiguous source aliases for {source}: {aliases}")
            if aliases:
                resolved = physical_sources[aliases[0]]
                if source in sources and sources[source] != resolved:
                    raise ValueError(f"Conflicting source mappings for {source}")
                sources[source] = resolved
    expected = tensor_shapes(plan["config"])
    calibration = {}
    calibration_metadata = None
    if calibration_path:
        from tools.validate_decoder import digest
        with np.load(calibration_path, allow_pickle=False) as saved:
            calibration_metadata = json.loads(str(saved["__metadata__"]))
            if (calibration_metadata.get("format") != "leaf-activation-calibration-v1" or
                calibration_metadata.get("model_config_sha256") != digest(snapshot / "config.json") or
                calibration_metadata.get("source_weight_sha256") != {p.name: digest(p) for p in files}):
                raise ValueError("Calibration provenance does not match source model")
            calibration = {name: saved[name].copy() for name in saved.files if name != "__metadata__"}
        if bits != 8:
            raise ValueError("Channel smoothing currently requires INT8 weights")
        for name, (rows, cols) in expected.items():
            if rows > 1 and name not in kept_fp32 and not name.startswith(("model.embed_tokens.", "model.position_embeddings.")):
                if name not in calibration or calibration[name].shape != (cols,):
                    raise ValueError(f"Missing/incompatible calibration for {name}")
        for name in calibration:
            if name not in expected or expected[name][0] <= 1:
                raise ValueError(f"Unexpected calibration operator {name}")
            if calibration[name].shape != (expected[name][1],):
                raise ValueError(f"Missing/incompatible calibration for {name}")
        # Protection preserves source FP32 values, not merely a FP32 container
        # for smoothed weights. No matching input rescaling is needed for them.
        calibration = {name: maximum for name, maximum in calibration.items() if name not in kept_fp32}
        expected.update({name.removesuffix(".weight") + ".input_scale": (1, expected[name][1])
                         for name in calibration})
    for name in plan["tensors"]:
        if name.endswith(".bias"):
            partner = name[:-4] + "weight"
            if partner not in expected:
                raise ValueError(f"Bias has no corresponding plan weight: {name}")
            expected[name] = (1, expected[partner][0] if expected[partner][0] > 1 else expected[partner][1])
    descriptors = []
    shared_payloads = {}
    offset = 0
    for name, (rows, cols) in expected.items():
        specification = plan["tensors"].get(name)
        if not name.endswith(".input_scale") and (specification is None or specification["source"] not in sources):
            raise ValueError(f"Missing tensor mapping/source for {name}")
        precision = bits if name.endswith(".weight") and rows > 1 and name not in kept_fp32 else 32
        group = (int8_group_size or cols) if precision == 8 else min(group_size, cols) if precision == 4 else 0
        if precision == 8 and cols % group:
            raise ValueError(f"INT8 group {group} must divide input width {cols}")
        if precision == 4 and (cols % group or group % 2):
            raise ValueError(f"INT4 group {group} must divide even input width {cols}")
        identity = None if name.endswith(".input_scale") else (
            specification["source"], json.dumps(specification.get("transforms", []), sort_keys=True),
            rows, cols, precision, group, name if name in calibration else None)
        if identity in shared_payloads:
            descriptors.append((name, rows, cols, precision, group, *shared_payloads[identity]))
            continue
        size = rows * cols * precision // 8
        offset = (offset + 63) & ~63
        data_offset = offset
        offset += size
        scales = rows * (cols // group) if precision != 32 else 0
        offset = (offset + 63) & ~63
        scale_offset = offset
        offset += scales * 4
        descriptors.append((name, rows, cols, precision, group, data_offset, size,
                            scale_offset, scales))
        if identity is not None:
            shared_payloads[identity] = (data_offset, size, scale_offset, scales)
    metadata_size = HEADER.size + sum(4 + len(d[0].encode()) + DESCRIPTOR.size for d in descriptors)
    data_base = (metadata_size + 63) & ~63
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    input_scales = {}
    written = set()
    try:
        with temporary.open("wb") as stream:
            stream.write(HEADER.pack(MAGIC, 2, len(descriptors), data_base, *values))
            for name, rows, cols, precision, group, start, size, scale_start, scales in descriptors:
                encoded = name.encode("utf-8")
                stream.write(struct.pack("<I", len(encoded)))
                stream.write(encoded)
                stream.write(DESCRIPTOR.pack(2, rows, cols, precision, group, start, size, scale_start, scales))
            for name, rows, cols, precision, group, start, _, scale_start, scales in descriptors:
                if (start, scale_start) in written:
                    continue
                written.add((start, scale_start))
                if name.endswith(".input_scale"):
                    array = input_scales[name]
                else:
                    specification = plan["tensors"][name]
                    source_name = specification["source"]
                    array = read_tensor(*sources[source_name])
                    if not np.isfinite(array).all():
                        raise ValueError(f"Non-finite source tensor for {name}")
                    array = transform_array(array, specification.get("transforms", []))
                    if not np.isfinite(array).all():
                        raise ValueError(f"Non-finite transformed tensor {name}")
                if array.shape not in ((rows, cols), (cols,) if rows == 1 else ()):
                    raise ValueError(f"Incompatible transformed tensor {name}: {array.shape}, expected {(rows, cols)}")
                array = np.ascontiguousarray(array.reshape(rows, cols))
                if name in calibration:
                    factors = channel_factors(array, calibration[name], alpha)
                    with np.errstate(over="ignore", invalid="ignore"):
                        array = array * factors[None, :]
                    input_scales[name.removesuffix(".weight") + ".input_scale"] = 1.0 / factors
                if not np.isfinite(array).all():
                    raise ValueError(f"Non-finite export tensor {name}")
                payload, scale_array = (array.astype("<f4", copy=False), None) if precision == 32 else quantize(array, precision, group)
                stream.seek(data_base + start)
                stream.write(payload.tobytes())
                if scales:
                    stream.seek(data_base + scale_start)
                    stream.write(scale_array.tobytes())
                del array, payload, scale_array
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    record = {"format": "leaf-decoder-v2", "architecture": plan.get("source_architecture", "custom"),
              "weight_bits": bits, "group_size": group_size if bits == 4 else None,
              "int8_group_size": int8_group_size if bits == 8 else None,
              "keep_fp32_tensors": sorted(kept_fp32),
              "tensor_count": len(descriptors), "artifact_bytes": output.stat().st_size,
              "unique_payloads": len(written),
              "source": snapshot.name}
    if calibration_metadata:
        record.update(smoothing_alpha=alpha, calibration=calibration_metadata)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(32, 8, 4), default=32)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--int8-group-size", type=int, default=0,
                        help="0 for per-row INT8; a positive even size must divide each quantized input width")
    parser.add_argument("--keep-fp32-tensors", nargs="+",
                        help="exact canonical plan tensor names to preserve in FP32")
    parser.add_argument("--plan", type=Path, help="custom architecture-independent decoder plan")
    parser.add_argument("--calibration", type=Path, help="independently collected input-channel maxima")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(export_decoder(args.model, args.output, args.bits, args.group_size, args.plan,
                                    args.calibration, args.alpha, args.keep_fp32_tensors, args.int8_group_size), indent=2))


if __name__ == "__main__":
    main()
