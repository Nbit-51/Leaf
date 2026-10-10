"""Native embedding protocol and explicit FP32 parity gates (no speed claims)."""

from pathlib import Path
import json
import struct
import subprocess
import tempfile

import numpy as np


def run_native(executable, artifact, sequences, *, threads=1, scalar=False):
    with tempfile.TemporaryDirectory(prefix="leaf_embedding_") as temp:
        request, output = Path(temp) / "request.bin", Path(temp) / "output.bin"
        with request.open("wb") as stream:
            stream.write(b"LEAFER01" + struct.pack("<I", len(sequences)))
            for ids in sequences:
                stream.write(struct.pack("<I", len(ids)))
                stream.write(np.asarray(ids, dtype="<u4").tobytes())
        command = [
            str(Path(executable).resolve()),
            str(Path(artifact).resolve()),
            str(request),
            str(output),
            str(threads),
        ]
        if scalar:
            command.append("--scalar")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        metrics = json.loads(result.stdout)
        values = np.fromfile(output, dtype="<f4")
        if (
            metrics["sequences"] != len(sequences)
            or values.size != len(sequences) * metrics["dimensions"]
        ):
            raise ValueError("Wrong native embedding output size")
        return values.reshape(len(sequences), metrics["dimensions"]), metrics


def parity(actual, reference):
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise ValueError("Invalid native embeddings")
    error = float(np.max(np.abs(actual - reference)))
    a64, r64 = actual.astype(np.float64), reference.astype(np.float64)
    cosine = np.clip(
        np.sum(a64 * r64, axis=1)
        / (np.linalg.norm(a64, axis=1) * np.linalg.norm(r64, axis=1)),
        -1.0,
        1.0,
    )
    norm_error = float(np.max(np.abs(np.linalg.norm(actual, axis=1) - 1)))
    result = dict(
        max_abs_error=error,
        min_cosine=float(cosine.min()),
        max_norm_error=norm_error,
        max_abs_limit=2e-4,
        min_cosine_limit=0.99999,
        max_norm_error_limit=1e-5,
    )
    result["passed"] = (
        error <= 2e-4 and result["min_cosine"] >= 0.99999 and norm_error <= 1e-5
    )
    if not result["passed"]:
        raise ValueError(f"Native embedding parity failed: {result}")
    return result
