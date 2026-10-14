"""Lightweight local document embedding; model computation stays in C++."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np


def executable(cache):
    from leaf.cli import source_engine

    configured = os.environ.get("LEAF_EMBED_BIN")
    if configured:
        path = Path(configured)
        if not path.is_file():
            raise FileNotFoundError("LEAF_EMBED_BIN does not point to a file. Remove this optional override "
                                    "(PowerShell: Remove-Item Env:LEAF_EMBED_BIN) and run leaf doctor.")
        return path.resolve()
    name = "leaf_embed.exe" if os.name == "nt" else "leaf_embed"
    bundled = Path(__file__).parent / "bin" / name
    if bundled.is_file():
        return bundled
    engine = source_engine()
    sources = [
        engine / x
        for x in (
            "src/embedding.cpp",
            "src/main_embed.cpp",
            "src/kernels/transformer.cpp",
        )
    ]
    checksum = hashlib.sha256((sys.platform + os.name).encode())
    for path in sources + sorted((engine / "include").rglob("*.h")):
        checksum.update(path.read_bytes())
    output = cache / "native-embedding" / checksum.hexdigest()[:16] / name
    if output.is_file():
        return output
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise RuntimeError("No bundled embedding runtime or C++17 compiler found. Install a Leaf platform "
                           "wheel, or install GCC/Clang for a source checkout; no binary override is needed.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    command = [
        compiler,
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-ffp-contract=off",
        "-pthread",
        "-I",
        str(engine / "include"),
        *map(str, sources),
        "-o",
        str(temporary),
    ]
    if os.name == "nt":
        command += ["-static", "-static-libgcc", "-static-libstdc++"]
    subprocess.run(command, check=True)
    temporary.replace(output)
    return output


def format_text(text, task):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Embedding text must be a nonempty string")
    if task == "query":
        return "task: search result | query: " + text
    if task == "document":
        return "title: none | text: " + text
    if task == "raw":
        return text
    raise ValueError("Unknown embedding task")


def documents(text, dataset, column):
    if text is not None:
        yield text
    else:
        with Path(dataset).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or column not in row:
                    raise ValueError(
                        f"JSONL row {line_number} lacks text column {column!r}"
                    )
                yield row[column]


def resolve_model(model, cache, offline=False):
    """Resolve a local snapshot, registered alias or supported Hub model."""
    local = Path(model)
    if local.is_dir():
        return local.resolve()
    aliases = cache / "aliases.json"
    if aliases.is_file():
        model = json.loads(aliases.read_text(encoding="utf-8")).get(model, model)
        if Path(model).is_dir():
            return Path(model).resolve()
    if Path(model).is_absolute() or model.startswith((".", "~")) or "\\" in model:
        raise ValueError(f"Model folder does not exist: {model}")

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import (
        GatedRepoError, LocalEntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError,
    )
    from tools.verify_embeddinggemma_reference import inspect_metadata
    from tools.export_embedding import embedding_config

    print(f"Preparing {model} from {'the local cache' if offline else 'Hugging Face'}...",
          file=sys.stderr)
    try:
        metadata = Path(snapshot_download(
            model, cache_dir=cache / "hub", local_files_only=offline,
            allow_patterns=["config.json", "modules.json", "1_Pooling/config.json",
                            "2_Dense/config.json", "3_Dense/config.json"],
        ))
        try:
            embedding_config(inspect_metadata(metadata))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(
                "This model's embedding configuration is not supported by Leaf. "
                "The supported pipeline is original EmbeddingGemma 300M. "
                f"Details: {error}"
            ) from error
        # Hub snapshots live at snapshots/<commit>. Fetch the same revision so
        # configuration and weights cannot come from different main revisions.
        snapshot = Path(snapshot_download(
            model, revision=metadata.name, cache_dir=cache / "hub", local_files_only=offline,
            allow_patterns=["*.json", "model.safetensors", "tokenizer.model",
                            "2_Dense/model.safetensors", "3_Dense/model.safetensors"],
        ))
    except GatedRepoError as error:
        raise ValueError(
            f"Access to {model} requires accepting its terms at "
            f"https://huggingface.co/{model}, then signing in with `hf auth login`. "
            "Rerun the same Leaf command afterwards."
        ) from error
    except LocalEntryNotFoundError as error:
        raise ValueError(
            f"Model {model} is not fully available in the cache. "
            "Run once online without --offline, or use a complete local model folder."
        ) from error
    except RepositoryNotFoundError as error:
        raise ValueError(f"Cannot access model {model}. Check the model ID and your Hugging Face login.") from error
    except HfHubHTTPError as error:
        raise ValueError(f"Could not download {model}. Check your connection and retry the same command.") from error
    return snapshot


def run(args):
    from tokenizers import Tokenizer
    from leaf.cli import cache_root
    from tools.export_embedding import export_embedding
    from tools.verify_embeddinggemma_reference import inspect_snapshot, digest

    start = time.perf_counter()
    if args.output.suffix != ".npy":
        raise ValueError("Embedding output must use .npy")
    if args.output.exists():
        raise FileExistsError("Choose a new output file")
    if args.metrics and (
        args.metrics.exists() or args.metrics.resolve() == args.output.resolve()
    ):
        raise ValueError(
            "Metrics require a new path distinct from the embedding output"
        )
    cache = cache_root()
    runtime = executable(cache)
    snapshot = resolve_model(args.model, cache, getattr(args, "offline", False))
    try:
        config, hashes = inspect_snapshot(snapshot)
    except FileNotFoundError as error:
        raise ValueError(
            "The model folder/cache is incomplete. Use the model ID "
            "(for example google/embeddinggemma-300m) online so Leaf downloads "
            "its supporting files automatically."
        ) from error
    key = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode() + b"leaf-embedding-v1"
    ).hexdigest()
    artifact = cache / "embeddings" / key / "model.leaf"
    if not artifact.is_file():
        export_embedding(snapshot, artifact)
    else:
        provenance = json.loads(artifact.with_suffix(".provenance.json").read_text())
        if (
            provenance["snapshot_sha256"] != hashes
            or digest(artifact) != provenance["artifact_sha256"]
        ):
            raise ValueError("Cached embedding artifact failed integrity checks")
    prepared = time.perf_counter()
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="leaf_embed_", dir=args.output.parent
    ) as temporary:
        folder = Path(temporary)
        request = folder / "request.bin"
        raw = folder / "embeddings.bin"
        count = 0
        tokens = 0
        with request.open("wb") as stream:
            stream.write(b"LEAFER01" + struct.pack("<I", 0))
            for text in documents(args.text, args.dataset, args.text_column):
                ids = tokenizer.encode(format_text(text, args.task)).ids
                if not ids or len(ids) > config["max_position_embeddings"]:
                    raise ValueError(
                        f"Document {count + 1} exceeds the model context; split it explicitly"
                    )
                count += 1
                tokens += len(ids)
                if count > 1000000:
                    raise ValueError("At most one million documents per invocation")
                stream.write(
                    struct.pack("<I", len(ids)) + np.asarray(ids, dtype="<u4").tobytes()
                )
            if not count:
                raise ValueError("No documents to embed")
            stream.seek(8)
            stream.write(struct.pack("<I", count))
        tokenized = time.perf_counter()
        result = subprocess.run(
            [str(runtime), str(artifact), str(request), str(raw), str(args.threads)],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        native_done = time.perf_counter()
        native = json.loads(result.stdout)
        if (
            native["sequences"] != count
            or raw.stat().st_size != count * native["dimensions"] * 4
        ):
            raise ValueError("Native embedding output is incomplete")
        values = np.memmap(
            raw, dtype="<f4", mode="r", shape=(count, native["dimensions"])
        )
        staged = folder / "output.npy"
        output = np.lib.format.open_memmap(
            staged, mode="w+", dtype="<f4", shape=values.shape
        )
        for begin in range(0, count, 256):
            block = values[begin : begin + 256]
            if not np.isfinite(block).all():
                raise ValueError("Native embeddings are nonfinite")
            output[begin : begin + 256] = block
        output.flush()
        del output, values, block
        staged.replace(args.output)
    finished = time.perf_counter()
    metrics = dict(
        documents=count,
        tokens=tokens,
        dimensions=native["dimensions"],
        threads=args.threads,
        task=args.task,
        precision="FP32",
        avx2=native["avx2"],
        preparation_ms=(prepared - start) * 1000,
        tokenization_ms=(tokenized - prepared) * 1000,
        native_process_ms=(native_done - tokenized) * 1000,
        output_write_ms=(finished - native_done) * 1000,
        end_to_end_ms=(finished - start) * 1000,
        timing_scope="One CLI observation; native process includes startup and artifact loading, not a speed qualification.",
        source_sha256=hashes,
        native_sha256=digest(runtime),
        artifact_sha256=digest(artifact),
    )
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved {count} x {native['dimensions']} FP32 embeddings to {args.output}")
