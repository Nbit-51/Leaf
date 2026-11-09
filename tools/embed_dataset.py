"""Embed a JSONL dataset in recoverable batches using Leaf's embedding runtime."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from leaf import embedding
from leaf.cli import cache_root
from leaf.evaluation import write_record
from tools.verify_embeddinggemma_reference import digest, inspect_snapshot


@contextmanager
def job_lock(path: Path):
    with path.open('a+b') as stream:
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Another process holds this embedding checkpoint') from error
        yield  # Closing the descriptor releases the lock, including after a crash.


def verified_chunk(directory: Path, rows: int, input_sha: str, identity: dict,
                   artifact_sha: str | None, dimensions: int | None) -> dict:
    record = json.loads((directory / 'chunk.json').read_text(encoding='utf-8'))
    metrics_path = directory / 'metrics.json'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    values = directory / 'embeddings.npy'
    if (record['input_sha256'] != input_sha or record['rows'] != rows or
            record['vectors_sha256'] != digest(values) or record['metrics_sha256'] != digest(metrics_path) or
            metrics['documents'] != rows or metrics['precision'] != 'FP32' or
            metrics['task'] != identity['task'] or metrics['threads'] != identity['threads'] or
            metrics['source_sha256'] != identity['model_sha256'] or metrics['native_sha256'] != identity['native_sha256']):
        raise ValueError(f'Checkpoint batch failed its identity checks: {directory.name}')
    array = np.load(values, mmap_mode='r', allow_pickle=False)
    if array.dtype != np.dtype('<f4') or array.shape != (rows, metrics['dimensions']):
        raise ValueError(f'Checkpoint batch has invalid vector shape: {directory.name}')
    if ((artifact_sha is not None and metrics['artifact_sha256'] != artifact_sha) or
            (dimensions is not None and metrics['dimensions'] != dimensions)):
        raise ValueError('Embedding artifact or dimensions changed across checkpoint batches')
    return metrics


def index_dataset(args) -> dict:
    import tokenizers

    args.directory.mkdir(parents=True, exist_ok=True)
    with job_lock(args.directory / 'job.lock'):
        cache = cache_root()
        runtime = embedding.executable(cache)
        snapshot = embedding.resolve_model(args.model, cache, args.offline)
        _, model_hashes = inspect_snapshot(snapshot)
        identity = dict(format='leaf-embedding-checkpoint-v1', dataset_sha256=digest(args.dataset),
                        model_sha256=model_hashes, native_sha256=digest(runtime),
                        frontend_sha256=digest(Path(embedding.__file__)), tokenizer_version=tokenizers.__version__,
                        task=args.task, threads=args.threads, text_column=args.text_column, chunk_size=args.chunk_size)
        manifest = args.directory / 'job.json'
        if manifest.exists():
            if json.loads(manifest.read_text(encoding='utf-8')) != identity:
                raise ValueError('Dataset, model, runtime, tokenizer, or embedding settings differ from the checkpoint')
        else:
            if any(path.name not in ('job.lock', 'job.json.partial') for path in args.directory.iterdir()):
                raise ValueError('Choose an empty directory or an existing embedding checkpoint')
            write_record(manifest, identity)

        chunks, new_chunks, rows, dimensions, artifact_sha = [], 0, 0, None, None
        with args.dataset.open(encoding='utf-8', newline='') as stream:
            lines = (line for line in stream if line.strip())
            for number, batch in enumerate(iter(lambda: list(islice(lines, args.chunk_size)), [])):
                data = ''.join(batch).encode('utf-8')
                input_sha = hashlib.sha256(data).hexdigest()
                chunk = args.directory / f'batch-{number:06d}'
                if not chunk.exists():
                    if args.max_new_chunks is not None and new_chunks >= args.max_new_chunks:
                        if digest(args.dataset) != identity['dataset_sha256']:
                            raise ValueError('Dataset changed during indexing')
                        return dict(complete=False, rows=rows, chunks=len(chunks), new_chunks=new_chunks)
                    with tempfile.TemporaryDirectory(prefix='staging-', dir=args.directory) as scratch:
                        staging = Path(scratch) / 'batch'
                        staging.mkdir()
                        request = Path(scratch) / 'rows.jsonl'
                        request.write_bytes(data)
                        vectors = staging / 'embeddings.npy'
                        embedding.run(SimpleNamespace(model=str(snapshot), offline=True, dataset=request, text=None,
                                                      text_column=args.text_column, task=args.task, threads=args.threads,
                                                      output=vectors, metrics=staging / 'metrics.json'))
                        write_record(staging / 'chunk.json', dict(rows=len(batch), input_sha256=input_sha,
                                                                 vectors_sha256=digest(vectors),
                                                                 metrics_sha256=digest(staging / 'metrics.json')))
                        metrics = verified_chunk(staging, len(batch), input_sha, identity, artifact_sha, dimensions)
                        staging.rename(chunk)
                    new_chunks += 1
                else:
                    metrics = verified_chunk(chunk, len(batch), input_sha, identity, artifact_sha, dimensions)
                dimensions = metrics['dimensions']
                artifact_sha = metrics['artifact_sha256']
                chunks.append((chunk / 'embeddings.npy', len(batch)))
                rows += len(batch)
        if not chunks:
            raise ValueError('No documents to embed')
        if digest(args.dataset) != identity['dataset_sha256']:
            raise ValueError('Dataset changed during indexing')

        output = args.directory / 'embeddings.npy'
        result_path = args.directory / 'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding='utf-8'))
            if (result['complete'] is not True or result['rows'] != rows or result['dimensions'] != dimensions or
                    result['chunks'] != len(chunks) or result['artifact_sha256'] != artifact_sha or
                    result['vectors_sha256'] != digest(output)):
                raise ValueError('Completed embedding output failed its integrity check')
            return dict(result, new_chunks=new_chunks)
        with tempfile.TemporaryDirectory(prefix='assemble-', dir=args.directory) as scratch:
            staged = Path(scratch) / 'embeddings.npy'
            matrix = np.lib.format.open_memmap(staged, mode='w+', dtype='<f4', shape=(rows, dimensions))
            start = 0
            for path, count in chunks:
                matrix[start:start + count] = np.load(path, mmap_mode='r', allow_pickle=False)
                start += count
            matrix.flush()
            del matrix
            checksum = digest(staged)
            staged.replace(output)
        result = dict(complete=True, rows=rows, dimensions=dimensions, chunks=len(chunks),
                      artifact_sha256=artifact_sha, vectors_sha256=checksum)
        write_record(result_path, result)
        return dict(result, new_chunks=new_chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--directory', required=True, type=Path, help='checkpoint and output directory')
    parser.add_argument('--text-column', default='text')
    parser.add_argument('--task', choices=('document', 'query', 'raw'), default='document')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--max-new-chunks', type=int, help='pause after this many newly completed batches')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.threads <= 64 or not 1 <= args.chunk_size <= 1_000_000 or (
            args.max_new_chunks is not None and args.max_new_chunks < 1):
        parser.error('Use 1–64 threads, a chunk size from 1 to 1,000,000 and a positive batch limit')
    print(json.dumps(index_dataset(args), indent=2), flush=True)


if __name__ == '__main__':
    main()
