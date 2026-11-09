import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import embed_dataset


@pytest.fixture
def job(tmp_path, monkeypatch):
    dataset = tmp_path / 'rows.jsonl'
    dataset.write_text(''.join(f'{{"text":"{n}"}}\n\n' for n in range(5)), encoding='utf-8')
    runtime = tmp_path / 'runtime'
    runtime.write_bytes(b'fixed runtime')
    model_hashes = {'config.json': 'fixed model'}
    monkeypatch.setattr(embed_dataset.embedding, 'executable', lambda _: runtime)
    monkeypatch.setattr(embed_dataset.embedding, 'resolve_model', lambda *args: tmp_path)
    monkeypatch.setattr(embed_dataset, 'inspect_snapshot', lambda _: ({}, model_hashes))
    calls = []

    def embed(args):
        values = [int(json.loads(row)['text']) for row in args.dataset.read_text().splitlines()]
        calls.append(values)
        np.save(args.output, np.array([[n, n + 1] for n in values], dtype='<f4'))
        args.metrics.write_text(json.dumps(dict(documents=len(values), dimensions=2, task=args.task,
            threads=args.threads, source_sha256=model_hashes, native_sha256=embed_dataset.digest(runtime),
            precision='FP32', artifact_sha256='fixed artifact')))

    monkeypatch.setattr(embed_dataset.embedding, 'run', embed)
    args = SimpleNamespace(model='local', offline=True, dataset=dataset, directory=tmp_path / 'job',
                           task='raw', threads=1, text_column='text', chunk_size=2, max_new_chunks=1)
    return args, calls, embed


def test_resumes_in_row_order_and_reuses_finished_batches(job):
    args, calls, _ = job
    assert embed_dataset.index_dataset(args) == dict(complete=False, rows=2, chunks=1, new_chunks=1)
    assert not (args.directory / 'embeddings.npy').exists()
    args.max_new_chunks = None
    result = embed_dataset.index_dataset(args)
    assert result['complete'] and result['rows'] == 5 and result['new_chunks'] == 2
    np.testing.assert_array_equal(np.load(args.directory / 'embeddings.npy'), [[n, n + 1] for n in range(5)])
    assert embed_dataset.index_dataset(args)['new_chunks'] == 0
    assert calls == [[0, 1], [2, 3], [4]]


def test_interrupted_batch_is_retried_without_publishing_partial_vectors(job, monkeypatch):
    args, calls, embed = job
    embed_dataset.index_dataset(args)

    def interrupt(batch):
        embed(batch)
        raise KeyboardInterrupt

    monkeypatch.setattr(embed_dataset.embedding, 'run', interrupt)
    with pytest.raises(KeyboardInterrupt):
        embed_dataset.index_dataset(args)
    assert not (args.directory / 'batch-000001').exists()
    monkeypatch.setattr(embed_dataset.embedding, 'run', embed)
    assert embed_dataset.index_dataset(args)['rows'] == 4
    assert calls == [[0, 1], [2, 3], [2, 3]]


def test_rejects_changed_inputs_and_corrupted_batches(job):
    args, calls, _ = job
    embed_dataset.index_dataset(args)
    args.threads = 2
    with pytest.raises(ValueError, match='differ from the checkpoint'):
        embed_dataset.index_dataset(args)
    args.threads = 1
    original = args.dataset.read_bytes()
    args.dataset.write_bytes(original + b'{"text":"5"}\n')
    with pytest.raises(ValueError, match='differ from the checkpoint'):
        embed_dataset.index_dataset(args)
    args.dataset.write_bytes(original)
    with (args.directory / 'batch-000000/embeddings.npy').open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='identity checks'):
        embed_dataset.index_dataset(args)
    assert calls == [[0, 1]]


def test_job_lock_rejects_second_writer(tmp_path):
    path = tmp_path / 'job.lock'
    with embed_dataset.job_lock(path):
        with pytest.raises(RuntimeError, match='Another process'):
            with embed_dataset.job_lock(path):
                pytest.fail('Second writer was allowed')
    with embed_dataset.job_lock(path):
        assert path.is_file()
