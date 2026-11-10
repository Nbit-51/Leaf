"""Prepare the pinned BEIR SciFact test split from local Parquet and qrels files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tools.verify_package_runtime import digest


CORPUS_REVISION = 'b3b5335604bf5ee3c4447671af975ea25143d4f5'
QRELS_REVISION = '2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029'
SOURCE_SHA256 = {
    'corpus': '243324b35f03d82bd6d98a5f575966876e86cad7ce16e5333a35b1b793dc4f45',
    'queries': '1c37956c5dc8b810b60302323c24d1a9e79e26411ba8f5ad9d0888642e2a9034',
    'qrels': '0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c',
}


def prepare(corpus: Path, queries: Path, qrels: Path, output: Path) -> dict:
    for name, path in (('corpus', corpus), ('queries', queries), ('qrels', qrels)):
        if digest(path) != SOURCE_SHA256[name]:
            raise ValueError(f'{path} does not match the pinned SciFact {name} file')
    import pyarrow.parquet as parquet

    judgments = qrels.read_bytes()
    judged_queries = {
        row['query-id'] for row in csv.DictReader(judgments.decode('utf-8').splitlines(), delimiter='\t')
        if int(row['score']) > 0
    }
    documents = [
        dict(_id=row['_id'], text=f"title: {row['title'] or 'none'} | text: {row['text']}")
        for row in parquet.read_table(corpus, columns=['_id', 'title', 'text']).to_pylist()
    ]
    test_queries = [
        row for row in parquet.read_table(queries, columns=['_id', 'text']).to_pylist()
        if row['_id'] in judged_queries
    ]
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (('documents.jsonl', documents), ('queries.jsonl', test_queries)):
        # Explicit CRLF keeps the prepared hashes portable and matches the original checkpoint.
        with (output / name).open('x', encoding='utf-8', newline='\r\n') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    (output / 'test.tsv').write_bytes(judgments)
    record = dict(
        corpus_revision=CORPUS_REVISION,
        qrels_revision=QRELS_REVISION,
        documents=len(documents),
        queries=len(test_queries),
        sha256={name: digest(output / name) for name in ('documents.jsonl', 'queries.jsonl', 'test.tsv')},
    )
    (output / 'dataset.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus', 'queries', 'qrels', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.corpus, args.queries, args.qrels, args.output), indent=2))


if __name__ == '__main__':
    main()
