"""Score existing Leaf or reference embeddings against BEIR-format judgments."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np

from tools.verify_package_runtime import digest


def read_ids(path: Path) -> list[str]:
    with path.open(encoding='utf-8') as stream:
        ids = [json.loads(line)['_id'] for line in stream if line.strip()]
    if not ids or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f'Expected unique nonempty string _id values in {path}')
    return ids


def read_qrels(path: Path, query_ids: list[str], document_ids: list[str]) -> dict:
    relevance = {query_id: {} for query_id in query_ids}
    corpus = set(document_ids)
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if reader.fieldnames != ['query-id', 'corpus-id', 'score']:
            raise ValueError('Expected BEIR qrels columns query-id, corpus-id and score')
        for row in reader:
            query, document = row['query-id'], row['corpus-id']
            if query not in relevance:
                continue
            score = int(row['score'])
            if score < 0 or document in relevance[query] or (score > 0 and document not in corpus):
                raise ValueError(f'Invalid, duplicate, or missing judged document for query {query!r}')
            relevance[query][document] = score
    if any(not any(row.values()) for row in relevance.values()):
        raise ValueError('Every supplied query must have a relevant document in the supplied corpus')
    return relevance


def score_retrieval(query_ids, document_ids, queries, documents, relevance, k):
    if (type(k) is not int or k < 1 or queries.ndim != 2 or documents.ndim != 2 or
            queries.shape[0] != len(query_ids) or documents.shape[0] != len(document_ids) or
            queries.shape[1] != documents.shape[1] or not np.issubdtype(queries.dtype, np.floating) or
            not np.issubdtype(documents.dtype, np.floating) or
            not np.isfinite(queries).all() or not np.isfinite(documents).all()):
        raise ValueError('Expected aligned, finite floating-point query and document matrices and positive k')
    query_norms, document_norms = np.linalg.norm(queries, axis=1), np.linalg.norm(documents, axis=1)
    if (not np.isfinite(query_norms).all() or not np.isfinite(document_norms).all() or
            np.any(query_norms <= 0) or np.any(document_norms <= 0)):
        raise ValueError('Cosine ranking requires finite nonzero embeddings')
    labels = np.asarray(document_ids)
    discounts = 1 / np.log2(np.arange(min(k, len(document_ids))) + 2)
    per_query = []
    for index, query in enumerate(query_ids):
        scores = (documents @ (queries[index] / query_norms[index])) / document_norms
        if not np.isfinite(scores).all():
            raise ValueError(f'Cosine scores are non-finite for query {query!r}')
        top = np.lexsort((labels, -scores))[:k]
        judgments = relevance[query]
        ideal = sorted((value for value in judgments.values() if value > 0), reverse=True)
        gains = [judgments.get(document_ids[i], 0) for i in top]
        ndcg = np.dot(gains, discounts) / np.dot(ideal[:k], discounts[:len(ideal)])
        recall = sum(value > 0 for value in gains) / len(ideal)
        per_query.append(dict(query_id=query, ndcg=float(ndcg), recall=recall,
                              results=[dict(document_id=document_ids[i], score=float(scores[i])) for i in top]))
    return dict(k=k, ndcg=statistics.mean(row['ndcg'] for row in per_query),
                recall=statistics.mean(row['recall'] for row in per_query), queries=per_query)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('query-rows', 'document-rows', 'query-vectors', 'document-vectors', 'qrels', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report path')
    paths = [args.query_rows, args.document_rows, args.query_vectors, args.document_vectors, args.qrels]
    hashes = {str(path): digest(path) for path in paths}
    query_ids, document_ids = read_ids(args.query_rows), read_ids(args.document_rows)
    relevance = read_qrels(args.qrels, query_ids, document_ids)
    queries = np.load(args.query_vectors, mmap_mode='r', allow_pickle=False)
    documents = np.load(args.document_vectors, mmap_mode='r', allow_pickle=False)
    result = score_retrieval(query_ids, document_ids, queries, documents, relevance, args.k)
    if any(digest(path) != hashes[str(path)] for path in paths):
        raise ValueError('Retrieval inputs changed during evaluation')
    result.update(format='leaf-retrieval-v1', complete=True, query_count=len(query_ids),
                  document_count=len(document_ids), similarity='cosine', gain='linear', input_sha256=hashes)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(f'{len(query_ids)} queries, {len(document_ids)} documents, nDCG@{args.k} {result["ndcg"]:.6f}, '
          f'Recall@{args.k} {result["recall"]:.6f}', flush=True)


if __name__ == '__main__':
    main()
