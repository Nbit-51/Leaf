import numpy as np
import pytest

from tools.evaluate_retrieval import read_ids, read_qrels, score_retrieval


def test_cosine_ties_and_graded_relevance():
    q = np.array([[1., 0.]])
    d = np.array([[4., 0.], [2., 0.], [0., 1.]])
    result = score_retrieval(['q'], ['b', 'a', 'c'], q, d, {'q': {'b': 2, 'c': 1}}, 2)
    assert [row['document_id'] for row in result['queries'][0]['results']] == ['a', 'b']
    assert result['recall'] == 0.5
    assert result['ndcg'] == pytest.approx((2 / np.log2(3)) / (2 + 1 / np.log2(3)))


def test_qrels_requires_all_positive_documents_and_unique_ids(tmp_path):
    docs = tmp_path / 'docs.jsonl'
    docs.write_text('{"_id":"d"}\n{"_id":"d"}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='unique'):
        read_ids(docs)
    qrels = tmp_path / 'test.tsv'
    qrels.write_text('query-id\tcorpus-id\tscore\nq\toutside\t1\n', encoding='utf-8')
    with pytest.raises(ValueError, match='missing judged document'):
        read_qrels(qrels, ['q'], ['d'])


def test_rejects_misaligned_or_zero_embeddings():
    with pytest.raises(ValueError, match='aligned'):
        score_retrieval(['q'], ['d', 'e'], np.ones((1, 2)), np.ones((1, 2)), {}, 10)
    with pytest.raises(ValueError, match='nonzero'):
        score_retrieval(['q'], ['d'], np.ones((1, 2)), np.zeros((1, 2)), {}, 10)
