import json
import numpy as np
import pytest

from leaf.embedding import documents, format_text
from tools.embedding_validation import parity


def test_explicit_query_document_and_raw_prefixes():
    assert format_text("hello", "query") == "task: search result | query: hello"
    assert format_text("hello", "document") == "title: none | text: hello"
    assert format_text("hello", "raw") == "hello"


@pytest.mark.parametrize("text", [None, 42, "", "   "])
def test_invalid_document_is_not_silently_encoded(text):
    with pytest.raises(ValueError):
        format_text(text, "document")


def test_jsonl_iteration_keeps_order_and_does_not_skip_missing_fields(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text('{"body":"first"}\n\n{"body":"second"}\n{}\n')
    rows = documents(None, path, "body")
    assert next(rows) == "first"
    assert next(rows) == "second"
    with pytest.raises(ValueError, match="row 4"):
        next(rows)


def test_parity_rejects_incorrect_direction_even_with_unit_norm():
    with pytest.raises(ValueError, match="parity failed"):
        parity(np.array([[0.0, 1.0]]), np.array([[1.0, 0.0]]))


def test_parity_does_not_report_cosine_above_one():
    vector = np.array([[0.6, 0.8]], dtype=np.float32)
    assert 0.99999 <= parity(vector, vector)["min_cosine"] <= 1
