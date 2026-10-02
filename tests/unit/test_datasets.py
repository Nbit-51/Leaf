import json

import pytest

from tools.datasets import text_rows, token_prefix


@pytest.mark.parametrize("suffix", [".json", ".jsonl", ".csv", ".txt"])
def test_text_dataset_formats_and_custom_column(tmp_path, suffix):
    path = tmp_path / ("local" + suffix)
    content = {".json": json.dumps([{"sentence": "hello"}, {"sentence": " "}]),
               ".jsonl": '{"sentence":"hello"}\n{"sentence":" "}\n',
               ".csv": "sentence\nhello\n \n", ".txt": "hello"}[suffix]
    path.write_text(content, encoding="utf-8")
    assert text_rows(path, "sentence") == ["hello"]


def test_missing_column_is_not_silently_dropped(tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text('[{"label":42}]')
    with pytest.raises(ValueError, match="text strings"):
        text_rows(path)


def test_empty_dataset_is_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   ")
    with pytest.raises(ValueError, match="empty"):
        text_rows(path)


def test_bounded_token_prefix_preserves_text_and_row_separators(tmp_path):
    class Characters:
        def encode(self, text, **kwargs):
            return [ord(value) for value in text]
    text = tmp_path / "long.txt"
    text.write_text("hello\nworld" * 10000)
    assert token_prefix(Characters(), text, 20) == [ord(c) for c in text.read_text()[:20]]
    jsonl = tmp_path / "sentences.jsonl"
    jsonl.write_text('{"text":"hello"}\n{"text":"world"}\n')
    assert token_prefix(Characters(), jsonl, 10) == [ord(c) for c in "hello\n\nworld"[:10]]


def test_insufficient_prefix_tokens_are_rejected(tmp_path):
    class Characters:
        def encode(self, text, **kwargs):
            return list(text)
    text = tmp_path / "short.txt"
    text.write_text("abc")
    with pytest.raises(ValueError, match="insufficient tokens"):
        token_prefix(Characters(), text, 4)
