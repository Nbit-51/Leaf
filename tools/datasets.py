"""Dataset format adapters; numerical gates consume tokens, not dataset names."""
from __future__ import annotations
import csv
import json
from pathlib import Path


def iter_text_rows(path: Path, column: str = "text"):
    extension = path.suffix.lower()
    if extension in (".txt", ".md"):
        rows = [path.read_text(encoding="utf-8")]
    elif extension == ".parquet":
        import pyarrow.parquet as pq
        rows = (row for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=[column])
                for row in batch.column(0).to_pylist())
    elif extension in (".jsonl", ".ndjson"):
        def records():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)
        rows = records()
    elif extension == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("JSON text datasets must contain a list")
    elif extension == ".csv":
        def records():
            with path.open(encoding="utf-8", newline="") as stream:
                yield from csv.DictReader(stream)
        rows = records()
    else:
        raise ValueError("Supported text formats: TXT, JSON, JSONL, CSV and Parquet")
    found = False
    for row in rows:
        value = row.get(column) if isinstance(row, dict) else row
        if not isinstance(value, str):
            raise ValueError(f"Dataset field {column!r} must contain text strings")
        if value.strip():
            found = True
            yield value
    if not found:
        raise ValueError("Text dataset is empty")


def text_rows(path: Path, column: str = "text") -> list[str]:
    return list(iter_text_rows(path, column))


def token_prefix(tokenizer, path: Path, count: int, column: str = "text") -> list[int]:
    """Tokenize just a sufficient prefix, not a multi-gigabyte corpus.

    Preserve the existing double-newline row separator; plain text is read in
    bounded fragments without inserting separators at arbitrary chunk edges.
    JSON arrays remain eager; JSONL, CSV, and Parquet use streaming readers.
    """
    if count <= 0:
        raise ValueError("Requested token count must be positive")
    text, ids = "", []
    def encode(piece, separator):
        nonlocal text, ids
        text += (separator if text else "") + piece
        if len(text) >= count * 4:
            ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
        return len(ids) > count  # leave a suffix token beyond the evaluated boundary
    if path.suffix.lower() in (".txt", ".md"):
        with path.open(encoding="utf-8") as stream:
            for fragment in iter(lambda: stream.read(max(4096, count * 8)), ""):
                if encode(fragment, ""):
                    return ids[:count]
    else:
        for row in iter_text_rows(path, column):
            if encode(row, "\n\n"):
                return ids[:count]
    ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    if len(ids) < count:
        raise ValueError("Dataset has insufficient tokens")
    return ids[:count]
