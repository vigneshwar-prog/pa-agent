"""tests/test_ingest.py — unit tests for ingest pipeline."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingest import _tag_category, _sha256, SPLITTER


# ── Category tagging ──────────────────────────────────────────────────────────

def test_tag_category_skills():
    assert _tag_category("He is proficient in Python and SQL") == "skills"


def test_tag_category_experience():
    assert _tag_category("He worked at Infosys as a software engineer") == "experience"


def test_tag_category_education():
    assert _tag_category("He graduated with a degree in Computer Science") == "education"


def test_tag_category_general():
    assert _tag_category("This is a random sentence with no matching keywords") == "general"


# ── SHA-256 dedup ─────────────────────────────────────────────────────────────

def test_sha256_deterministic():
    assert _sha256("hello world") == _sha256("hello world")


def test_sha256_unique():
    assert _sha256("chunk A") != _sha256("chunk B")


# ── Chunking ──────────────────────────────────────────────────────────────────

def test_chunking_splits_long_text():
    long_text = "This is a sentence. " * 200  # ~600+ tokens
    from langchain_core.documents import Document
    docs = [Document(page_content=long_text, metadata={"source": "test"})]
    chunks = SPLITTER.split_documents(docs)
    assert len(chunks) > 1, "Long text should be split into multiple chunks"


def test_chunking_preserves_metadata():
    from langchain_core.documents import Document
    docs = [Document(page_content="Short text.", metadata={"source": "myfile.txt"})]
    chunks = SPLITTER.split_documents(docs)
    assert all(c.metadata["source"] == "myfile.txt" for c in chunks)


# ── Ingest dedup ──────────────────────────────────────────────────────────────

def test_ingest_skips_duplicate_chunks(tmp_path):
    """Second ingest of the same file should upsert 0 new chunks."""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Vigneshwar is proficient in Python and SQL.\n" * 5)

    log_path = tmp_path / "ingestion_log.json"

    with (
        patch("src.ingest.INGESTION_LOG_PATH", log_path),
        patch("src.ingest.get_vectorstore") as mock_vs,
    ):
        mock_vs.return_value = MagicMock()
        from src.ingest import ingest
        first  = ingest(test_file)
        second = ingest(test_file)

    assert first > 0,  "First ingest should upsert chunks"
    assert second == 0, "Second ingest should skip all (already ingested)"
