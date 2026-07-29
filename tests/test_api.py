"""tests/test_api.py — FastAPI endpoint tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def _mock_chain_result(answer: str = "Test answer"):
    mock_doc = MagicMock()
    mock_doc.metadata = {"source": "test_source.txt"}
    return {"answer": answer, "context": [mock_doc]}


# ── /health ───────────────────────────────────────────────────────────────────

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── /ask ─────────────────────────────────────────────────────────────────────

def test_ask_returns_answer():
    with patch("src.api.get_chain") as mock_get_chain:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _mock_chain_result("Vigneshwar knows Python.")
        mock_get_chain.return_value = mock_chain

        resp = client.post("/ask", json={"question": "What does Vigneshwar know?"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Vigneshwar knows Python."
    assert "session_id" in data
    assert isinstance(data["sources"], list)


def test_ask_generates_session_id_when_missing():
    with patch("src.api.get_chain") as mock_get_chain:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _mock_chain_result()
        mock_get_chain.return_value = mock_chain

        resp = client.post("/ask", json={"question": "Test?"})

    assert resp.status_code == 200
    assert len(resp.json()["session_id"]) > 0


def test_ask_preserves_session_id():
    with patch("src.api.get_chain") as mock_get_chain:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _mock_chain_result()
        mock_get_chain.return_value = mock_chain

        resp = client.post(
            "/ask",
            json={"question": "Test?", "session_id": "my-session-123"},
        )

    assert resp.json()["session_id"] == "my-session-123"
