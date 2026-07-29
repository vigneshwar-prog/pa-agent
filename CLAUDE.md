# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A **personal-knowledge RAG "second brain"** that answers natural-language questions about Vigneshwar using his own documents. 100% free/local stack — no API keys, no cloud bills. See `PROJECT.md` for the full architecture spec, phase tracker, and session log.

## One-Time Setup

```bash
# Python environment
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Ollama (LLM)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b      # ~4.7 GB
ollama serve                  # runs on http://localhost:11434

# HuggingFace embeddings download automatically on first run (~130 MB for bge-small)
# Copy and fill in .env
cp .env.example .env
```

## Common Commands

```bash
# Run the full stack
docker compose up -d
docker compose logs -f api

# Ingest documents into FAISS
python -m src.ingest data/raw/chatgpt_export.txt

# CLI chat loop (Phase 1)
python -m src.main

# API server (Phase 4+)
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Streamlit UI (Phase 4+)
streamlit run src/ui.py
```

## Development & Testing

```bash
# Lint
ruff check src/

# Type check
mypy src/ --ignore-missing-imports

# Run all tests
pytest tests/ -v --tb=short

# Run a single test file
pytest tests/test_ingest.py -v

# Run a single test
pytest tests/test_ingest.py::test_chunking -v

# RAGAs evaluation
python scripts/run_eval.py
```

## Architecture Overview

The pipeline flows: **Raw documents → `ingest.py` → `vectorstore.py` → `chain.py` → `api.py` / `ui.py`**

### `src/` modules

| File | Role |
|---|---|
| `ingest.py` | `FileTypeRouter` (txt/pdf/docx/csv/json/image/audio/video/xlsx), `RecursiveCharacterTextSplitter` (chunk_size=600, overlap=100), keyword-based metadata tagging (`category` field) |
| `vectorstore.py` | FAISS index at `data/faiss_index/personal/`; incremental upsert via SHA-256 dedup; `filtered_search()` for post-hoc metadata filtering; Pinecone is the cloud upgrade path (swap env vars only) |
| `chain.py` | LangChain LCEL RAG chain; `ChatOllama(model="llama3.1:8b", temperature=0.3)`; `RunnableWithMessageHistory` for multi-turn memory; `search_type="mmr"` with `k=6, fetch_k=20` |
| `api.py` | FastAPI; `POST /ask` returns `{answer, session_id, sources}`; `GET /health`; Prometheus metrics at `/metrics` |
| `ui.py` | Streamlit chat UI that calls the API; session state holds `session_id` and message history |
| `logger.py` | Structured stdout logger; level controlled by `LOG_LEVEL` env var |

### Key design decisions

- **Embeddings**: `BAAI/bge-small-en-v1.5` (Phase 1) → `BAAI/bge-large-en-v1.5` (Phase 3+). Must pass `normalize_embeddings=True` for BGE cosine similarity to work correctly.
- **Memory**: In-memory `dict[session_id → ChatMessageHistory]` in Phase 2. Upgrade to Redis for production by swapping `get_session_history()` in `chain.py`.
- **FAISS namespaces**: Separate index directories per knowledge domain (`personal/`, `work/`, `journal/`) rather than a single index.
- **Chunking**: `tiktoken cl100k_base` for token-accurate length function. Separators: `["\n\n", "\n", ". ", " ", ""]`.
- **Metadata filtering**: FAISS has no native filter — retrieve `k*3` candidates, then filter by `metadata["category"]` in Python.
- **LangFuse tracing**: Pass `CallbackHandler()` to chain invocations. Set `LANGFUSE_*` env vars to activate — zero code changes.

### Environment config (`.env`)

```ini
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
VECTORSTORE=faiss
FAISS_INDEX_PATH=data/faiss_index/personal
LOG_LEVEL=INFO
# Optional — Phase 5 LangFuse tracing
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=http://localhost:3000
```

Swapping to cloud services (Pinecone, OpenAI) requires only env var changes — no code changes.

## RAGAs Evaluation Targets

| Metric | Target |
|---|---|
| `faithfulness` | ≥ 0.85 |
| `answer_relevancy` | ≥ 0.80 |
| `context_recall` | ≥ 0.75 |
| `context_precision` | ≥ 0.70 |

Results saved to `data/eval/results/YYYY-MM-DD_HH-MM.json`. Low faithfulness → tighten prompt / lower temperature. Low context_recall → increase `k`. Low context_precision → switch to MMR or add cross-encoder re-ranker.

## Build Phases (check `PROJECT.md` §9 for current status)

1. **Phase 1** — CLI prototype: ingest → FAISS → terminal loop
2. **Phase 2** — Pinecone + conversational memory (`RunnableWithMessageHistory`)
3. **Phase 3** — RAGAs eval pipeline + golden dataset (50 Q&A pairs)
4. **Phase 4** — FastAPI + Streamlit UI
5. **Phase 5** — LangFuse tracing + Prometheus/Grafana dashboards
6. **Phase 6** — Docker Compose + GitHub Actions CI/CD + ngrok public URL
