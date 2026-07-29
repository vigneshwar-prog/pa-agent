# 🧠 PA-Agent — Personal Knowledge RAG "Second Brain"

A personal-knowledge RAG system that answers natural-language questions about me using my own documents. Built phase by phase as a learning project — covers the full production RAG stack from ingestion to deployment.

---

## What It Does

Drop in any personal document (resume, ChatGPT export, notes, PDFs) → ask questions in plain English → get accurate, source-cited answers powered by a local LLM.

```
"What programming languages do I know?"
→ "You know Python, SQL, JavaScript, and Shell scripting." (Source: chatgpt_export.txt)
```

---

## Stack

| Layer | Technology | Why |
|---|---|---|
| **LLM** | Ollama + `llama3.1:8b` | Free, local, no API key, runs on Mac M-series |
| **Embeddings** | `BAAI/bge-large-en-v1.5` | Top MTEB score among open models, 1024-dim, free |
| **Vector DB** | Pinecone (free tier) | Managed, native metadata filtering, industry-standard |
| **Framework** | LangChain LCEL | Composable pipeline, swap any component via env vars |
| **API** | FastAPI | Async, Pydantic validation, Prometheus metrics |
| **UI** | Streamlit | Zero-friction chat interface |
| **Tracing** | LangFuse | LLM trace debugging, self-hosted |
| **CI/CD** | GitHub Actions | Lint → Test → Docker build on every push |

---

## Architecture

```
Raw Documents (txt/pdf/docx/csv/json/image/audio)
        │
        ▼
   src/ingest.py
   ├── FileTypeRouter   — picks the right loader per file type
   ├── RecursiveCharacterTextSplitter (600 tokens, 100 overlap)
   ├── Keyword category tagger  — skills / experience / education / etc.
   └── SHA-256 dedup  — skip already-ingested chunks
        │
        ▼
   src/vectorstore.py
   └── Pinecone (bge-large-en-v1.5 embeddings, namespace=personal)
        │
        ▼
   src/chain.py
   ├── MMR Retriever (k=6, fetch_k=20)
   ├── ChatOllama (llama3.1:8b, temp=0.3)
   ├── ChatPromptTemplate
   └── RunnableWithMessageHistory  — multi-turn memory
        │
        ▼
   src/api.py          →  POST /ask  →  { answer, session_id, sources }
   src/ui.py           →  Streamlit chat interface
```

---

## Build Phases

| Phase | Goal | Status |
|---|---|---|
| **1** | CLI prototype — ingest → Pinecone → terminal chat | 🔵 In Progress |
| **2** | Conversational memory (`RunnableWithMessageHistory`) | ⬜ Pending |
| **3** | RAGAs evaluation pipeline + golden dataset (50 Q&A pairs) | ⬜ Pending |
| **4** | FastAPI REST API + Streamlit chat UI | ⬜ Pending |
| **5** | LangFuse tracing + Prometheus/Grafana dashboards | ⬜ Pending |
| **6** | Docker Compose + GitHub Actions CI/CD + ngrok public URL | ⬜ Pending |

---

## Quick Start

### 1. Setup

```bash
# Clone and create virtual environment
git clone https://github.com/vigneshwar-prog/pa-agent.git
cd pa-agent
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

### 2. Ollama (local LLM)

```bash
# Install Ollama and pull the model (~4.7 GB)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b

# Run in a separate terminal — keep it running
ollama serve
```

### 3. Environment config

```bash
cp .env.example .env
# Open .env and fill in your Pinecone API key
```

Get a free Pinecone API key at [pinecone.io](https://pinecone.io) — no credit card needed.

### 4. Ingest your documents

```bash
python -m src.ingest data/raw/your_file.txt
```

### 5. Chat

```bash
# CLI
python -m src.main

# API server (Phase 4+)
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# UI (Phase 4+)
streamlit run src/ui.py
```

---

## Development

```bash
# Lint
ruff check src/

# Type check
mypy src/ --ignore-missing-imports

# Tests
pytest tests/ -v --tb=short

# RAGAs evaluation (Phase 3+)
python scripts/run_eval.py
```

---

## RAGAs Evaluation Targets

| Metric | Measures | Target |
|---|---|---|
| `faithfulness` | Answer grounded in context (no hallucination) | ≥ 0.85 |
| `answer_relevancy` | Answer addresses the question | ≥ 0.80 |
| `context_recall` | Retrieved context covers the ground truth | ≥ 0.75 |
| `context_precision` | Retrieved context is signal, not noise | ≥ 0.70 |

---

## Key Design Decisions

- **Pinecone over FAISS** — managed, native metadata filtering, always-on (no manual save/load)
- **bge-large over bge-small** — 1024-dim vs 384-dim; better semantic precision for personal data
- **MMR over plain similarity** — avoids retrieving 6 near-identical chunks from repeated content
- **SHA-256 dedup** — safe to re-run ingest without polluting the vector index
- **keyword category tagging** — enables filtered search (skills-only, experience-only) without LLM overhead
- **`normalize_embeddings=True`** — required for correct cosine similarity with BGE models

Full decision log with explanations → [`docs/learnings.html`](docs/learnings.html)

---

## Project Structure

```
pa-agent/
├── src/
│   ├── ingest.py       # Document loading, chunking, tagging, Pinecone upsert
│   ├── vectorstore.py  # Pinecone setup, embeddings, filtered search
│   ├── chain.py        # LangChain RAG chain with conversational memory
│   ├── api.py          # FastAPI — POST /ask, GET /health, /metrics
│   ├── ui.py           # Streamlit chat UI
│   ├── main.py         # CLI chat loop
│   └── logger.py       # Structured stdout logger
├── tests/
│   ├── test_ingest.py  # Chunking, dedup, category tagging
│   └── test_api.py     # API endpoint tests
├── scripts/
│   └── run_eval.py     # RAGAs evaluation pipeline
├── docs/
│   └── learnings.html  # Living log — decisions, gotchas, Q&A per phase
├── data/
│   └── raw/            # Drop your documents here
├── .env.example        # Config template
├── Dockerfile
├── Dockerfile.ui
└── docker-compose.yml
```
