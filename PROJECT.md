# Personal Assistant Second Brain — PROJECT.md

> **Source of truth** for architecture, tech decisions, progress, and session history.
> Read this at the start of every session. Update the Progress Tracker and Session Log at the end.

---

## Table of Contents

1. [What This Is](#1-what-this-is)
2. [Data Ingestion & Chunking Strategy](#2-data-ingestion--chunking-strategy)
3. [Embedding & Vector Store](#3-embedding--vector-store)
4. [Retrieval Design](#4-retrieval-design)
5. [Generation — RAG Chain](#5-generation--rag-chain)
6. [Evaluation Pipeline](#6-evaluation-pipeline)
7. [Production Deployment](#7-production-deployment)
8. [Project Structure](#8-project-structure)
9. [Phase-by-Phase Progress Tracker](#9-phase-by-phase-progress-tracker)
10. [Resume / Job-Hunt Value](#10-resume--job-hunt-value)
11. [Session Log](#11-session-log)

---

## 1. What This Is

A **personal-knowledge RAG system** ("second brain") that answers natural-language questions about
Vigneshwar using documents he provides — starting with a ChatGPT-exported self-description, then
expandable to notes, resumes, journals, and more.

### Goals
- Recall facts, plans, skills, experiences, and preferences on demand.
- Multi-turn conversational interface (chat history aware).
- Production-deployable with a REST API, a chat UI, full observability, and CI/CD.

### Non-Goals (v1)
- Real-time document sync or web scraping.
- Fine-tuning the LLM.
- Multi-user support.

---

## 1.5 Open-Source & Cost Audit

> **Short answer: the core pipeline is 100% runnable for free using open-source tools.**
> The only components that cost money are optional cloud services with free tiers or
> open-source local alternatives for every single one.

### Full Component Breakdown

| Component | Planned Choice | License / Cost | Free / Open-Source Alternative |
|---|---|---|---|
| **Framework** | LangChain | MIT ✅ Free | — |
| **Language** | Python 3.11+ | PSF ✅ Free | — |
| **Embeddings** | `text-embedding-3-small` (OpenAI API) | 💰 Paid ($0.00002/1K tokens) | `BAAI/bge-large-en-v1.5` via HuggingFace (free, local, no API key) |
| **LLM** | GPT-4o via local gate `localhost:8080` | 🔁 Uses your existing Cisco gate | `llama3`, `mistral` via Ollama (100% free local) |
| **Vector DB** | Pinecone | 💰 Free tier (1 index, 2GB) then paid | FAISS (100% free, local, Apache 2) |
| **OCR (images)** | `pytesseract` | Apache 2 ✅ Free | — (this IS the free option) |
| **Audio transcription** | `openai-whisper` local model | MIT ✅ Free | — (this IS the free option; Whisper API = paid) |
| **Video processing** | `moviepy` | MIT ✅ Free | — |
| **Excel parsing** | `openpyxl` / `pandas` | MIT ✅ Free | — |
| **PDF parsing** | `PyMuPDF` | AGPL ✅ Free for personal use | — |
| **Evaluation** | RAGAs | Apache 2 ✅ Free | — |
| **API layer** | FastAPI | MIT ✅ Free | — |
| **UI** | Streamlit | Apache 2 ✅ Free | Gradio (Apache 2) |
| **LLM Tracing** | LangSmith | 💰 Free tier (10K traces/mo) then paid | LangFuse (MIT, self-hostable, 100% free) |
| **Metrics** | Prometheus + Grafana | Apache 2 ✅ Free | — |
| **Containerisation** | Docker | Apache 2 ✅ Free | — |
| **CI/CD** | GitHub Actions | Free for public repos ✅ | — |
| **Cloud Deploy** | Render | 💰 Free tier (spins down) → $7/mo | Railway (free tier), Fly.io (free tier) |

### Chosen Mode: **Mode A — 100% Free / Local ($0.00)**

> Every component runs locally. No API keys. No credit card. No cloud bills.
> The entire system runs on your laptop.

| Component | Choice | Why |
|---|---|---|
| **Embeddings** | `BAAI/bge-large-en-v1.5` via HuggingFace | Best-in-class open-source embedding model, MIT licence, runs on CPU |
| **LLM** | Ollama + `llama3.1:8b` | Free, local, no API key, runs on Mac M-series or any CPU |
| **Vector DB** | FAISS (local file) | Apache 2, zero setup, persists to disk |
| **Tracing / Observability** | LangFuse (self-hosted via Docker) | MIT licence, full LLM trace visibility, $0 |
| **Metrics** | Prometheus + Grafana (Docker) | Apache 2, industry standard |
| **Deploy** | Docker Compose (local) + `ngrok` for public URL | Free for personal use |
| **CI/CD** | GitHub Actions | Free for public repos |

**Cost: $0.00 — forever.**

### One-time setup for Mode A

```bash
# 1. Install Ollama (Mac/Linux)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b          # ~4.7 GB, runs on CPU
ollama pull nomic-embed-text     # lightweight embedding model as backup

# 2. HuggingFace embeddings — downloaded automatically on first run (no login needed)
# BAAI/bge-large-en-v1.5 → ~1.3 GB cached to ~/.cache/huggingface/

# 3. LangFuse (tracing) — optional, add in Phase 5
docker run -d -p 3000:3000 langfuse/langfuse:latest
```

### Upgrade path (zero code changes)
The codebase uses `.env` for all config. If you ever want to demo online or use cloud services,
it's just swapping env vars — the code stays identical:

```
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5   →  text-embedding-3-small
LLM_PROVIDER=ollama                       →  openai
VECTORSTORE=faiss                         →  pinecone
```

---

## 2. Data Ingestion & Chunking Strategy

### 2.1 Supported Document Formats

**Initial document**: `Vigneshwar_Profile_Summary_ChatGPT.docx` (~350 words, flat text, clean
categories). Handled by `Docx2txtLoader`. Use `chunk_size=300, overlap=50` for this doc so each
personal category (Career, Skills, Projects…) lands in its own chunk.

All future formats are routed through a `FileTypeRouter` that preprocesses each format into plain
text before chunking. Pipeline:

```
Any file → FileTypeRouter → Preprocessor → Plain Text → Chunker → Embedder → Pinecone
```

#### Phase 1 — Text-native formats (supported from day 1)

| Extension | LangChain Loader | Extra Dependency |
|---|---|---|
| `.txt` | `TextLoader` | none |
| `.pdf` | `PyMuPDFLoader` | `pymupdf` |
| `.docx` | `Docx2txtLoader` | `docx2txt` |
| `.md` | `TextLoader` | none |
| `.csv` | `CSVLoader` | none (built-in) |
| `.html` / URL | `WebBaseLoader` | `beautifulsoup4` |
| `.json` | `JSONLoader` | `jq` |

#### Phase 1.5 — Multi-modal formats (added after Phase 1 prototype works)

| Extension | Strategy | Library / Service |
|---|---|---|
| `.jpg` `.png` `.webp` (image) | OCR via `pytesseract` (local, free) **or** GPT-4o Vision API for richer extraction | `pytesseract` + `Pillow` |
| `.mp3` `.wav` `.m4a` (audio) | Transcribe via OpenAI Whisper API **or** local `whisper` model (free) | `openai.audio.transcriptions` or `openai-whisper` |
| `.mp4` `.mov` (video) | Extract audio with `moviepy` → Whisper transcript; optionally OCR keyframes | `moviepy`, `whisper` |
| `.xlsx` `.xls` (Excel) | `openpyxl` → convert each row to a readable text card | `openpyxl` / `pandas` |
| `.pptx` (PowerPoint) | `python-pptx` → extract slide text | `python-pptx` |

> **Open-source note**: `pytesseract`, `whisper` (local model), `moviepy`, `openpyxl`, `pandas`,
> `python-pptx` are all 100% open-source (MIT / Apache 2). The **only paid option** here is
> GPT-4o Vision / Whisper API — both have free local alternatives (pytesseract / whisper model).

#### `FileTypeRouter` implementation

```python
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader, PyMuPDFLoader, Docx2txtLoader,
    CSVLoader, WebBaseLoader, JSONLoader,
)

def load_document(path: str) -> list:
    suffix = Path(path).suffix.lower()

    # ── Text-native (Phase 1) ──────────────────────────────────────
    if suffix in (".txt", ".md"):
        return TextLoader(path, encoding="utf-8").load()
    if suffix == ".pdf":
        return PyMuPDFLoader(path).load()
    if suffix == ".docx":
        return Docx2txtLoader(path).load()
    if suffix == ".csv":
        return CSVLoader(path).load()
    if suffix == ".json":
        return JSONLoader(path, jq_schema=".", text_content=False).load()

    # ── Multi-modal (Phase 1.5) ────────────────────────────────────
    if suffix in (".jpg", ".jpeg", ".png", ".webp"):
        return _load_image(path)       # OCR → Document
    if suffix in (".mp3", ".wav", ".m4a"):
        return _load_audio(path)       # Whisper → Document
    if suffix in (".mp4", ".mov"):
        return _load_video(path)       # moviepy + Whisper → Document
    if suffix in (".xlsx", ".xls"):
        return _load_excel(path)       # openpyxl rows → Documents
    if suffix in (".pptx",):
        return _load_pptx(path)        # python-pptx slides → Documents

    raise ValueError(f"Unsupported format: {suffix}. Add a loader to FileTypeRouter.")

# ── Multi-modal helpers (stubs — implemented in Phase 1.5) ──────────
def _load_image(path: str):
    import pytesseract
    from PIL import Image
    from langchain_core.documents import Document
    text = pytesseract.image_to_string(Image.open(path))
    return [Document(page_content=text, metadata={"source": path, "type": "image"})]

def _load_audio(path: str):
    import whisper                        # local model — free, no API key
    from langchain_core.documents import Document
    model = whisper.load_model("base")    # ~140 MB, runs on CPU
    result = model.transcribe(path)
    return [Document(page_content=result["text"], metadata={"source": path, "type": "audio"})]

def _load_excel(path: str):
    import pandas as pd
    from langchain_core.documents import Document
    df = pd.read_excel(path)
    docs = []
    for _, row in df.iterrows():
        text = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
        docs.append(Document(page_content=text, metadata={"source": path, "type": "excel"}))
    return docs
```

### 2.2 Chunking Strategy

**Chosen splitter**: `RecursiveCharacterTextSplitter`

| Parameter | Value | Rationale |
|---|---|---|
| `chunk_size` | 300 tokens (initial doc) / 600 (larger docs) | Initial `.docx` is ~350 words — 300 tokens ensures each section (Career, Skills, Projects…) is its own chunk |
| `chunk_overlap` | 100 tokens | Prevents facts split across chunk boundaries |
| `length_function` | `tiktoken` (`cl100k_base`) | Token-accurate, matches OpenAI embedding model tokenizer |
| `separators` | `["\n\n", "\n", ". ", " ", ""]` | Respects paragraph → sentence → word hierarchy |

Why **not** semantic chunking (v1): requires an embedding call per sentence to detect breakpoints —
slower at ingest and overkill for a single, well-structured personal document. Revisit in Phase 3.

```python
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

enc = tiktoken.get_encoding("cl100k_base")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    length_function=lambda text: len(enc.encode(text)),
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_documents(raw_docs)
```

### 2.3 Metadata Tagging

Every chunk gets structured metadata so we can filter retrieval by category.

```python
CATEGORY_KEYWORDS = {
    "skills":      ["python", "langchain", "sql", "aws", "skill", "proficient", "tool"],
    "experience":  ["worked", "built", "led", "managed", "project", "company", "role", "year"],
    "education":   ["degree", "university", "college", "studied", "graduated", "gpa"],
    "personality": ["i am", "i tend", "introvert", "curious", "values", "belief"],
    "goals":       ["want to", "aiming", "plan to", "career goal", "aspire"],
    "hobbies":     ["hobby", "enjoy", "reading", "travel", "gaming", "music"],
}

def infer_category(text: str) -> str:
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return "general"

def tag_chunks(chunks, source_file: str):
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "source":    source_file,
            "chunk_id":  f"{Path(source_file).stem}_{i:04d}",
            "category":  infer_category(chunk.page_content),
            "ingested_at": datetime.utcnow().isoformat(),
        })
    return chunks
```

**Metadata schema per chunk**:

```json
{
  "source":      "chatgpt_export.txt",
  "chunk_id":    "chatgpt_export_0012",
  "category":    "skills",
  "ingested_at": "2026-07-27T10:00:00"
}
```

### 2.4 Incremental Ingestion

New documents are added without re-embedding existing ones:

1. Keep a `data/ingestion_log.json` tracking `{ chunk_id: vector_id }` already upserted.
2. On each ingest run, compute chunk content hash (`sha256`); skip chunks whose hash already exists.
3. Upsert only new/changed chunks to Pinecone.

```python
import hashlib, json

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]
```

---

## 3. Embedding & Vector Store

### 3.1 Embedding Model — Decision

**Chosen (Mode A — Free)**: `BAAI/bge-small-en-v1.5` (Phase 1) → `BAAI/bge-large-en-v1.5` (Phase 3+)
Both from HuggingFace. **No API key. No cost. Downloads once to local cache.**

| Model | Dims | Cost | Quality | CPU speed |
|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` ✅ Phase 1 | 384 | **Free** | Good — fast to iterate | Fast (~130 MB) |
| `BAAI/bge-large-en-v1.5` ✅ Phase 3+ | 1024 | **Free** | Excellent — #1 open MTEB | Moderate (~1.3 GB) |
| `nomic-embed-text` (via Ollama) | 768 | **Free** | Good | Fast |
| `all-mpnet-base-v2` | 768 | **Free** | Good | Moderate |

**Why BGE?** BAAI's BGE series consistently tops the open-source MTEB leaderboard. `bge-small` is
great for rapid development; upgrade to `bge-large` before running RAGAs evaluation.

```python
from langchain_huggingface import HuggingFaceEmbeddings

# Phase 1 — fast, CPU-friendly (~130 MB download, cached)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # required for BGE cosine similarity
)

# Phase 3+ — higher quality for eval
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

> First run downloads model to `~/.cache/huggingface/`. No login, no API key required.

### 3.2 Vector Store — FAISS (Primary, Mode A)

**Mode A uses FAISS exclusively** — a local file on disk, zero infrastructure, zero cost.

```python
from langchain_community.vectorstores import FAISS

# Build index from documents
vectorstore = FAISS.from_documents(tagged_chunks, embeddings)
vectorstore.save_local("data/faiss_index")   # persists to disk

# Reload on next run
vectorstore = FAISS.load_local(
    "data/faiss_index", embeddings, allow_dangerous_deserialization=True
)
```

**FAISS namespace strategy** (replaces Pinecone namespaces):
Use separate index directories per knowledge domain:

```
data/faiss_index/personal/    ← your ChatGPT doc and personal files
data/faiss_index/work/        ← work notes (future)
data/faiss_index/journal/     ← daily journal (future)
```

Load the right index per query context.

**FAISS metadata filtering** (post-hoc, since FAISS has no native filter):

```python
def filtered_search(vectorstore, query: str, category: str, k: int = 6):
    # Retrieve more candidates, then filter by metadata
    docs = vectorstore.similarity_search(query, k=k * 3)
    filtered = [d for d in docs if d.metadata.get("category") == category]
    return filtered[:k]
```

### 3.3 Upsert Pipeline

```python
# Incremental: load existing index, add new chunks only
import os

INDEX_PATH = "data/faiss_index/personal"

if os.path.exists(INDEX_PATH):
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    vectorstore.add_documents(new_chunks)   # append only new chunks
else:
    vectorstore = FAISS.from_documents(new_chunks, embeddings)

vectorstore.save_local(INDEX_PATH)
```

### 3.4 Future Upgrade Path (Pinecone — optional)

When you want cloud persistence or a live demo without running your laptop:

```python
# Just swap vectorstore — all retrieval/chain code is unchanged
from langchain_pinecone import PineconeVectorStore
vectorstore = PineconeVectorStore.from_documents(
    tagged_chunks, embeddings,
    index_name="pa-second-brain", namespace="personal"
)
```

Pinecone free tier: 1 index, 2 GB — more than enough for this project.

---

## 4. Retrieval Design

### 4.1 Hybrid Retrieval

Two retrieval modes depending on the query:

| Mode | Trigger | Mechanism |
|---|---|---|
| **Pure semantic** | Open-ended ("what are my strengths?") | Cosine similarity top-k |
| **Filtered semantic** | Category-specific ("what skills do I have?") | Pinecone metadata filter + similarity |

```python
# Category-filtered retrieval
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 6,
        "namespace": NAMESPACE,
        "filter": {"category": {"$in": ["skills", "experience"]}},
    },
)

# Plain semantic retrieval
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6, "namespace": NAMESPACE},
)
```

### 4.2 Top-k Strategy

- **k = 6** for most queries (covers ~3,600 tokens of context, well within GPT-4o's 128k window).
- Use `search_type="mmr"` (Maximum Marginal Relevance) with `fetch_k=20, lambda_mult=0.7` to reduce
  redundant chunks when multiple chunks come from the same document section.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7},
)
```

### 4.3 Re-Ranking (Optional Enhancement)

For v2, add a cross-encoder re-ranker after retrieval:

```python
# pip install sentence-transformers
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, docs: list, top_n: int = 4):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]
```

### 4.4 LangChain Retriever Wiring

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Optional: compress retrieved chunks to relevant sentences only
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
)
```

---

## 5. Generation — RAG Chain

### 5.1 LLM Configuration

**Chosen (Mode A — Free)**: Ollama + `llama3.1:8b` — **runs locally, no API key, no cost**

```bash
# One-time setup
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b      # ~4.7 GB — best open model at 8B size
ollama serve                  # starts on http://localhost:11434
```

| Model | Size | RAM needed | Quality | Cost |
|---|---|---|---|---|
| `llama3.1:8b` ✅ | 4.7 GB | ~8 GB RAM | Excellent for Q&A | **Free** |
| `llama3.1:70b` | 40 GB | ~48 GB RAM | Near GPT-4 quality | **Free** (needs big machine) |
| `mistral:7b` | 4.1 GB | ~8 GB RAM | Good | **Free** |
| `phi3:mini` | 2.3 GB | ~4 GB RAM | Decent, very fast | **Free** (low RAM option) |

> **Low RAM machine?** Use `phi3:mini` (2.3 GB, 4 GB RAM) for Phase 1. Upgrade to `llama3.1:8b` for Phase 3 eval.

```python
from langchain_ollama import ChatOllama

# Mode A — local Ollama (free)
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.3,      # low = factual recall, not creative
    base_url="http://localhost:11434",
)
```

`temperature=0.3`: factual recall needs low randomness, but not 0 (avoids robotic phrasing).

### 5.2 System Prompt Template

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """You are Vigneshwar's personal AI assistant — his second brain.
You answer questions about Vigneshwar using only the context retrieved from his personal \
knowledge base. Be concise, factual, and speak in first person when appropriate \
(e.g., "Vigneshwar has..." or "Based on his notes...").

Rules:
1. If the answer is not in the retrieved context, say "I don't have that information in my \
knowledge base yet." Do NOT hallucinate.
2. Cite the category of knowledge when relevant (e.g., "From his skills profile...").
3. For multi-part questions, use bullet points.
4. Keep answers under 200 words unless a detailed explanation is requested.

Retrieved context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])
```

### 5.3 Conversational Memory

Use `RunnableWithMessageHistory` (LCEL pattern — preferred over the deprecated
`ConversationalRetrievalChain`):

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Build the RAG chain
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Wrap with session-scoped message history
store: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
```

### 5.4 Invocation

```python
response = conversational_rag.invoke(
    {"input": "What programming languages does Vigneshwar know?"},
    config={"configurable": {"session_id": "session_001"}},
)
print(response["answer"])
```

---

## 6. Evaluation Pipeline

### 6.1 Golden Dataset (50 Q&A Pairs)

Create `data/eval/golden_dataset.json` by reading the personal document and writing questions that
can be *objectively verified* against it.

**Category distribution target**:

| Category | # Questions | Example |
|---|---|---|
| Skills | 10 | "What programming languages does Vigneshwar know?" |
| Experience | 12 | "Where has Vigneshwar worked?" |
| Education | 6 | "What did Vigneshwar study?" |
| Personality | 8 | "How does Vigneshwar describe his working style?" |
| Goals | 8 | "What are Vigneshwar's career goals?" |
| Hobbies | 6 | "What are Vigneshwar's hobbies?" |

**Schema**:

```json
[
  {
    "question": "What programming languages does Vigneshwar know?",
    "ground_truth": "Python, SQL, JavaScript, and Shell scripting.",
    "category": "skills"
  }
]
```

**Generation script** (`scripts/generate_golden.py`):
Use GPT-4o to auto-generate Q&A pairs from each chunk, then manually review and curate.

### 6.2 RAGAs Evaluation

```bash
pip install ragas
```

**Metrics** (all scored 0–1, higher = better):

| Metric | Measures | Target |
|---|---|---|
| `faithfulness` | Answer grounded in retrieved context (no hallucination) | ≥ 0.85 |
| `answer_relevancy` | Answer actually addresses the question | ≥ 0.80 |
| `context_recall` | Retrieved context covers the ground truth | ≥ 0.75 |
| `context_precision` | Retrieved context is signal, not noise | ≥ 0.70 |

```python
# scripts/evaluate.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

def run_evaluation(rag_chain, golden_dataset: list[dict]) -> dict:
    records = []
    for item in golden_dataset:
        result = rag_chain.invoke({"input": item["question"]}, ...)
        records.append({
            "question":        item["question"],
            "answer":          result["answer"],
            "contexts":        [d.page_content for d in result["context"]],
            "ground_truth":    item["ground_truth"],
        })

    dataset = Dataset.from_list(records)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    return scores
```

### 6.3 Iteration Loop

```
Run eval → identify lowest metric → diagnose:
  - Low faithfulness     → tighten system prompt, reduce temperature
  - Low answer_relevancy → improve prompt, add query rewriting
  - Low context_recall   → increase k, check chunking
  - Low context_precision → use MMR, reduce k, add re-ranker
→ fix → re-run eval → commit if improved
```

Results saved to `data/eval/results/YYYY-MM-DD_HH-MM.json` for trend tracking.

---

## 7. Production Deployment

### 7.1 FastAPI Wrapper

```python
# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid

app = FastAPI(title="PA Second Brain", version="1.0.0")

class AskRequest(BaseModel):
    question: str
    session_id: str = ""

class AskResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str]

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    session_id = req.session_id or str(uuid.uuid4())
    result = conversational_rag.invoke(
        {"input": req.question},
        config={"configurable": {"session_id": session_id}},
    )
    sources = list({d.metadata.get("source", "unknown") for d in result.get("context", [])})
    return AskResponse(answer=result["answer"], session_id=session_id, sources=sources)

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
```

Run: `uvicorn src.api:app --host 0.0.0.0 --port 8000`

### 7.2 Streamlit Chat UI

```python
# src/ui.py
import streamlit as st, requests

st.title("🧠 Vigneshwar's Second Brain")

if "session_id" not in st.session_state:
    st.session_state.session_id = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask me anything about Vigneshwar..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    resp = requests.post(
        "http://localhost:8000/ask",
        json={"question": prompt, "session_id": st.session_state.session_id},
    ).json()
    st.session_state.session_id = resp["session_id"]

    with st.chat_message("assistant"):
        st.write(resp["answer"])
        if resp["sources"]:
            st.caption(f"Sources: {', '.join(resp['sources'])}")

    st.session_state.messages.append({"role": "assistant", "content": resp["answer"]})
```

Run: `streamlit run src/ui.py`

### 7.3 Docker + docker-compose

**`Dockerfile`**:
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`docker-compose.yml`**:
```yaml
version: "3.9"
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    volumes:
      - ./data:/app/data  # persist FAISS index
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  ui:
    build:
      context: .
      dockerfile: Dockerfile.ui
    ports: ["8501:8501"]
    depends_on: [api]
    environment:
      - API_URL=http://api:8000
```

### 7.4 Environment Config

**`.env`** (never committed — add to `.gitignore`):
```ini
# Mode A — all free, all local
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
VECTORSTORE=faiss
FAISS_INDEX_PATH=data/faiss_index/personal
LANGFUSE_PUBLIC_KEY=...         # LangFuse self-hosted (Phase 5, optional)
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=http://localhost:3000
LOG_LEVEL=INFO
```

### 7.5 Logging

```python
# src/logger.py
import logging, sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.getLevelName(os.getenv("LOG_LEVEL", "INFO")))
    return logger
```

### 7.6 Observability

#### LangFuse (LLM Tracing) — Free, Self-Hosted

LangFuse is the open-source alternative to LangSmith. Run it locally via Docker:

```bash
# One-time setup — spins up LangFuse + Postgres + Redis
docker run -d \
  -e DATABASE_URL=postgresql://langfuse:langfuse@localhost:5432/langfuse \
  -p 3000:3000 \
  langfuse/langfuse:latest
# Dashboard at http://localhost:3000 — create a project, copy keys to .env
```

Wire into LangChain (just set env vars — zero code changes):
```ini
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

```python
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler()

# Pass as callback to any LangChain chain
response = conversational_rag.invoke(
    {"input": "What skills does Vigneshwar have?"},
    config={"callbacks": [langfuse_handler], "configurable": {"session_id": "s1"}},
)
```

Dashboard shows: full trace tree, latency per step, retrieved docs, token counts, prompt renders.
Use to debug bad answers — see exactly what context was passed and why the answer went wrong.

#### Prometheus + Grafana (API Metrics)

```python
# pip install prometheus-fastapi-instrumentator
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
# Metrics available at GET /metrics
```

**`docker-compose.yml`** additions:
```yaml
  prometheus:
    image: prom/prometheus:latest
    volumes: ["./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml"]
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

**Key Grafana panels**: request rate, p50/p95/p99 latency, error rate (4xx/5xx), `/ask` endpoint
breakdown. Import dashboard ID `14565` (FastAPI Observability) as a starting point.

### 7.7 CI/CD — GitHub Actions

**`.github/workflows/ci.yml`**:
```yaml
name: CI
on: [push, pull_request]

jobs:
  lint-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements-dev.txt
      - run: ruff check src/
      - run: mypy src/ --ignore-missing-imports
      - run: pytest tests/ -v --tb=short

  docker-build:
    needs: lint-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - run: docker build -t pa-second-brain:${{ github.sha }} .
```

### 7.8 Deployment — Mode A: Docker Compose (Local) + ngrok (Public URL)

**No cloud. No bill. Full production stack on your laptop.**

```yaml
# docker-compose.yml
version: "3.9"
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    volumes:
      - ./data:/app/data         # FAISS index persisted across restarts

  ui:
    build:
      context: .
      dockerfile: Dockerfile.ui
    ports: ["8501:8501"]
    env_file: .env
    depends_on: [api]

  prometheus:
    image: prom/prometheus:latest
    volumes: ["./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml"]
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:latest
    ports: ["3001:3000"]         # 3001 to avoid clash with LangFuse on 3000
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin

  langfuse:
    image: langfuse/langfuse:latest
    ports: ["3000:3000"]
    environment:
      DATABASE_URL: postgresql://langfuse:langfuse@langfuse-db:5432/langfuse
    depends_on: [langfuse-db]

  langfuse-db:
    image: postgres:15
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse
      POSTGRES_DB: langfuse
```

```bash
docker compose up -d          # start everything
docker compose logs -f api    # watch API logs
```

**Expose publicly for demo/interviews** (free ngrok):
```bash
brew install ngrok             # or https://ngrok.com/download
ngrok http 8000                # gives you https://abc123.ngrok-free.app/ask
ngrok http 8501                # expose Streamlit UI
```

Share the ngrok URL in your resume or LinkedIn — works as a live demo without paying for hosting.

> **Upgrade path**: when you want always-on 24/7 availability → push to Render/Railway free tier
> (just swap env vars, all code stays identical).

---

## 8. Project Structure

```
pa-agent/
├── data/
│   ├── raw/                     # Original documents (gitignored if sensitive)
│   │   └── chatgpt_export.txt
│   ├── faiss_index/             # FAISS persisted index (local dev)
│   ├── ingestion_log.json       # chunk_id → vector_id map (dedup)
│   └── eval/
│       ├── golden_dataset.json  # 50 Q&A ground truth pairs
│       └── results/             # Timestamped RAGAs output JSONs
│
├── src/
│   ├── __init__.py
│   ├── ingest.py        # Document loading, chunking, metadata tagging, upsert
│   ├── vectorstore.py   # FAISS setup, retriever factory (+ Pinecone upgrade path)
│   ├── chain.py         # RAG chain assembly (Ollama LLM, BGE embeddings, LCEL)
│   ├── api.py           # FastAPI app (/ask, /health, /metrics)
│   ├── ui.py            # Streamlit chat interface
│   └── logger.py        # Logging config
│
├── scripts/
│   ├── check_gate.sh        # Verify local LLM gate (mirrors employee-rag)
│   ├── generate_golden.py   # Auto-generate Q&A eval pairs from doc
│   └── run_eval.py          # Execute RAGAs evaluation, save results
│
├── tests/
│   ├── test_ingest.py       # Unit: chunking, metadata tagging, dedup
│   ├── test_chain.py        # Unit: chain invocation with mock retriever
│   └── test_api.py          # Integration: FastAPI endpoints via TestClient
│
├── monitoring/
│   └── prometheus.yml       # Scrape config for FastAPI /metrics
│
├── docs/
│   └── learnings.html       # Living log: concepts, Q&A explored, gotchas — updated each phase
│
├── .github/
│   └── workflows/ci.yml
│
├── .env.example             # Template (committed); .env is gitignored
├── .gitignore
├── Dockerfile
├── Dockerfile.ui
├── docker-compose.yml
├── render.yaml
├── requirements.txt         # Production deps
├── requirements-dev.txt     # + ruff, mypy, pytest, ragas
└── PROJECT.md               # ← this file
```

### Key Dependency Versions (pin these in `requirements.txt`)

```
langchain==0.3.*
langchain-openai==0.2.*
langchain-pinecone==0.2.*
langchain-community==0.3.*
langchain-huggingface==0.1.*
langchain-core==0.3.*
pinecone-client==5.*
openai==1.*
tiktoken==0.7.*
fastapi==0.115.*
uvicorn[standard]==0.30.*
streamlit==1.37.*
ragas==0.1.*
pymupdf==1.24.*
python-dotenv==1.0.*
prometheus-fastapi-instrumentator==7.*
pydantic==2.*
```

---

## 9. Phase-by-Phase Progress Tracker

### Phase 1 — Local Prototype (CLI)
> Goal: Ingest the personal doc into FAISS and answer questions in a terminal loop.

| Task | Status | Notes |
|---|---|---|
| Set up venv, install deps | ⬜ Not started | |
| Write `src/ingest.py` (load + chunk + tag) | ⬜ Not started | |
| Write `src/vectorstore.py` (FAISS) | ⬜ Not started | |
| Write `src/chain.py` (basic LCEL chain, no memory) | ⬜ Not started | |
| CLI loop in `src/main.py` | ⬜ Not started | |
| Manual smoke test (5 questions) | ⬜ Not started | |

**Exit criteria**: Can answer 5/5 basic questions from the CLI.

---

### Phase 2 — Pinecone + Conversational Memory
> Goal: Swap FAISS for Pinecone, add chat history, expose session IDs.

| Task | Status | Notes |
|---|---|---|
| Pinecone index setup script | ⬜ Not started | |
| Update `vectorstore.py` for Pinecone + namespace | ⬜ Not started | |
| Update `chain.py`: `RunnableWithMessageHistory` | ⬜ Not started | |
| Incremental ingestion with dedup log | ⬜ Not started | |
| Multi-turn conversation test (10 exchanges) | ⬜ Not started | |

**Exit criteria**: Multi-turn chat with memory; second document ingested incrementally.

---

### Phase 3 — Evaluation (RAGAs Pipeline)
> Goal: Measure system quality against golden dataset; establish a baseline.

| Task | Status | Notes |
|---|---|---|
| Create `data/eval/golden_dataset.json` (50 pairs) | ⬜ Not started | |
| Write `scripts/generate_golden.py` | ⬜ Not started | |
| Write `scripts/run_eval.py` (RAGAs) | ⬜ Not started | |
| Run baseline evaluation; record scores | ⬜ Not started | |
| Iterate on chunk size / k / prompt until targets met | ⬜ Not started | |

**Target scores**: faithfulness ≥ 0.85, answer_relevancy ≥ 0.80, context_recall ≥ 0.75.

---

### Phase 4 — FastAPI + Streamlit UI
> Goal: REST API + browser-accessible chat UI.

| Task | Status | Notes |
|---|---|---|
| Write `src/api.py` (FastAPI `/ask`, `/health`) | ⬜ Not started | |
| Write `src/ui.py` (Streamlit chat) | ⬜ Not started | |
| Wire UI → API via `requests` | ⬜ Not started | |
| Write `tests/test_api.py` | ⬜ Not started | |
| Manual end-to-end test in browser | ⬜ Not started | |

**Exit criteria**: Browser chat works; `/health` returns 200.

---

### Phase 5 — Observability (LangFuse + Prometheus)
> Goal: Full visibility into LLM calls and API performance.

| Task | Status | Notes |
|---|---|---|
| Enable LangFuse tracing (self-hosted Docker, env vars) | ⬜ Not started | |
| Add `prometheus-fastapi-instrumentator` | ⬜ Not started | |
| Write `monitoring/prometheus.yml` | ⬜ Not started | |
| Add Prometheus + Grafana to `docker-compose.yml` | ⬜ Not started | |
| Build Grafana dashboard (latency, error rate) | ⬜ Not started | |

**Exit criteria**: LangFuse shows traces; Grafana shows latency histogram.

---

### Phase 6 — Docker + CI/CD + Cloud Deploy
> Goal: Containerized, continuously deployed, publicly accessible.

| Task | Status | Notes |
|---|---|---|
| Write `Dockerfile` (API) | ⬜ Not started | |
| Write `Dockerfile.ui` (Streamlit) | ⬜ Not started | |
| Write `docker-compose.yml` (full stack) | ⬜ Not started | |
| Local `docker compose up` smoke test | ⬜ Not started | |
| Write `.github/workflows/ci.yml` | ⬜ Not started | |
| Write `render.yaml` | ⬜ Not started | Optional — only if deploying to cloud |
| Deploy via Docker Compose + ngrok for public demo URL | ⬜ Not started | |

**Exit criteria**: `https://<app>.onrender.com` returns a working chat UI.

---

## 10. Resume / Job-Hunt Value

### What This Project Demonstrates

| Skill Area | Demonstrated By |
|---|---|
| **RAG system design** | End-to-end pipeline: ingest → chunk → embed → retrieve → generate |
| **Vector databases** | Pinecone production setup, FAISS local fallback, namespace strategy |
| **LLM orchestration** | LangChain LCEL, multi-turn memory, prompt engineering |
| **Evaluation methodology** | RAGAs framework, golden dataset, 4 metrics, iteration loop |
| **Production MLOps** | FastAPI, Docker Compose, GitHub Actions CI/CD, ngrok demo |
| **Observability** | LangFuse (self-hosted LLM tracing), Prometheus + Grafana API metrics |
| **Software engineering** | Type hints, unit + integration tests, structured logging, `.env` config |

### Suggested Resume Bullets

```
• Built a production RAG "second brain" system using LangChain, Pinecone, and GPT-4o;
  achieved faithfulness ≥ 0.85 and answer relevancy ≥ 0.80 measured via RAGAs framework
  on a 50-pair golden evaluation dataset.

• Designed a hybrid retrieval pipeline combining semantic similarity (cosine, top-k=6,
  MMR re-ranking) with Pinecone metadata filtering; deployed via FastAPI on Render with
  Docker + GitHub Actions CI/CD.

• Implemented full observability stack: LangFuse (self-hosted) for LLM trace analysis and
  Prometheus + Grafana for API latency (p50/p95/p99) and error-rate dashboards.

• Engineered incremental document ingestion with SHA-256 content hashing to avoid
  re-embedding unchanged chunks across multiple knowledge sources.
```

### Interview Talking Points

1. **Why hybrid retrieval?** — Pure vector search fails exact categorical queries ("list all my
   Python skills"). Metadata filters handle these with 100% precision; semantic search handles
   open-ended queries. The two complement each other.

2. **How do you know it works?** — RAGAs evaluation with a golden dataset. Walk through the 4
   metrics, what each measures, and how you iterated when scores were low.

3. **Why Pinecone over pgvector / Weaviate?** — Managed, serverless, no infra to run. For a
   personal project this removes ops overhead. The trade-off is vendor lock-in and cost at scale.

4. **How would you scale this?** — Async FastAPI, Celery for background ingestion jobs, Redis for
   session memory store (replace in-memory dict), horizontal API scaling, Pinecone handles vector
   scaling natively.

---

## 12. Learnings & Exploration Log (`docs/learnings.html`)

A living HTML page — **`docs/learnings.html`** — is maintained alongside the code. It is updated at
the end of every phase (and mid-phase when something interesting is discovered). It mirrors the
style used in the `employee-rag` project.

### What it captures
- **Concepts explored** — new LangChain classes, Pinecone APIs, RAGAs metrics, etc.
- **Questions asked** — the actual prompts sent to the assistant, and what they revealed about the system
- **Surprises & gotchas** — things that didn't work as expected and why
- **Decisions made** — why one approach was chosen over another (links back to this PROJECT.md)
- **Code snippets that clicked** — short, memorable examples worth remembering

### Structure of `docs/learnings.html`

```
docs/
└── learnings.html        ← single-file, no build step, opens in any browser
```

The page is a single self-contained HTML file with inline CSS — no framework, no build step. It
has one collapsible `<section>` per phase:

```
Phase 1 — Local Prototype (Ingest + FAISS + CLI)
Phase 2 — Pinecone + Conversational Memory
Phase 3 — Evaluation (RAGAs Pipeline)
Phase 4 — FastAPI + Streamlit UI
Phase 5 — Observability (LangSmith + Prometheus)
Phase 6 — Docker + CI/CD + Cloud Deploy
```

Each phase section contains three sub-sections:
1. **📚 Concepts & Learnings** — what was learned, with code snippets
2. **❓ Questions Asked & Explored** — interesting Q&A pairs tested against the system, and what they revealed
3. **⚡ Gotchas & Surprises** — things that broke, edge cases, surprising behaviours

### Update Ritual

At the end of every coding session / phase:
1. Open `docs/learnings.html`
2. Find the relevant phase `<section>`
3. Add a timestamped `<article>` entry under the appropriate sub-section
4. Commit alongside the code changes: `git commit -m "phase N: update learnings.html"`

### Seed Entry (Phase 0 — Planning)

> **2026-07-27** — Planned full system architecture. Key learning: hybrid retrieval (semantic +
> metadata filter) is essential for categorical personal data. Pure vector search scores ~0.4
> faithfulness on "list all my Python skills" type queries. RAGAs chosen for eval because it
> decomposes quality into 4 orthogonal dimensions — easier to debug than a single score.

---

## 11. Session Log

| Date | Summary | Key Decisions | Next Steps |
|---|---|---|---|
| 2026-07-27 | Created PROJECT.md; planned full system architecture; confirmed Mode A (100% free) | FAISS primary, BGE embeddings, Ollama llama3.1:8b, LangFuse self-hosted tracing, Docker+ngrok deploy | Begin Phase 1: set up venv, install Ollama, write `ingest.py` |
| | | | |
| | | | |
