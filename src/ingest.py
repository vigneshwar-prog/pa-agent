"""src/ingest.py — document loading, chunking, metadata tagging, and Pinecone upsert."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

from src.logger import get_logger
from src.vectorstore import get_vectorstore

load_dotenv()
logger = get_logger(__name__)

# ── Chunking config ────────────────────────────────────────────────────────────
_enc = tiktoken.get_encoding("cl100k_base")

def _token_len(text: str) -> int:
    return len(_enc.encode(text))

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    length_function=_token_len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ── Generic keyword → category mapping ────────────────────────────────────────
# Categories are domain-agnostic — works for personal docs, tech docs,
# learning material, runbooks, or any knowledge base.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    # Personal / bio
    "personal":       ["resume", "cv", "career", "hobby", "interest", "personality",
                       "goal", "aspire", "strength", "weakness", "about me", "profile"],
    # Professional skills & experience
    "experience":     ["worked", "company", "role", "job", "position", "employer",
                       "project", "client", "responsible", "led", "managed", "delivered"],
    "skills":         ["python", "java", "sql", "aws", "skill", "proficient", "tool",
                       "framework", "technology", "stack", "language", "platform"],
    "education":      ["degree", "university", "college", "studied", "graduated",
                       "course", "certification", "gpa", "diploma", "training"],
    # Technical knowledge domains
    "technical":      ["architecture", "design", "system", "api", "database", "cloud",
                       "docker", "kubernetes", "microservice", "infrastructure", "config",
                       "protocol", "network", "ospf", "bgp", "rest", "grpc"],
    "troubleshooting":["error", "issue", "bug", "fix", "resolve", "debug", "traceback",
                       "exception", "fail", "crash", "timeout", "symptom", "root cause",
                       "workaround", "known issue", "incident", "alert", "log"],
    "learning":       ["chapter", "section", "definition", "concept", "explain",
                       "understand", "theory", "principle", "example", "exercise",
                       "summary", "note", "lecture", "lesson", "module", "topic"],
    "reference":      ["specification", "rfc", "standard", "manual", "guide",
                       "documentation", "syntax", "command", "parameter", "flag",
                       "option", "table", "list", "appendix", "glossary"],
    "procedure":      ["step", "how to", "install", "configure", "setup", "deploy",
                       "run", "execute", "procedure", "process", "workflow", "checklist"],
}

INGESTION_LOG_PATH = Path("data/ingestion_log.json")


def _load_ingestion_log() -> dict[str, str]:
    if INGESTION_LOG_PATH.exists():
        with open(INGESTION_LOG_PATH) as f:
            return json.load(f)
    return {}


def _save_ingestion_log(log: dict[str, str]) -> None:
    INGESTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INGESTION_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _tag_category(text: str) -> str:
    """
    Assign a category to a chunk based on keyword presence.
    Returns the FIRST matching category (order in dict = priority).
    Falls back to 'general' if nothing matches.
    """
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return "general"


# ── File-type router ───────────────────────────────────────────────────────────

def _load_txt(path: Path) -> list[Document]:
    from langchain_community.document_loaders import TextLoader
    return TextLoader(str(path), encoding="utf-8").load()


def _load_pdf(path: Path) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader(str(path)).load()


def _load_docx(path: Path) -> list[Document]:
    from langchain_community.document_loaders import Docx2txtLoader
    return Docx2txtLoader(str(path)).load()


def _load_csv(path: Path) -> list[Document]:
    from langchain_community.document_loaders import CSVLoader
    return CSVLoader(str(path)).load()


def _load_json(path: Path) -> list[Document]:
    from langchain_community.document_loaders import JSONLoader
    return JSONLoader(str(path), jq_schema=".", text_content=False).load()


def _load_xlsx(path: Path) -> list[Document]:
    from langchain_community.document_loaders import UnstructuredExcelLoader
    return UnstructuredExcelLoader(str(path)).load()


def _load_image(path: Path) -> list[Document]:
    from PIL import Image
    import pytesseract
    text = pytesseract.image_to_string(Image.open(path))
    return [Document(page_content=text, metadata={"source": str(path)})]


def _load_audio(path: Path) -> list[Document]:
    from faster_whisper import WhisperModel
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(path))
    text = " ".join(seg.text for seg in segments)
    return [Document(page_content=text, metadata={"source": str(path)})]


_ROUTER: dict[str, Any] = {
    ".txt":  _load_txt,
    ".pdf":  _load_pdf,
    ".docx": _load_docx,
    ".csv":  _load_csv,
    ".json": _load_json,
    ".xlsx": _load_xlsx,
    ".xls":  _load_xlsx,
    ".jpg":  _load_image,
    ".jpeg": _load_image,
    ".png":  _load_image,
    ".webp": _load_image,
    ".mp3":  _load_audio,
    ".wav":  _load_audio,
    ".m4a":  _load_audio,
}


def load_file(path: str | Path) -> list[Document]:
    """Load a single file using the appropriate loader."""
    p = Path(path)
    ext = p.suffix.lower()
    loader_fn = _ROUTER.get(ext)
    if loader_fn is None:
        raise ValueError(f"Unsupported file type: {ext}")
    logger.info("Loading %s", p.name)
    docs = loader_fn(p)
    for doc in docs:
        doc.metadata.setdefault("source", str(p))
    return docs


# ── Ingestion pipeline ─────────────────────────────────────────────────────────

def ingest(file_path: str | Path, namespace: str = "default") -> int:
    """
    Load → chunk → tag → deduplicate → upsert to Pinecone.

    Args:
        file_path: Path to the document to ingest.
        namespace:  Pinecone namespace — logical partition for a user or
                    knowledge domain (e.g. "alice", "networking-101").
                    Defaults to "default".

    Returns:
        Number of NEW chunks upserted.
    """
    docs = load_file(file_path)
    chunks = SPLITTER.split_documents(docs)

    ingestion_log = _load_ingestion_log()
    new_chunks: list[Document] = []

    for chunk in chunks:
        # Include namespace in the hash key so the same doc ingested into
        # two different namespaces gets separate entries.
        content_hash = _sha256(f"{namespace}:{chunk.page_content}")
        if content_hash in ingestion_log:
            continue

        chunk.metadata["category"]     = _tag_category(chunk.page_content)
        chunk.metadata["content_hash"] = content_hash
        chunk.metadata["namespace"]    = namespace

        new_chunks.append(chunk)
        ingestion_log[content_hash] = "pending"

    if not new_chunks:
        logger.info("No new chunks — nothing to upsert.")
        return 0

    logger.info("Upserting %d new chunks to Pinecone (namespace=%s)…", len(new_chunks), namespace)
    vs = get_vectorstore(namespace=namespace)
    vs.add_documents(new_chunks)

    for chunk in new_chunks:
        ingestion_log[chunk.metadata["content_hash"]] = "ingested"

    _save_ingestion_log(ingestion_log)
    logger.info("Done. %d chunks upserted.", len(new_chunks))
    return len(new_chunks)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest <file_path> [namespace]")
        sys.exit(1)
    ns = sys.argv[2] if len(sys.argv) > 2 else "default"
    count = ingest(sys.argv[1], namespace=ns)
    print(f"✅ Upserted {count} new chunks into namespace '{ns}'.")
