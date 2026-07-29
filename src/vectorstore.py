"""src/vectorstore.py — Pinecone vector store factory."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── Embeddings ─────────────────────────────────────────────────────────────────
# bge-large-en-v1.5 — 1024-dim, top MTEB score among open models
# normalize_embeddings=True is REQUIRED for correct cosine similarity with BGE
_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", _EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name=_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ── Pinecone setup ─────────────────────────────────────────────────────────────
_PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY", "")
_PINECONE_INDEX    = os.getenv("PINECONE_INDEX", "pa-second-brain")
_DEFAULT_NS        = os.getenv("PINECONE_NAMESPACE", "default")

# Dimension must match the embedding model
_EMBEDDING_DIM: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

# Cache one PineconeVectorStore per namespace
_vectorstore_cache: dict[str, PineconeVectorStore] = {}


def _ensure_pinecone_index() -> None:
    """Create the Pinecone index if it doesn't already exist."""
    pc = Pinecone(api_key=_PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    if _PINECONE_INDEX not in existing:
        dim = _EMBEDDING_DIM.get(_EMBEDDING_MODEL, 1024)
        logger.info("Creating Pinecone index '%s' (dim=%d)…", _PINECONE_INDEX, dim)
        pc.create_index(
            name=_PINECONE_INDEX,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info("Index created.")
    else:
        logger.debug("Pinecone index '%s' already exists.", _PINECONE_INDEX)


def get_vectorstore(namespace: str | None = None) -> PineconeVectorStore:
    """Return a PineconeVectorStore for the given namespace (cached per namespace)."""
    global _vectorstore_cache
    ns = namespace or _DEFAULT_NS
    if ns not in _vectorstore_cache:
        if not _PINECONE_API_KEY:
            raise EnvironmentError(
                "PINECONE_API_KEY is not set. Copy .env.example to .env and fill in your key."
            )
        _ensure_pinecone_index()
        embeddings = get_embeddings()
        logger.info(
            "Connecting to Pinecone index '%s' namespace='%s'",
            _PINECONE_INDEX,
            ns,
        )
        _vectorstore_cache[ns] = PineconeVectorStore(
            index_name=_PINECONE_INDEX,
            embedding=embeddings,
            namespace=ns,
        )
    return _vectorstore_cache[ns]


# ── Filtered search (category metadata) ───────────────────────────────────────

def filtered_search(query: str, category: str, k: int = 6, namespace: str | None = None) -> list:
    """
    Semantic search filtered by metadata category.
    Pinecone supports native metadata filtering — no post-hoc needed.
    """
    ns = namespace or _DEFAULT_NS
    vs = get_vectorstore(namespace=ns)
    return vs.similarity_search(
        query,
        k=k,
        filter={"category": {"$eq": category}},
        namespace=ns,
    )
