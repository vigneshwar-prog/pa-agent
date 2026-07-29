"""src/chain.py — LangChain LCEL RAG chain with conversational memory."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama

from src.logger import get_logger
from src.vectorstore import get_vectorstore

load_dotenv()
logger = get_logger(__name__)

# ── LLM ───────────────────────────────────────────────────────────────────────
_llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0.3,
)

# ── Prompt ────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about Vigneshwar \
based strictly on the provided context. If the context does not contain enough information \
to answer, say "I don't have enough information about that in my knowledge base."

Do not make up facts. Be concise and direct.

Context:
{context}"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ── Retriever ─────────────────────────────────────────────────────────────────
def _build_retriever():
    vs = get_vectorstore()
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.7,
        },
    )

# ── Chain assembly ─────────────────────────────────────────────────────────────
def build_chain() -> RunnableWithMessageHistory:
    """Build and return the conversational RAG chain."""
    retriever = _build_retriever()
    combine_docs_chain = create_stuff_documents_chain(_llm, _prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # In-memory session history (swap to Redis for production)
    _store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in _store:
            _store[session_id] = ChatMessageHistory()
        return _store[session_id]

    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    logger.info("RAG chain ready.")
    return conversational_rag


# Singleton chain instance
_chain: RunnableWithMessageHistory | None = None


def get_chain() -> RunnableWithMessageHistory:
    global _chain
    if _chain is None:
        _chain = build_chain()
    return _chain
