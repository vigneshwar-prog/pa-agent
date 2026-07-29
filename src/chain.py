"""src/chain.py — LangChain LCEL RAG chain with conversational memory."""
from __future__ import annotations

import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

from src.logger import get_logger
from src.vectorstore import get_vectorstore

load_dotenv()
logger = get_logger(__name__)

# ── LLM — ClaudeGate (OpenAI-compatible gateway at localhost:8080) ─────────────
# Uses claude-sonnet-4.5 by default. Swap model name via CLAUDEGATE_MODEL env var.
_llm = ChatOpenAI(
    model=os.getenv("CLAUDEGATE_MODEL", "claude-sonnet-4.5"),
    base_url=os.getenv("CLAUDEGATE_BASE_URL", "http://localhost:8080/v1"),
    api_key="dummy",          # gateway handles auth — key value doesn't matter
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


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


# ── Chain assembly (pure LCEL) ─────────────────────────────────────────────────
def build_chain() -> RunnableWithMessageHistory:
    """Build and return the conversational RAG chain using pure LCEL."""
    retriever = _build_retriever()

    # Retrieve docs, keep them accessible for sources AND format for prompt
    rag_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("input") | retriever | _format_docs,
            # stash raw docs so we can return sources
            source_docs=itemgetter("input") | retriever,
        )
        | RunnablePassthrough.assign(
            answer=_prompt | _llm | StrOutputParser()
        )
    )

    # In-memory session history — swap to Redis for production
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
