"""src/api.py — FastAPI REST API with Prometheus metrics."""
from __future__ import annotations

import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from src.chain import get_chain
from src.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

app = FastAPI(title="Personal Assistant RAG API", version="1.0.0")

# Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)


# ── Request / Response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    session_id: str = ""
    namespace: str = ""   # Pinecone namespace — isolates one user/domain from another


class AskResponse(BaseModel):
    answer: str
    session_id: str
    namespace: str
    sources: list[str]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    session_id = req.session_id or str(uuid.uuid4())
    namespace = req.namespace or None
    logger.info("Session=%s | namespace=%s | question=%r", session_id, namespace, req.question[:80])

    try:
        chain = get_chain(namespace=namespace)
        result = chain.invoke(
            {"input": req.question},
            config={"configurable": {"session_id": session_id}},
        )
    except Exception as exc:
        logger.exception("Chain invocation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = list(
        {d.metadata.get("source", "unknown") for d in result.get("source_docs", [])}
    )
    return AskResponse(
        answer=result["answer"],
        session_id=session_id,
        namespace=req.namespace,
        sources=sources,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}
