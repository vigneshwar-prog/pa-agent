"""src/api.py — FastAPI REST API with Prometheus metrics."""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from src.chain import get_chain
from src.ingest import ingest
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


class IngestResponse(BaseModel):
    filename: str
    namespace: str
    chunks_upserted: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    namespace: str = Form(default="default"),
) -> IngestResponse:
    """
    Upload any supported file and ingest it into Pinecone.

    Supported: .pdf .txt .docx .csv .json .xlsx .xls
               .jpg .jpeg .png .webp .mp3 .wav .m4a .vtt .srt

    The file is saved to a temp directory, ingested, then deleted.
    Vectors are stored persistently in Pinecone — re-uploading the same
    file is safe (SHA-256 dedup skips already-ingested chunks).
    """
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        logger.info("Ingesting '%s' into namespace='%s'", file.filename, namespace)
        count = ingest(tmp_path, namespace=namespace)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingest failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)   # always clean up temp file

    return IngestResponse(
        filename=file.filename,
        namespace=namespace,
        chunks_upserted=count,
    )


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
