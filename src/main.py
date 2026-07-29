"""src/main.py — CLI chat loop (Phase 1)."""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from dotenv import load_dotenv

from src.chain import get_chain
from src.ingest import ingest
from src.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# Supported extensions (keep in sync with ingest._ROUTER)
_SUPPORTED_EXTS = {
    ".txt", ".pdf", ".docx", ".csv", ".json", ".xlsx", ".xls",
    ".jpg", ".jpeg", ".png", ".webp",
    ".mp3", ".wav", ".m4a",
    ".vtt", ".srt",
}

# Regex to pull a file path out of free-form text
# Matches a quoted path OR a bare path that ends with a known extension
_PATH_RE = re.compile(
    r"""['"]([^'"]+\.(?:""" + "|".join(e.lstrip(".") for e in _SUPPORTED_EXTS) + r"""))['"]"""
    r"""|(\S+\.(?:""" + "|".join(e.lstrip(".") for e in _SUPPORTED_EXTS) + r"""))""",
    re.IGNORECASE,
)


def _extract_path(text: str) -> Path | None:
    """
    If the user's message contains a file path (quoted or bare), return it.
    Otherwise return None.
    """
    m = _PATH_RE.search(text)
    if m:
        raw = m.group(1) or m.group(2)
        p = Path(raw.strip())
        if p.exists():
            return p
    return None


def main() -> None:
    print("🧠 Personal Assistant RAG — CLI")
    print("Type 'exit' or 'quit' to stop.")
    print("Tip: paste a file path (or type it) to ingest a document.\n")

    namespace = input("Namespace (press Enter for 'default'): ").strip() or "default"
    print(f"Using namespace: '{namespace}'\n")

    chain = get_chain(namespace=namespace)
    session_id = str(uuid.uuid4())
    logger.info("Session started: %s | namespace: %s", session_id, namespace)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # ── File ingest detection ──────────────────────────────────────────────
        file_path = _extract_path(user_input)
        if file_path:
            print(f"\n📂 Detected file: {file_path.name}")
            print(f"   Ingesting into namespace '{namespace}'…")
            try:
                count = ingest(file_path, namespace=namespace)
                if count > 0:
                    print(f"   ✅ Done — {count} new chunks stored in Pinecone.\n")
                else:
                    print("   ℹ️  Already ingested — no new chunks (dedup skipped all).\n")
            except ValueError as exc:
                print(f"   ❌ Unsupported file type: {exc}\n")
            except Exception as exc:
                print(f"   ❌ Ingest failed: {exc}\n")
            continue   # don't send the file path to the LLM

        # ── Normal RAG question ───────────────────────────────────────────────
        result = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"\nAssistant: {result['answer']}\n")

        sources = {d.metadata.get("source", "unknown") for d in result.get("source_docs", [])}
        if sources:
            print(f"  Sources: {', '.join(sources)}\n")


if __name__ == "__main__":
    main()
