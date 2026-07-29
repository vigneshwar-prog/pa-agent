"""src/main.py — CLI chat loop (Phase 1)."""
from __future__ import annotations

import uuid

from dotenv import load_dotenv

from src.chain import get_chain
from src.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def main() -> None:
    print("🧠 Vigneshwar's Second Brain — CLI")
    print("Type 'exit' or 'quit' to stop.\n")

    chain = get_chain()
    session_id = str(uuid.uuid4())
    logger.info("Session started: %s", session_id)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"\nAssistant: {result['answer']}\n")

        sources = {d.metadata.get("source", "unknown") for d in result.get("source_docs", [])}
        if sources:
            print(f"  Sources: {', '.join(sources)}\n")


if __name__ == "__main__":
    main()
