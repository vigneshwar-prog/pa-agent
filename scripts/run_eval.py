"""scripts/run_eval.py — RAGAs evaluation pipeline."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def run_evaluation() -> dict:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from src.chain import get_chain

    golden_path = Path("data/eval/golden_dataset.json")
    if not golden_path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found at {golden_path}. "
            "Run scripts/generate_golden.py first."
        )

    with open(golden_path) as f:
        golden: list[dict] = json.load(f)

    chain = get_chain()
    records = []

    print(f"Running evaluation on {len(golden)} questions…")
    for i, item in enumerate(golden, 1):
        print(f"  [{i}/{len(golden)}] {item['question'][:60]}")
        result = chain.invoke(
            {"input": item["question"]},
            config={"configurable": {"session_id": f"eval_{i}"}},
        )
        records.append(
            {
                "question":     item["question"],
                "answer":       result["answer"],
                "contexts":     [d.page_content for d in result.get("context", [])],
                "ground_truth": item["ground_truth"],
            }
        )

    dataset = Dataset.from_list(records)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    results = scores.to_pandas().to_dict(orient="list")

    # Save results
    out_dir = Path("data/eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
    out_path = out_dir / f"{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n── RAGAs Results ────────────────────────────")
    targets = {
        "faithfulness":      0.85,
        "answer_relevancy":  0.80,
        "context_recall":    0.75,
        "context_precision": 0.70,
    }
    for metric, target in targets.items():
        vals = results.get(metric, [])
        avg = sum(vals) / len(vals) if vals else 0
        status = "✅" if avg >= target else "❌"
        print(f"  {status} {metric:<22} {avg:.3f}  (target ≥ {target})")
    print(f"\nResults saved → {out_path}")

    return results


if __name__ == "__main__":
    run_evaluation()
