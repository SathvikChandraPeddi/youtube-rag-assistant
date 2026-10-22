"""
RAG Evaluation Script
Run this to evaluate and compare different RAG configurations.
Usage: python evaluation/run_eval.py
"""

import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from app.evaluation.evaluator import (
    load_eval_dataset,
    print_comparison_table,
    run_configuration_comparison,
    save_results_to_csv,
)
from app.utils.helpers import get_groq_llm

VIDEO_URL = os.getenv(
    "EVAL_VIDEO_URL", "https://www.youtube.com/watch?v=dY7nvNtu_ZA"
)


def main():
    print("=" * 60)
    print("YouTube RAG Assistant — Evaluation Suite")
    print("=" * 60)

    # Step 1 — Load LLM
    print("\nLoading Groq LLM...")
    llm = get_groq_llm()
    print("LLM ready.")

    # Step 2 — Load evaluation dataset
    eval_dataset = load_eval_dataset()

    # Step 3 — Run configuration comparison
    print(f"\nRunning evaluation on: {VIDEO_URL}")
    print("This will test 3 configurations × 8 questions.")
    print("Estimated time: 3-5 minutes (re-indexes video 3 times)\n")

    comparison_results = run_configuration_comparison(
        VIDEO_URL, eval_dataset, llm
    )

    # Step 4 — Print comparison table
    print_comparison_table(comparison_results)

    # Step 5 — Save results
    output_path = save_results_to_csv(comparison_results)
    print(f"\nDetailed results saved to: {output_path}")

    # Step 6 — Optional RAGAS evaluation on best config
    print("\nRunning RAGAS deep evaluation on best config...")
    print("(This makes additional LLM calls and may take 2-3 minutes)")

    from app.evaluation.evaluator import evaluate_with_ragas

    best_config = max(
        comparison_results, key=lambda x: x["avg_keyword_score"]
    )
    print(f"Best config: {best_config['config_name']}")

    ragas_scores = evaluate_with_ragas(best_config["results"], llm)

    if isinstance(ragas_scores, list):
        print("\n=== RAGAS Scores (Best Config) ===")
        for row in ragas_scores:
            print(f"Q: {row.get('question', '')[:60]}...")
            print(f"   Faithfulness: {row.get('faithfulness', 'N/A')}")
            print(f"   Answer Relevancy: {row.get('answer_relevancy', 'N/A')}")
    elif isinstance(ragas_scores, dict) and "error" not in ragas_scores:
        print("\n=== RAGAS Scores (Best Config) ===")
        print(ragas_scores)
    else:
        err_msg = (
            ragas_scores.get("error", "Unknown error")
            if isinstance(ragas_scores, dict)
            else str(ragas_scores)
        )
        print(f"RAGAS evaluation failed: {err_msg}")

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
