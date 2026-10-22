"""
Evaluation module.
Evaluates retrieval and generation quality using keyword overlap and RAGAS metrics.
Supports comparing different chunking and retrieval configurations.
Implemented in Phase 10.
"""

import csv
import os
from typing import Any, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_eval_dataset() -> list[dict]:
    """
    Load and return the evaluation dataset from eval_dataset.py.
    """
    from evaluation.eval_dataset import EVAL_DATASET

    dataset = EVAL_DATASET
    logger.info("Loaded %d evaluation test cases.", len(dataset))
    return dataset


def run_pipeline_on_question(
    question: str,
    retriever,
    memory,
    llm,
    use_reranking: bool = False,
    retriever_config=None,
) -> dict:
    """
    Run the full RAG pipeline on a single question and return
    all components needed for evaluation.
    """
    from app.generation.generator import generate_answer_with_sources

    try:
        if use_reranking and retriever_config is not None:
            result = generate_answer_with_sources(
                question, retriever_config, memory, llm, use_reranking=True
            )
        else:
            result = generate_answer_with_sources(
                question, retriever, memory, llm, use_reranking=False
            )

        return {
            "question": question,
            "answer": result["answer"],
            "contexts": [s["text"] for s in result.get("sources", [])],
            "source_timestamps": [
                s.get("start_timestamp_formatted", "")
                for s in result.get("sources", [])
            ],
            "not_found": result.get("not_found", False),
            "num_sources": result.get("num_sources", 0),
        }
    except Exception as e:
        return {
            "question": question,
            "answer": "ERROR: " + str(e),
            "contexts": [],
            "source_timestamps": [],
            "not_found": True,
            "num_sources": 0,
        }


def evaluate_retrieval_relevance(
    question: str,
    contexts: list[str],
    relevant_keywords: list[str],
) -> float:
    """
    Simple keyword-based retrieval relevance score.
    Checks what fraction of the relevant_keywords appear in
    the retrieved contexts combined.

    This is a lightweight alternative to running RAGAS for
    every evaluation — useful for quick checks.
    """
    if not relevant_keywords:
        return 1.0

    if not contexts:
        return 0.0

    combined_text = " ".join(contexts).lower()
    keywords_found = 0

    for kw in relevant_keywords:
        if kw.lower() in combined_text:
            keywords_found += 1

    return float(keywords_found / len(relevant_keywords))


def evaluate_with_ragas(
    eval_results: list[dict],
    llm,
) -> dict | list[dict]:
    """
    Run RAGAS evaluation on the pipeline outputs.
    RAGAS uses an LLM-as-judge approach to score:
    - faithfulness: is the answer supported by the context?
    - answer_relevancy: does the answer address the question?
    - context_precision: are the retrieved contexts relevant?
    - context_recall: do the contexts cover what is needed?

    Why RAGAS evaluation can be slow:
    - It makes one LLM call per metric per question.
    - For 8 questions and 2 metrics that is 16 LLM calls.
    - This is normal — evaluation is not meant to be real-time.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, faithfulness

        ragas_data: dict[str, list[Any]] = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for result in eval_results:
            if result.get("answer", "").startswith("ERROR:"):
                continue

            ragas_data["question"].append(result["question"])
            ragas_data["answer"].append(result["answer"])
            ragas_data["contexts"].append(result.get("contexts", []))
            ragas_data["ground_truth"].append("")

        if not ragas_data["question"]:
            return {"error": "No valid results to evaluate", "message": "All results were errors"}

        dataset = Dataset.from_dict(ragas_data)
        ragas_llm = LangchainLLMWrapper(llm)

        from ragas.embeddings import LangchainEmbeddingsWrapper
        from app.utils.helpers import get_embedding_model

        ragas_embeddings = LangchainEmbeddingsWrapper(get_embedding_model())

        score = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        return score.to_pandas().to_dict(orient="records")

    except Exception as e:
        logger.error("RAGAS evaluation failed: %s", e)
        return {"error": str(e), "message": "RAGAS evaluation failed"}


def run_configuration_comparison(
    video_url: str,
    eval_dataset: list[dict],
    llm,
    configs: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Run the full evaluation pipeline for multiple configurations
    and return a comparison table.
    """
    from app.generation.generator import create_conversation_memory
    from app.retrieval.retriever import (
        build_retriever,
        build_retriever_with_reranking,
    )
    from app.retrieval.vectorstore import index_video
    from app.utils.helpers import get_reranker_model

    if configs is None:
        configs = [
            {
                "name": "chunk_500_no_rerank",
                "chunk_size": 500,
                "chunk_overlap": 75,
                "use_reranking": False,
                "k": 5,
            },
            {
                "name": "chunk_1000_no_rerank",
                "chunk_size": 1000,
                "chunk_overlap": 150,
                "use_reranking": False,
                "k": 5,
            },
            {
                "name": "chunk_1000_with_rerank",
                "chunk_size": 1000,
                "chunk_overlap": 150,
                "use_reranking": True,
                "k": 5,
            },
        ]

    comparison_results = []

    for config in configs:
        print(f"\n{'=' * 50}")
        print(f"Testing config: {config['name']}")
        print(f"{'=' * 50}")

        save_path = f"vectorstore/eval_{config['name']}"
        vectorstore, _ = index_video(
            video_url,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            save_path=save_path,
        )

        retriever = build_retriever(vectorstore, k=config["k"])

        retriever_config = None
        if config["use_reranking"]:
            reranker = get_reranker_model()
            retriever_config = build_retriever_with_reranking(
                vectorstore,
                reranker,
                initial_k=config["k"] * 4,
                final_k=config["k"],
            )

        results = []
        keyword_scores = []

        for eval_entry in eval_dataset:
            memory = create_conversation_memory()
            result = run_pipeline_on_question(
                eval_entry["question"],
                retriever,
                memory,
                llm,
                use_reranking=config["use_reranking"],
                retriever_config=retriever_config,
            )

            score = evaluate_retrieval_relevance(
                result["question"],
                result["contexts"],
                eval_entry.get("relevant_keywords", []),
            )
            result["keyword_score"] = score
            result["expected_answer"] = eval_entry.get("expected_answer", "")

            results.append(result)
            keyword_scores.append(score)

        avg_score = (
            sum(keyword_scores) / len(keyword_scores)
            if keyword_scores
            else 0.0
        )
        not_found_count = sum(1 for r in results if r["not_found"])
        avg_sources = (
            sum(r["num_sources"] for r in results) / len(results)
            if results
            else 0.0
        )

        comparison_results.append(
            {
                "config_name": config["name"],
                "chunk_size": config["chunk_size"],
                "use_reranking": config["use_reranking"],
                "avg_keyword_score": round(avg_score, 2),
                "not_found_count": not_found_count,
                "avg_num_sources": round(avg_sources, 2),
                "results": results,
            }
        )

    return comparison_results


def save_results_to_csv(
    comparison_results: list[dict],
    output_path: str = "evaluation/eval_results.csv",
) -> str:
    """
    Save the comparison results to a CSV file for analysis.
    """
    folder = os.path.dirname(output_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    fieldnames = [
        "config_name",
        "chunk_size",
        "use_reranking",
        "question",
        "answer_preview",
        "num_sources",
        "keyword_score",
        "not_found",
    ]

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for config in comparison_results:
            config_name = config.get("config_name", "")
            chunk_size = config.get("chunk_size", "")
            use_reranking = config.get("use_reranking", False)

            for result in config.get("results", []):
                answer_full = result.get("answer", "")
                answer_preview = (
                    answer_full[:150].replace("\n", " ") + "..."
                    if len(answer_full) > 150
                    else answer_full.replace("\n", " ")
                )

                writer.writerow(
                    {
                        "config_name": config_name,
                        "chunk_size": chunk_size,
                        "use_reranking": use_reranking,
                        "question": result.get("question", ""),
                        "answer_preview": answer_preview,
                        "num_sources": result.get("num_sources", 0),
                        "keyword_score": result.get("keyword_score", 0.0),
                        "not_found": result.get("not_found", False),
                    }
                )

    print(f"Results saved to {output_path}")
    return output_path


def print_comparison_table(comparison_results: list[dict]):
    """
    Print a clean comparison table to the terminal.
    """
    print("\n=== RAG EVALUATION COMPARISON ===\n")
    print(
        f"{'Config':<26}| {'Chunks':<6} | {'Rerank':<6} | {'Keyword Score':<13} | {'Not Found':<9}"
    )
    print("-" * 26 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 15 + "|" + "-" * 10)

    best_config_name = ""
    highest_score = -1.0

    total_questions = 0

    for config in comparison_results:
        cfg_name = config.get("config_name", "")
        chunks = str(config.get("chunk_size", ""))
        rerank = "Yes" if config.get("use_reranking") else "No"
        score = config.get("avg_keyword_score", 0.0)
        not_found_cnt = config.get("not_found_count", 0)
        num_res = len(config.get("results", []))
        total_questions = num_res if num_res > 0 else 8

        if score > highest_score:
            highest_score = score
            best_config_name = cfg_name

        print(
            f"{cfg_name:<26}| {chunks:^6} | {rerank:^6} | {score:^13.2f} | {f'{not_found_cnt}/{total_questions}':^9}"
        )

    print()
    if best_config_name:
        print(f"Best config by keyword score: {best_config_name}\n")
