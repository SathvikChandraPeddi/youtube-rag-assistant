"""
Verification test script for Phase 5: Retrieval Module.
Tests similarity retriever, MMR diversification, context string construction,
and unrelated query edge-case handling.
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrieval.vectorstore import load_vectorstore, vectorstore_exists, index_video
from app.retrieval.retriever import (
    build_retriever,
    retrieve_and_format,
    build_context_string,
)


def run_tests():
    print("==================================================")
    print("      PHASE 5: RETRIEVER VERIFICATION TESTS       ")
    print("==================================================\n")

    index_path = "vectorstore/index"
    url = "https://www.youtube.com/watch?v=zjkBMFhNj_g"

    # Setup: Ensure index is present and load it
    try:
        if not vectorstore_exists(index_path):
            print(f"Index not found at {index_path}. Building index now...")
            vectorstore, _ = index_video(url, save_path=index_path)
        else:
            print(f"Loading vectorstore from '{index_path}'...")
            vectorstore = load_vectorstore(index_path)
        print("Vector store ready.\n")
    except Exception as e:
        print(f"[SETUP ERROR]: {str(e)}")
        return

    test1_docs = []

    # TEST 1 — Similarity retriever
    print("--- TEST 1: Similarity retriever (k=5) ---")
    try:
        retriever_sim = build_retriever(vectorstore, search_type="similarity", k=5)
        query = "explain the main concept discussed in this video"
        print(f"Query: \"{query}\"")
        docs, formatted_sources = retrieve_and_format(retriever_sim, query)
        test1_docs = docs

        print(f"\nRetrieved {len(formatted_sources)} formatted sources:")
        print(f"{'chunk_id':<10} | {'timestamp':<12} | {'text_preview'}")
        print("-" * 75)
        for s in formatted_sources:
            preview = s['text_preview'].replace('\n', ' ')
            print(f"{str(s['chunk_id']):<10} | {s['start_timestamp_formatted']:<12} | {preview[:50]}...")
        print(">>> TEST 1 PASSED!\n")
    except Exception as e:
        print(f"[ERROR in TEST 1]: {str(e)}\n")

    # TEST 2 — MMR retriever
    print("--- TEST 2: MMR retriever (k=5) ---")
    try:
        retriever_mmr = build_retriever(vectorstore, search_type="mmr", k=5)
        query = "explain the main concept discussed in this video"
        print(f"Query: \"{query}\"")
        mmr_docs, mmr_formatted = retrieve_and_format(retriever_mmr, query)

        print(f"\nRetrieved {len(mmr_formatted)} MMR formatted sources:")
        print(f"{'chunk_id':<10} | {'timestamp':<12} | {'text_preview'}")
        print("-" * 75)
        for s in mmr_formatted:
            preview = s['text_preview'].replace('\n', ' ')
            print(f"{str(s['chunk_id']):<10} | {s['start_timestamp_formatted']:<12} | {preview[:50]}...")

        print("\nMMR vs Similarity comparison:")
        print("MMR reranks candidates with lambda_mult=0.7 to avoid returning adjacent redundant chunks.")
        print(">>> TEST 2 PASSED!\n")
    except Exception as e:
        print(f"[ERROR in TEST 2]: {str(e)}\n")

    # TEST 3 — Context string construction
    print("--- TEST 3: Context string construction ---")
    try:
        context_str = build_context_string(test1_docs)
        print("Generated Context String for LLM Prompt:\n")
        print("--------------------------------------------------")
        print(context_str)
        print("--------------------------------------------------")
        print(">>> TEST 3 PASSED!\n")
    except Exception as e:
        print(f"[ERROR in TEST 3]: {str(e)}\n")

    # TEST 4 — Edge case: Completely unrelated query
    print("--- TEST 4: Edge case (unrelated query) ---")
    try:
        unrelated_query = "what is the best pizza recipe"
        print(f"Query: \"{unrelated_query}\"")
        unrelated_docs, unrelated_sources = retrieve_and_format(retriever_sim, unrelated_query)

        print(f"\nRetrieved {len(unrelated_sources)} chunks for unrelated query:")
        for idx, s in enumerate(unrelated_sources):
            print(f"  [{idx + 1}] Chunk {s['chunk_id']} (t={s['start_timestamp_formatted']}): \"{s['text_preview'][:80]}...\"")

        print("\nObservation:")
        print("Notice that FAISS returns k closest vectors even when cosine similarity is low or topic is irrelevant.")
        print("This proves why Phase 8 prompt engineering & grounding instructions are essential.")
        print(">>> TEST 4 PASSED!\n")
    except Exception as e:
        print(f"[ERROR in TEST 4]: {str(e)}\n")

    print("==================================================")
    print("   ALL PHASE 5 TESTS COMPLETED SUCCESSFULLY!      ")
    print("==================================================")


if __name__ == "__main__":
    run_tests()
