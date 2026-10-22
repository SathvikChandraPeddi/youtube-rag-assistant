"""
Phase 7 Tests: Cross-Encoder Reranking Pipeline.
Verifies the two-stage retrieval pipeline (FAISS → cross-encoder rerank),
compares results against plain similarity search, and validates that
reranker scores are properly injected into document metadata.
"""

import time
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.helpers import get_embedding_model, get_reranker_model
from app.retrieval.vectorstore import index_video, load_vectorstore
from app.retrieval.retriever import (
    build_retriever,
    retrieve_and_format,
    rerank_chunks,
    build_retriever_with_reranking,
    retrieve_and_format_with_reranking,
    compare_retrieval_methods,
)


# =====================================================================
# CONFIG
# =====================================================================
TEST_URL = "https://www.youtube.com/watch?v=cdiD-9MMpb0"
TEST_QUERY = "What is attention mechanism in transformers?"


def run_reranker_test():
    """Full Phase 7 test suite."""

    print("=" * 70)
    print("PHASE 7 TEST: CROSS-ENCODER RERANKING PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load or build the vector store
    # ------------------------------------------------------------------
    print("\n[1/6] Loading vector store...")
    try:
        vectorstore, _ = load_vectorstore()
        print("Loaded existing vector store from disk.")
    except Exception:
        print("No existing store found — indexing video...")
        vectorstore, _ = index_video(TEST_URL)

    # ------------------------------------------------------------------
    # Step 2: Load the reranker model
    # ------------------------------------------------------------------
    print("\n[2/6] Loading reranker model...")
    reranker = get_reranker_model()
    print(f"Reranker loaded: {type(reranker).__name__}")

    # ------------------------------------------------------------------
    # Step 3: Test rerank_chunks directly
    # ------------------------------------------------------------------
    print("\n[3/6] Testing rerank_chunks() directly...")

    # Get some documents first
    retriever = build_retriever(vectorstore, "similarity", k=15)
    from app.retrieval.retriever import retrieve_chunks
    raw_docs = retrieve_chunks(retriever, TEST_QUERY)
    print(f"Got {len(raw_docs)} raw candidate chunks")

    reranked = rerank_chunks(TEST_QUERY, raw_docs, reranker, top_n=4)
    print(f"After reranking: {len(reranked)} chunks selected")

    # Verify reranker scores are injected
    for doc in reranked:
        assert "reranker_score" in doc.metadata, "Missing reranker_score in metadata!"
        score = doc.metadata["reranker_score"]
        print(f"  Chunk {doc.metadata.get('chunk_id', '?')}: score={score:.4f}")

    # Verify descending order
    scores = [doc.metadata["reranker_score"] for doc in reranked]
    assert scores == sorted(scores, reverse=True), "Reranked docs not in descending score order!"
    print("✓ rerank_chunks() works correctly — scores injected and sorted")

    # ------------------------------------------------------------------
    # Step 4: Test build_retriever_with_reranking
    # ------------------------------------------------------------------
    print("\n[4/6] Testing build_retriever_with_reranking()...")
    config = build_retriever_with_reranking(
        vectorstore, reranker, initial_k=20, final_k=4
    )
    assert config["mode"] == "reranked"
    assert config["initial_k"] == 20
    assert config["final_k"] == 4
    assert config["reranker"] is reranker
    print("✓ Retriever config built correctly")

    # ------------------------------------------------------------------
    # Step 5: Test retrieve_and_format_with_reranking (end-to-end)
    # ------------------------------------------------------------------
    print("\n[5/6] Testing retrieve_and_format_with_reranking() end-to-end...")
    docs, sources = retrieve_and_format_with_reranking(TEST_QUERY, config)

    assert len(docs) <= 4, f"Expected <=4 docs, got {len(docs)}"
    assert len(sources) == len(docs)

    print(f"Returned {len(docs)} reranked documents with {len(sources)} formatted sources")
    for s in sources:
        print(f"  [{s['start_timestamp_formatted']}] {s['text_preview'][:80]}...")
    print("✓ Two-stage pipeline works end-to-end")

    # ------------------------------------------------------------------
    # Step 6: Speed comparison — similarity vs reranked
    # ------------------------------------------------------------------
    print("\n[6/6] Speed comparison: similarity vs reranked...")

    # Method A — plain similarity
    t0 = time.time()
    retriever_a = build_retriever(vectorstore, "similarity", k=5)
    docs_a, _ = retrieve_and_format(retriever_a, TEST_QUERY)
    time_similarity = time.time() - t0

    # Method B — two-stage reranked
    t0 = time.time()
    config_b = build_retriever_with_reranking(
        vectorstore, reranker, initial_k=20, final_k=5
    )
    docs_b, _ = retrieve_and_format_with_reranking(TEST_QUERY, config_b)
    time_reranked = time.time() - t0

    print(f"\n  Similarity: {len(docs_a)} docs in {time_similarity:.3f}s")
    print(f"  Reranked:   {len(docs_b)} docs in {time_reranked:.3f}s")
    print(f"  Overhead:   {time_reranked - time_similarity:.3f}s")

    # Show side-by-side chunk IDs
    ids_a = [d.metadata.get("chunk_id", "?") for d in docs_a]
    ids_b = [d.metadata.get("chunk_id", "?") for d in docs_b]
    print(f"\n  Similarity chunk IDs: {ids_a}")
    print(f"  Reranked chunk IDs:   {ids_b}")

    overlap = set(ids_a) & set(ids_b)
    print(f"  Overlap: {len(overlap)}/{len(ids_a)} chunks in common")

    # ------------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ALL PHASE 7 TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_reranker_test()
