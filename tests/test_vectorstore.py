"""
Verification test script for Phase 4: Vector Store and Embeddings.
Tests indexing pipeline, FAISS vector similarity search, and index persistence/loading.
"""

import json
import os
import sys

# Ensure root folder is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrieval.vectorstore import index_video, load_vectorstore, vectorstore_exists


def run_tests():
    url = "https://www.youtube.com/watch?v=zjkBMFhNj_g"
    save_path = "vectorstore/index"
    vectorstore = None
    test2_results_text = []

    print("==================================================")
    print("      PHASE 4: VECTOR STORE VERIFICATION TESTS    ")
    print("==================================================\n")

    # TEST 1 — Full indexing pipeline
    print("--- TEST 1: Indexing pipeline ---")
    try:
        vectorstore, metadata = index_video(url, save_path=save_path)
        print("\nIndexing Metadata:")
        print(json.dumps(metadata, indent=2))
        print(f"Vector store type: {type(vectorstore)}")
        print(">>> TEST 1 PASSED!\n")
    except Exception as e:
        print(f"[ERROR in TEST 1]: {str(e)}\n")
        return

    # TEST 2 — Similarity search
    print("--- TEST 2: Similarity search ---")
    try:
        query = "what is the main topic of this video"
        print(f"Query: \"{query}\"")
        results = vectorstore.similarity_search(query, k=3)
        print(f"Found {len(results)} matching chunks:\n")

        for idx, doc in enumerate(results):
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            ts_formatted = doc.metadata.get("start_timestamp_formatted", "0:00")
            ts_url = doc.metadata.get("timestamp_url", "N/A")
            content_preview = doc.page_content[:150].strip()

            test2_results_text.append(doc.page_content)

            print(f"Result #{idx + 1}:")
            print(f"  Chunk ID: {chunk_id}")
            print(f"  Timestamp: {ts_formatted}")
            print(f"  Timestamp URL: {ts_url}")
            print(f"  Content snippet: \"{content_preview}...\"\n")

        print(">>> TEST 2 PASSED!\n")
    except Exception as e:
        print(f"[ERROR in TEST 2]: {str(e)}\n")
        return

    # TEST 3 — Save and reload
    print("--- TEST 3: Save and reload ---")
    try:
        exists = vectorstore_exists(save_path)
        print(f"Checking vectorstore_exists('{save_path}'): {exists}")
        if not exists:
            raise FileNotFoundError(f"vectorstore_exists returned False for {save_path}")

        print(f"Loading vectorstore from '{save_path}'...")
        reloaded_vectorstore = load_vectorstore(save_path)
        print(f"Reloaded vector store type: {type(reloaded_vectorstore)}")

        query = "what is the main topic of this video"
        reloaded_results = reloaded_vectorstore.similarity_search(query, k=3)
        reloaded_results_text = [d.page_content for d in reloaded_results]

        # Confirm results match TEST 2
        matches = test2_results_text == reloaded_results_text
        print(f"Similarity search results match TEST 2 exactly: {matches}")
        if not matches:
            raise ValueError("Reloaded search results did not match in-memory search results!")

        print(">>> TEST 3 PASSED!\n")
    except Exception as e:
        print(f"[ERROR in TEST 3]: {str(e)}\n")
        return

    print("==================================================")
    print("   ALL PHASE 4 TESTS COMPLETED SUCCESSFULLY!      ")
    print("==================================================")


if __name__ == "__main__":
    run_tests()
