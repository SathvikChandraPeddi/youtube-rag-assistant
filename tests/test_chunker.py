"""
Verification test script for Phase 3: Chunking.
Tests document splitting, metadata enrichment, timestamp URL generation,
and runs a comparison across multiple chunk sizes.
"""

import json
import os
import sys

# Ensure the project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.transcript import build_transcript_documents
from app.ingestion.chunker import create_chunks, summarize_chunks


def run_test():
    url = "https://www.youtube.com/watch?v=zjkBMFhNj_g"
    print(f"=== Starting Chunking Pipeline Test ===")
    print(f"Target URL: {url}\n")

    try:
        # Step 4: Run full pipeline
        print("--- Step 1: Building transcript documents ---")
        documents = build_transcript_documents(url)
        print(f"Documents received: {len(documents)}\n")

        print("--- Step 2: Creating chunks (chunk_size=1000, overlap=150) ---")
        chunks = create_chunks(documents, chunk_size=1000, chunk_overlap=150)
        print()

        # Step 5: Prints
        print("=== Chunk Set Summary ===")
        summary = summarize_chunks(chunks)
        print(json.dumps(summary, indent=2))
        print()

        if chunks:
            print("=== Chunk #0 (First Chunk) Metadata ===")
            print(json.dumps(chunks[0].metadata, indent=2))
            print()

            print("=== Chunk #5 (Sixth Chunk) Metadata ===")
            if len(chunks) > 5:
                print(json.dumps(chunks[5].metadata, indent=2))
            else:
                print(
                    f"Note: Total chunks is {len(chunks)} (video is short ~38s). "
                    f"Displaying metadata for available chunk #{len(chunks) - 1}:"
                )
                print(json.dumps(chunks[-1].metadata, indent=2))
            print()

            print("=== Chunk #0 Page Content (Full Text) ===")
            print(chunks[0].page_content)
            print()

            print("=== Chunk #0 Timestamp Deep-Link URL ===")
            print(chunks[0].metadata.get("timestamp_url"))
            print()

        # Step 6: Comparison test across chunk sizes
        print("=== Running Chunk Size Comparison Test ===")
        print("Generating chunks with sizes 500, 1000, 1500...")
        chunks_500 = create_chunks(documents, chunk_size=500, chunk_overlap=75)
        chunks_1000 = create_chunks(documents, chunk_size=1000, chunk_overlap=150)
        chunks_1500 = create_chunks(documents, chunk_size=1500, chunk_overlap=200)

        sum_500 = summarize_chunks(chunks_500)
        sum_1000 = summarize_chunks(chunks_1000)
        sum_1500 = summarize_chunks(chunks_1500)

        print("\n=== Chunk Size Comparison Table ===")
        print(f"{'chunk_size':<12} | {'total_chunks':<14} | {'avg_length_chars':<16}")
        print("-" * 48)
        print(
            f"{500:<12} | {sum_500['total_chunks']:<14} | {sum_500['avg_length_chars']:<16}"
        )
        print(
            f"{1000:<12} | {sum_1000['total_chunks']:<14} | {sum_1000['avg_length_chars']:<16}"
        )
        print(
            f"{1500:<12} | {sum_1500['total_chunks']:<14} | {sum_1500['avg_length_chars']:<16}"
        )
        print("-" * 48)

        print("\nPipeline test completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] Pipeline test failed: {str(e)}")


if __name__ == "__main__":
    run_test()
