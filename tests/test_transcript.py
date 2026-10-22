"""
Verification test script for Phase 2: Transcript Ingestion.
Extracts transcripts and metadata for a 3Blue1Brown video and prints summary metrics.
"""

import json
import os
import sys

# Ensure the root directory of the project is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.transcript import build_transcript_documents


def run_test():
    test_url = "https://www.youtube.com/watch?v=zjkBMFhNj_g"
    print(f"Starting transcript ingestion test with URL: {test_url}\n")

    try:
        # Execute the ingestion pipeline
        documents = build_transcript_documents(test_url)

        # 1. Total number of documents returned
        total_docs = len(documents)
        print(f"\n==========================================")
        print(f"Total documents returned: {total_docs}")
        print(f"==========================================\n")

        if total_docs == 0:
            print("No documents were generated.")
            return

        # 2. Metadata of the first document (pretty printed)
        print("Metadata of first document:")
        print(json.dumps(documents[0].metadata, indent=2))
        print("\n------------------------------------------\n")

        # 3. Page content of the first 3 documents
        print("Page content of the first 3 documents:")
        for idx, doc in enumerate(documents[:3]):
            start_time = doc.metadata.get("start", 0)
            print(f"[{idx + 1}] (t={start_time}s): \"{doc.page_content}\"")
        print("\n------------------------------------------\n")

        # 4. Start timestamp of the last document
        last_doc_start = documents[-1].metadata.get("start", 0)
        print(f"Start timestamp of the last document: {last_doc_start} seconds")
        print("\nTest completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {str(e)}")


if __name__ == "__main__":
    run_test()
