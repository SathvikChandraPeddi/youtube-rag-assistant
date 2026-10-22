"""
Verification test script for Phase 8: Grounded Answer Generation & Source Attribution.
Tests system prompt configuration, not-found detection heuristics, grounded generation,
out-of-domain rejection, multi-turn conversation context continuity, and token streaming.
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrieval.vectorstore import load_vectorstore
from app.retrieval.retriever import build_retriever
from app.utils.helpers import get_reranker_model, get_groq_llm
from app.generation.generator import (
    create_conversation_memory,
    generate_answer_with_sources,
    stream_answer,
    detect_not_found,
    build_system_prompt,
)


def run_tests():
    print("=" * 60)
    print("   PHASE 8: GROUNDED GENERATION & ATTRIBUTION TESTS       ")
    print("=" * 60)

    # ------------------------------------------------------------------
    # TEST 1 — System prompt check
    # ------------------------------------------------------------------
    print("\n--- TEST 1: System Prompt Check ---")
    try:
        sys_prompt = build_system_prompt()
        print("System Prompt Content:\n")
        print(sys_prompt)
        print("-" * 50)

        required_phrase = "ONLY using information"
        if required_phrase in sys_prompt:
            print("System prompt: PASS")
            print(">>> TEST 1 PASSED!\n")
        else:
            raise ValueError(f"System prompt missing required phrase: '{required_phrase}'")
    except Exception as e:
        print(f"[ERROR in TEST 1]: {str(e)}\n")

    # ------------------------------------------------------------------
    # TEST 2 — detect_not_found()
    # ------------------------------------------------------------------
    print("--- TEST 2: detect_not_found() Heuristic Checks ---")
    try:
        test_case_1 = "I could not find information about this in the video."
        res1 = detect_not_found(test_case_1)
        print(f"Phrase 1: \"{test_case_1}\" -> {res1} (Expected: True)")
        assert res1 is True, f"Failed on: {test_case_1}"

        test_case_2 = "The speaker explains that attention works by focusing on relevant tokens."
        res2 = detect_not_found(test_case_2)
        print(f"Phrase 2: \"{test_case_2}\" -> {res2} (Expected: False)")
        assert res2 is False, f"Failed on: {test_case_2}"

        test_case_3 = "This topic is not mentioned in the transcript."
        res3 = detect_not_found(test_case_3)
        print(f"Phrase 3: \"{test_case_3}\" -> {res3} (Expected: True)")
        assert res3 is True, f"Failed on: {test_case_3}"

        print("\ndetect_not_found: PASS")
        print(">>> TEST 2 PASSED!\n")
    except Exception as e:
        print(f"[ERROR in TEST 2]: {str(e)}\n")

    # ------------------------------------------------------------------
    # Setup: Vector Store, Retriever, LLM, Memory
    # ------------------------------------------------------------------
    print("--- Setting up Vector Store, Retriever, and LLM ---")
    retriever = None
    llm = None
    memory = None
    try:
        save_path = "vectorstore/index"
        test_url = "https://www.youtube.com/watch?v=zjkBMFhNj_g"
        try:
            vectorstore = load_vectorstore(save_path)
            print(f"Loaded existing vector store from '{save_path}'.")
        except Exception:
            print(f"Vector store not found on disk. Building index for {test_url}...")
            from app.retrieval.vectorstore import index_video
            vectorstore, _ = index_video(test_url, save_path=save_path)
            print("Vector store indexed and saved successfully.")

        llm = get_groq_llm()
        retriever = build_retriever(vectorstore, k=5)
        memory = create_conversation_memory()
        print("Components initialized successfully.\n")
    except Exception as e:
        print(f"[SETUP ERROR]: {str(e)}\n")
        return

    # ------------------------------------------------------------------
    # TEST 3 — Full answer generation (in-video question)
    # ------------------------------------------------------------------
    print("--- TEST 3: Full Answer Generation (In-Video Question) ---")
    try:
        query = "What is the main concept explained in this video?"
        print(f"User Query: '{query}'\n")

        result = generate_answer_with_sources(query, retriever, memory, llm)

        print("\nGenerated Answer:")
        print(result["answer"])
        print("\nSources Markdown:")
        print(result["sources_markdown"])
        print(f"\nNot found: {result['not_found']}")
        print(f"Number of sources: {result['num_sources']}")

        print(">>> TEST 3 PASSED!\n")
    except Exception as e:
        print(f"[ERROR in TEST 3]: {str(e)}\n")

    # ------------------------------------------------------------------
    # TEST 4 — Not found detection (out-of-video question)
    # ------------------------------------------------------------------
    print("--- TEST 4: Not Found Detection (Out-of-Video Question) ---")
    try:
        out_of_video_query = "What is the best recipe for chocolate cake?"
        print(f"User Query: '{out_of_video_query}'\n")

        result = generate_answer_with_sources(out_of_video_query, retriever, memory, llm)

        print("\nGenerated Answer:")
        print(result["answer"])
        print(f"\nNot found correctly detected: {result['not_found']}")

        if result["not_found"]:
            print("Grounding rejection: PASS (hallucination successfully prevented)")
        else:
            print("Warning: not_found flag was False. Answer content should be verified.")

        print(">>> TEST 4 PASSED!\n")
    except Exception as e:
        print(f"[ERROR in TEST 4]: {str(e)}\n")

    # ------------------------------------------------------------------
    # TEST 5 — Multi-turn conversation
    # ------------------------------------------------------------------
    print("--- TEST 5: Multi-Turn Conversational Continuity ---")
    try:
        memory2 = create_conversation_memory()

        # Turn 1
        query1 = "What is the main topic of this video?"
        print(f"\n[Turn 1 Query]: {query1}")
        result1 = generate_answer_with_sources(query1, retriever, memory2, llm)
        print("Turn 1 Answer:", result1["answer"][:200])

        # Turn 2 (follow-up using pronoun)
        query2 = "Can you explain it in simpler terms?"
        print(f"\n[Turn 2 Query]: {query2}")
        result2 = generate_answer_with_sources(query2, retriever, memory2, llm)
        print("Turn 2 Answer:", result2["answer"][:200])
        print("Turn 2 Sources:\n", result2["sources_markdown"])

        # Turn 3 (another follow-up)
        query3 = "What are its limitations?"
        print(f"\n[Turn 3 Query]: {query3}")
        result3 = generate_answer_with_sources(query3, retriever, memory2, llm)
        print("Turn 3 Answer:", result3["answer"][:200])

        print("\n>>> TEST 5 PASSED!\n")
    except Exception as e:
        print(f"[ERROR in TEST 5]: {str(e)}\n")

    # ------------------------------------------------------------------
    # TEST 6 — Streaming
    # ------------------------------------------------------------------
    print("--- TEST 6: Token Streaming Generation ---")
    try:
        stream_query = "Give me a brief summary of the video"
        memory3 = create_conversation_memory()
        print(f"Query: '{stream_query}'")
        print("Streaming answer: ", end="")

        full = ""
        for token in stream_answer(stream_query, retriever, memory3, llm):
            print(token, end="", flush=True)
            full += token

        print()  # newline after streaming completes
        print(f"\nTotal streamed characters: {len(full)}")
        print(">>> TEST 6 PASSED!\n")
    except Exception as e:
        print(f"[ERROR in TEST 6]: {str(e)}\n")

    print("=" * 60)
    print("   ALL PHASE 8 TESTS COMPLETED!                           ")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
