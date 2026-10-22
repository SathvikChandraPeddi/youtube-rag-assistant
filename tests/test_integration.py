"""
Integration tests for the full RAG pipeline.
These tests make real network calls and take longer to run.
Run with: pytest tests/test_integration.py -v -s

These tests require:
- A valid GROQ_API_KEY in .env
- Internet connection
- The video https://www.youtube.com/watch?v=zjkBMFhNj_g to be available
"""

import pytest
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_KEY_AVAILABLE = bool(os.getenv("GROQ_API_KEY", "").strip())
VIDEO_URL = "https://www.youtube.com/watch?v=zjkBMFhNj_g"


@pytest.mark.skipif(
    not GROQ_KEY_AVAILABLE, 
    reason="GROQ_API_KEY not set"
)
class TestFullPipeline:
    
    @pytest.fixture(scope="class")
    def pipeline_components(self):
        from app.retrieval.vectorstore import index_video
        from app.retrieval.retriever import build_retriever
        from app.generation.generator import create_conversation_memory
        from app.utils.helpers import get_groq_llm
        
        vectorstore, metadata = index_video(VIDEO_URL)
        retriever = build_retriever(vectorstore, k=5)
        memory = create_conversation_memory()
        llm = get_groq_llm()
        
        return {
            "vectorstore": vectorstore,
            "retriever": retriever,
            "memory": memory,
            "llm": llm,
            "metadata": metadata,
        }
    
    def test_indexing_returns_metadata(self, pipeline_components):
        metadata = pipeline_components["metadata"]
        assert "video_id" in metadata
        assert "total_chunks" in metadata
        assert metadata["total_chunks"] > 0
    
    def test_retrieval_returns_documents(self, pipeline_components):
        from app.retrieval.retriever import retrieve_chunks
        docs = retrieve_chunks(
            pipeline_components["retriever"],
            "what is this video about"
        )
        assert len(docs) > 0
        assert len(docs) <= 5
    
    def test_answer_generation_returns_string(self, pipeline_components):
        from app.generation.generator import generate_answer_with_sources
        result = generate_answer_with_sources(
            "What is this video about?",
            pipeline_components["retriever"],
            pipeline_components["memory"],
            pipeline_components["llm"],
        )
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 20
        assert "sources" in result
    
    def test_not_found_detection_works(self, pipeline_components):
        from app.generation.generator import (
            generate_answer_with_sources,
            create_conversation_memory,
        )
        fresh_memory = create_conversation_memory()
        result = generate_answer_with_sources(
            "What is the best chocolate cake recipe?",
            pipeline_components["retriever"],
            fresh_memory,
            pipeline_components["llm"],
        )
        assert result["not_found"] is True
