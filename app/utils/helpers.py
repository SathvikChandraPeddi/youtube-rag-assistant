"""
Shared utilities used across all modules.
Provides the Groq LLM instance (LLaMA 3.3 70B via Groq API),
the local HuggingFace embedding model (all-MiniLM-L6-v2),
and the cross-encoder reranker model (ms-marco-MiniLM-L-6-v2).
"""

import os
from dotenv import load_dotenv

from app.utils.logger import get_logger
from app.utils.errors import LLMError

logger = get_logger(__name__)

# Module-level caches for models to avoid redundant loads
_reranker_model = None


def get_groq_llm():
    """
    Returns a LangChain ChatOpenAI instance pointed at Groq's API.
    """
    from langchain_openai import ChatOpenAI

    load_dotenv(override=True)

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        logger.error("GROQ_API_KEY not found in environment.")
        raise LLMError(
            "GROQ_API_KEY not found. "
            "Please add it to your .env file as: GROQ_API_KEY=your_key_here"
        )

    logger.info("Groq API key loaded: %s...", api_key[:8])

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0.2,
        max_tokens=1024,
    )


def check_environment() -> bool:
    """
    Check that all required environment variables are set.
    Call this at app startup to catch missing keys early.
    """
    load_dotenv(override=True)

    required_keys = ["GROQ_API_KEY"]
    missing = []

    for key in required_keys:
        value = os.getenv(key, "").strip()
        if not value:
            missing.append(key)
        else:
            logger.info("Environment variable loaded: %s", key)

    if missing:
        logger.error("Missing environment variables: %s", missing)
        return False

    logger.info("All environment variables loaded.")
    return True


def get_embedding_model():
    """
    Returns a HuggingFace sentence-transformers embedding model.
    Model: all-MiniLM-L6-v2
    Runs fully locally. No API key required.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_reranker_model():
    """
    Load and return a HuggingFace cross-encoder model for reranking.
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    Runs fully locally on CPU. No API key required.
    """
    global _reranker_model

    # Return cached instance if already loaded
    if _reranker_model is not None:
        logger.info("Using cached reranker model.")
        return _reranker_model

    from sentence_transformers import CrossEncoder

    logger.info("Loading reranker model: cross-encoder/ms-marco-MiniLM-L-6-v2...")
    _reranker_model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length=512,
        device="cpu",
    )
    logger.info("Reranker model loaded: cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model
