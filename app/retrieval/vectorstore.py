"""
Vector store module.
Embeds chunks using HuggingFace all-MiniLM-L6-v2, builds a FAISS vector index,
and persists the index to disk. Also handles loading existing indexes on startup.
"""

import os
from dotenv import load_dotenv

from app.utils.errors import VectorStoreError
from app.utils.logger import get_logger

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    try:
        from langchain_community.vectorstores.faiss import FAISS
    except ImportError:
        FAISS = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None

logger = get_logger(__name__)
load_dotenv()

# Module-level cache for the embedding model to avoid redundant loads
_embedding_model = None


def get_embedding_model():
    """
    Load and return the HuggingFace sentence-transformer embedding model.
    Caches the instance in _embedding_model for subsequent calls.
    """
    global _embedding_model

    if _embedding_model is not None:
        logger.debug("Using cached embedding model.")
        return _embedding_model

    if HuggingFaceEmbeddings is None:
        raise VectorStoreError(
            "HuggingFaceEmbeddings is not available. Please install sentence-transformers and langchain-community."
        )

    logger.info("Loading embedding model: all-MiniLM-L6-v2...")
    _embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Embedding model loaded.")
    return _embedding_model


def build_vectorstore(chunks: list):
    """
    Take the list of LangChain Document chunks and build a FAISS vector store.
    """
    if not chunks:
        logger.warning("No chunks provided to build_vectorstore.")
        raise VectorStoreError("Cannot build vector store from empty chunk list.")

    if FAISS is None:
        raise VectorStoreError(
            "FAISS is not available. Please ensure faiss-cpu and langchain-community are installed."
        )

    embedding_model = get_embedding_model()
    logger.info("Building vector store from %d chunks...", len(chunks))
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    logger.info("Vector store built successfully.")
    return vectorstore


def save_vectorstore(vectorstore, path: str = "vectorstore/index"):
    """
    Save the FAISS index to disk so re-embedding is not required on app restart.
    """
    folder = os.path.dirname(path) or "."
    index_name = os.path.basename(path) or "index"
    os.makedirs(folder, exist_ok=True)

    vectorstore.save_local(folder_path=folder, index_name=index_name)
    logger.info("Vector store saved to %s", path)


def load_vectorstore(path: str = "vectorstore/index"):
    """
    Load a previously saved FAISS index from disk.
    """
    folder = os.path.dirname(path) or "."
    index_name = os.path.basename(path) or "index"

    faiss_file = f"{path}.faiss"
    pkl_file = f"{path}.pkl"
    dir_faiss = os.path.join(path, f"{index_name}.faiss")
    dir_pkl = os.path.join(path, f"{index_name}.pkl")

    if not (os.path.exists(faiss_file) and os.path.exists(pkl_file)):
        if os.path.exists(dir_faiss) and os.path.exists(dir_pkl):
            folder = path
        else:
            raise VectorStoreError(
                f"No vector store found at {path}. Process a video first."
            )

    embedding_model = get_embedding_model()

    if FAISS is None:
        raise VectorStoreError(
            "FAISS is not available. Please ensure faiss-cpu and langchain-community are installed."
        )

    vectorstore = FAISS.load_local(
        folder_path=folder,
        embeddings=embedding_model,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )

    logger.info("Vector store loaded from %s", path)
    return vectorstore


def index_video(
    url: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    save_path: str = "vectorstore/index",
) -> tuple:
    """
    Master indexing function running the complete pipeline:
    YouTube URL -> transcript -> chunks -> embeddings -> FAISS index -> saved to disk.
    """
    try:
        logger.info("Starting indexing pipeline for: %s", url)

        from app.ingestion.transcript import build_transcript_documents
        documents = build_transcript_documents(url)
        logger.info("Step 1/4: Transcript fetched.")

        from app.ingestion.chunker import create_chunks
        chunks = create_chunks(
            documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        logger.info("Step 2/4: Created %d chunks.", len(chunks))

        vectorstore = build_vectorstore(chunks)
        logger.info("Step 3/4: Vector store built.")

        save_vectorstore(vectorstore, save_path)
        logger.info("Step 4/4: Vector store saved.")

        first_meta = chunks[0].metadata if chunks else {}
        video_metadata = {
            "video_id": first_meta.get("video_id", ""),
            "title": first_meta.get("title", "Unknown Title"),
            "channel": first_meta.get("channel", "Unknown Channel"),
            "duration_seconds": first_meta.get("duration_seconds", 0),
            "total_chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "url": url,
        }

        logger.info("Indexing complete.")
        return vectorstore, video_metadata

    except Exception as e:
        logger.error("Indexing pipeline failed for %s: %s", url, str(e))
        raise


def vectorstore_exists(path: str = "vectorstore/index") -> bool:
    """
    Check whether a saved vector store already exists at the given path.
    """
    faiss_path = f"{path}.faiss"
    pkl_path = f"{path}.pkl"

    if os.path.exists(faiss_path) and os.path.exists(pkl_path):
        return True

    index_name = os.path.basename(path) or "index"
    dir_faiss = os.path.join(path, f"{index_name}.faiss")
    dir_pkl = os.path.join(path, f"{index_name}.pkl")

    if os.path.exists(dir_faiss) and os.path.exists(dir_pkl):
        return True

    return False
