"""
Custom exception classes for YouTube RAG Assistant.
Using specific exception types makes error handling cleaner
and error messages more helpful to the user.
"""


class RAGException(Exception):
    """Base exception for all RAG pipeline errors."""

    pass


class InvalidURLError(RAGException):
    """Raised when the YouTube URL cannot be parsed."""

    def __init__(self, url: str):
        self.url = url
        super().__init__(
            f"Invalid YouTube URL: '{url}'. "
            "Please provide a URL like: "
            "https://www.youtube.com/watch?v=VIDEO_ID"
        )


class TranscriptNotAvailableError(RAGException):
    """Raised when the video has no available transcript."""

    def __init__(self, video_id: str, reason: str = ""):
        self.video_id = video_id
        super().__init__(
            f"No transcript available for video '{video_id}'. "
            f"{reason} "
            "Try a video that has captions enabled."
        )


class VideoUnavailableError(RAGException):
    """Raised when the video cannot be accessed."""

    def __init__(self, video_id: str):
        self.video_id = video_id
        super().__init__(
            f"Video '{video_id}' is unavailable. "
            "It may be private, deleted, or age-restricted."
        )


class EmbeddingError(RAGException):
    """Raised when the embedding model fails."""

    def __init__(self, reason: str):
        super().__init__(
            f"Embedding generation failed: {reason}. "
            "Check that sentence-transformers is installed correctly."
        )


class VectorStoreError(RAGException):
    """Raised when the vector store cannot be built or loaded."""

    def __init__(self, reason: str):
        super().__init__(
            f"Vector store error: {reason}. "
            "Try processing the video again."
        )


class LLMError(RAGException):
    """Raised when the Groq LLM call fails."""

    def __init__(self, reason: str):
        super().__init__(
            f"LLM generation failed: {reason}. "
            "Check your GROQ_API_KEY and internet connection."
        )


class RetrievalError(RAGException):
    """Raised when retrieval fails unexpectedly."""

    def __init__(self, query: str, reason: str):
        self.query = query
        super().__init__(
            f"Retrieval failed for query '{query[:50]}': {reason}"
        )
