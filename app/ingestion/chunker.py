"""
Chunking module.
Splits cleaned transcript documents into smaller overlapping chunks
suitable for semantic retrieval. Preserves metadata such as timestamps
on every chunk so the source can be traced back to the video.
"""

from app.utils.logger import get_logger

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

logger = get_logger(__name__)


class _FallbackRecursiveTextSplitter:
    """Lightweight fallback implementing recursive character splitting."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separators: list = None,
        length_function=len,
        add_start_index: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        self.length_function = length_function
        self.add_start_index = add_start_index

    def split_text(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        separator = self.separators[-1]
        for sep in self.separators:
            if sep in text:
                separator = sep
                break

        splits = text.split(separator) if separator else list(text)
        chunks = []
        current = []
        current_len = 0

        for s in splits:
            item_len = len(s) + (len(separator) if current else 0)
            if current_len + item_len > self.chunk_size and current:
                chunk = separator.join(current)
                chunks.append(chunk)
                while current and current_len > self.chunk_overlap:
                    removed = current.pop(0)
                    current_len -= len(removed) + (len(separator) if current else 0)
            current.append(s)
            current_len += len(s) + (len(separator) if len(current) > 1 else 0)

        if current:
            chunks.append(separator.join(current))
        return chunks

    def split_documents(self, documents: list) -> list:
        results = []
        for doc in documents:
            text_splits = self.split_text(doc.page_content)
            for split in text_splits:
                meta = dict(doc.metadata)
                results.append(Document(page_content=split, metadata=meta))
        return results


def format_timestamp(seconds: float) -> str:
    """
    Convert a float number of seconds into a human-readable MM:SS string.
    """
    try:
        total_seconds = max(0, int(seconds))
    except (TypeError, ValueError):
        total_seconds = 0

    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes}:{secs:02d}"


def build_youtube_timestamp_url(video_id: str, seconds: float) -> str:
    """
    Build a YouTube URL that deep-links to a specific timestamp in the video.
    """
    try:
        sec_int = max(0, int(seconds))
    except (TypeError, ValueError):
        sec_int = 0

    return f"https://www.youtube.com/watch?v={video_id}&t={sec_int}s"


def estimate_chunk_start_time(chunk_text: str, source_documents: list) -> float:
    """
    Given a chunk's text, find the most likely start timestamp by matching
    it against original transcript segments.
    """
    if not chunk_text or not source_documents:
        return 0.0

    try:
        probe = chunk_text[:80].strip()
        if not probe:
            return 0.0

        for doc in source_documents:
            content = getattr(doc, "page_content", "").strip()
            if not content:
                continue

            if probe in content or (len(content) >= 10 and content in probe):
                return float(doc.metadata.get("start", 0.0))

        first_few_words = " ".join(probe.split()[:4]).lower()
        if first_few_words:
            for doc in source_documents:
                content = getattr(doc, "page_content", "").strip().lower()
                if first_few_words in content:
                    return float(doc.metadata.get("start", 0.0))

    except Exception:
        return 0.0

    return 0.0


def create_chunks(
    documents: list,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list:
    """
    Split the list of LangChain Documents into smaller overlapping chunks
    using RecursiveCharacterTextSplitter.
    """
    if not documents:
        logger.warning("create_chunks received an empty documents list.")
        return []

    splitter_cls = RecursiveCharacterTextSplitter or _FallbackRecursiveTextSplitter
    splitter = splitter_cls(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    if len(documents) > 1 and all(len(d.page_content) < chunk_size for d in documents):
        full_text = " ".join(
            d.page_content.strip() for d in documents if d.page_content.strip()
        )
        base_meta = dict(documents[0].metadata)
        input_docs = [Document(page_content=full_text, metadata=base_meta)]
    else:
        input_docs = documents

    raw_chunks = splitter.split_documents(input_docs)
    enriched_chunks = []
    video_id = documents[0].metadata.get("video_id", "")

    for chunk in raw_chunks:
        content = chunk.page_content.strip()
        if len(content) < 50:
            continue

        start_secs = estimate_chunk_start_time(content, documents)
        formatted_ts = format_timestamp(start_secs)
        timestamp_url = build_youtube_timestamp_url(video_id, start_secs)

        chunk.metadata.update(
            {
                "chunk_id": len(enriched_chunks),
                "chunk_size_used": chunk_size,
                "start_timestamp_seconds": start_secs,
                "start_timestamp_formatted": formatted_ts,
                "timestamp_url": timestamp_url,
            }
        )
        enriched_chunks.append(chunk)

    total_chunks = len(enriched_chunks)
    avg_length = (
        sum(len(c.page_content) for c in enriched_chunks) / total_chunks
        if total_chunks > 0
        else 0
    )

    logger.info("Created %d chunks from %d documents", total_chunks, len(documents))
    logger.info("Chunk size: %d | Overlap: %d", chunk_size, chunk_overlap)
    logger.info("Average chunk length: %.0f characters", avg_length)

    return enriched_chunks


def get_chunk_preview(chunk, max_chars: int = 200) -> str:
    """
    Return a short preview of a chunk's text for display in the UI.
    """
    if not chunk or not hasattr(chunk, "page_content"):
        return ""
    text = chunk.page_content.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def summarize_chunks(chunks: list) -> dict:
    """
    Return a summary dict describing the chunk set.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_length_chars": 0,
            "min_length_chars": 0,
            "max_length_chars": 0,
            "chunk_size_used": None,
            "video_title": None,
        }

    lengths = [len(c.page_content) for c in chunks]
    total = len(chunks)
    avg_length = sum(lengths) / total if total > 0 else 0
    first_metadata = getattr(chunks[0], "metadata", {})

    return {
        "total_chunks": total,
        "avg_length_chars": round(avg_length, 1),
        "min_length_chars": min(lengths) if lengths else 0,
        "max_length_chars": max(lengths) if lengths else 0,
        "chunk_size_used": first_metadata.get("chunk_size_used"),
        "video_title": first_metadata.get("title"),
    }
