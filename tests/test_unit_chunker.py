"""
Unit tests for chunker.py
No network calls — tests chunking logic only.
"""

import pytest
try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document
from app.ingestion.chunker import (
    format_timestamp,
    build_youtube_timestamp_url,
    create_chunks,
    get_chunk_preview,
    summarize_chunks,
)


class TestFormatTimestamp:
    
    def test_zero_seconds(self):
        assert format_timestamp(0.0) == "0:00"
    
    def test_one_minute(self):
        assert format_timestamp(60.0) == "1:00"
    
    def test_mixed_minutes_seconds(self):
        assert format_timestamp(734.0) == "12:14"
    
    def test_negative_treated_as_zero(self):
        result = format_timestamp(-5.0)
        assert result == "0:00"
    
    def test_returns_string(self):
        result = format_timestamp(100.0)
        assert isinstance(result, str)
        assert ":" in result


class TestBuildYouTubeTimestampUrl:
    
    def test_basic_url(self):
        url = build_youtube_timestamp_url("dQw4w9WgXcQ", 734.0)
        assert "dQw4w9WgXcQ" in url
        assert "734" in url
        assert url.startswith("https://")
    
    def test_zero_timestamp(self):
        url = build_youtube_timestamp_url("abc123", 0.0)
        assert "t=0s" in url
    
    def test_float_is_converted_to_int(self):
        url = build_youtube_timestamp_url("abc123", 734.9)
        assert "t=734s" in url
        assert "734.9" not in url


class TestCreateChunks:
    
    @pytest.fixture
    def sample_documents(self):
        long_text = " ".join(["word"] * 500)
        return [
            Document(
                page_content=long_text,
                metadata={
                    "video_id": "test123",
                    "title": "Test Video",
                    "channel": "Test Channel",
                    "duration_seconds": 600,
                    "url": "https://youtube.com/watch?v=test123",
                    "start": 0.0,
                    "duration": 10.0,
                    "source": "YouTube: Test Video"
                }
            ),
            Document(
                page_content=long_text,
                metadata={
                    "video_id": "test123",
                    "title": "Test Video",
                    "channel": "Test Channel",
                    "duration_seconds": 600,
                    "url": "https://youtube.com/watch?v=test123",
                    "start": 60.0,
                    "duration": 10.0,
                    "source": "YouTube: Test Video"
                }
            ),
        ]
    
    def test_returns_list(self, sample_documents):
        chunks = create_chunks(sample_documents)
        assert isinstance(chunks, list)
    
    def test_chunks_have_metadata(self, sample_documents):
        chunks = create_chunks(sample_documents)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert "video_id" in chunk.metadata
            assert "start_timestamp_formatted" in chunk.metadata
            assert "timestamp_url" in chunk.metadata
    
    def test_chunk_ids_are_sequential(self, sample_documents):
        chunks = create_chunks(sample_documents)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert ids == list(range(len(chunks)))
    
    def test_larger_chunk_size_means_fewer_chunks(self, sample_documents):
        chunks_small = create_chunks(sample_documents, chunk_size=200)
        chunks_large = create_chunks(sample_documents, chunk_size=800)
        assert len(chunks_small) >= len(chunks_large)
    
    def test_empty_documents_returns_empty_list(self):
        chunks = create_chunks([])
        assert chunks == []
    
    def test_all_chunks_have_content(self, sample_documents):
        chunks = create_chunks(sample_documents)
        for chunk in chunks:
            assert len(chunk.page_content.strip()) >= 50


class TestGetChunkPreview:
    
    def test_short_text_unchanged(self):
        doc = Document(page_content="short text", metadata={})
        result = get_chunk_preview(doc, max_chars=200)
        assert result == "short text"
    
    def test_long_text_truncated(self):
        doc = Document(page_content="a" * 300, metadata={})
        result = get_chunk_preview(doc, max_chars=200)
        assert len(result) <= 203
        assert result.endswith("...")
