"""
Unit tests for transcript.py
These tests do not make network calls — they test logic only.
"""

import pytest
from app.ingestion.transcript import (
    extract_video_id,
    clean_transcript_text,
)
from app.utils.errors import InvalidURLError


class TestExtractVideoId:
    
    def test_standard_watch_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"
    
    def test_short_url(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"
    
    def test_embed_url(self):
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"
    
    def test_url_with_timestamp(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120s"
        assert extract_video_id(url) == "dQw4w9WgXcQ"
    
    def test_url_with_playlist(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLxxx"
        assert extract_video_id(url) == "dQw4w9WgXcQ"
    
    def test_invalid_url_raises_error(self):
        with pytest.raises((InvalidURLError, ValueError)):
            extract_video_id("https://www.google.com")
    
    def test_empty_string_raises_error(self):
        with pytest.raises((InvalidURLError, ValueError)):
            extract_video_id("")
    
    def test_random_string_raises_error(self):
        with pytest.raises((InvalidURLError, ValueError)):
            extract_video_id("not a url at all")


class TestCleanTranscriptText:
    
    def test_removes_extra_spaces(self):
        text = "hello   world   this  is  a  test"
        result = clean_transcript_text(text)
        assert "  " not in result
    
    def test_strips_whitespace(self):
        text = "  hello world  "
        result = clean_transcript_text(text)
        assert result == result.strip()
    
    def test_handles_empty_string(self):
        result = clean_transcript_text("")
        assert isinstance(result, str)
    
    def test_preserves_content(self):
        text = "attention is all you need"
        result = clean_transcript_text(text)
        assert "attention" in result
        assert "need" in result
    
    def test_removes_filler_words(self):
        text = "um so the um model uh uses attention"
        result = clean_transcript_text(text)
        assert result.count("um") < text.count("um")
