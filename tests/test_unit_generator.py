"""
Unit tests for generator.py
Tests that do not require API calls.
"""

import pytest
from app.generation.generator import (
    detect_not_found,
    build_system_prompt,
    needs_rewriting,
    format_sources_for_display,
)


class TestDetectNotFound:
    
    def test_detects_could_not_find(self):
        answer = "I could not find information about this in the video."
        assert detect_not_found(answer) is True
    
    def test_detects_not_mentioned(self):
        answer = "This topic is not mentioned in the transcript."
        assert detect_not_found(answer) is True
    
    def test_normal_answer_returns_false(self):
        answer = "The speaker explains that attention works by..."
        assert detect_not_found(answer) is False
    
    def test_case_insensitive(self):
        answer = "COULD NOT FIND this information."
        assert detect_not_found(answer) is True
    
    def test_empty_string_returns_false(self):
        assert detect_not_found("") is False


class TestBuildSystemPrompt:
    
    def test_returns_string(self):
        prompt = build_system_prompt()
        assert isinstance(prompt, str)
    
    def test_contains_grounding_instruction(self):
        prompt = build_system_prompt()
        assert "ONLY" in prompt or "only" in prompt
    
    def test_contains_not_found_instruction(self):
        prompt = build_system_prompt()
        assert "not found" in prompt.lower() or \
               "cannot find" in prompt.lower()
    
    def test_not_empty(self):
        prompt = build_system_prompt()
        assert len(prompt) > 100


class TestNeedsRewriting:
    
    def test_first_message_never_needs_rewriting(self):
        result = needs_rewriting(
            "What is attention?", 
            "No previous conversation."
        )
        assert result is False
    
    def test_pronoun_needs_rewriting(self):
        result = needs_rewriting(
            "What are its limitations?",
            "User: What is attention?\nAssistant: Attention is..."
        )
        assert result is True
    
    def test_standalone_question_no_rewrite(self):
        result = needs_rewriting(
            "What is the transformer architecture?",
            "User: What is attention?\nAssistant: Attention is..."
        )
        assert result is False


class TestFormatSourcesForDisplay:
    
    def test_empty_sources_returns_string(self):
        result = format_sources_for_display([])
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_formats_sources_correctly(self):
        sources = [
            {
                "chunk_id": 5,
                "start_timestamp_formatted": "12:14",
                "timestamp_url": "https://youtube.com/watch?v=abc&t=734s",
                "text_preview": "Attention is a mechanism...",
                "title": "Test Video",
                "channel": "Test Channel",
            }
        ]
        result = format_sources_for_display(sources)
        assert "12:14" in result
        assert isinstance(result, str)
