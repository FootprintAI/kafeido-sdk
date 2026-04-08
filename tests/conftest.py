"""Pytest fixtures and configuration."""

import pytest
import httpx
import respx

from kafeido import OpenAI


@pytest.fixture
def api_key():
    """Mock API key for testing."""
    return "sk-test123_dGVzdGtleQ=="


@pytest.fixture
def base_url():
    """Base URL for testing."""
    return "https://api.kafeido.app"


@pytest.fixture
def client(api_key, base_url):
    """Create test client."""
    return OpenAI(api_key=api_key, base_url=base_url)


@pytest.fixture
def mock_chat_response():
    """Mock chat completion response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-oss-20b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop",
                "logprobs": None
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }


@pytest.fixture
def mock_models_list():
    """Mock models list response."""
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-oss-20b",
                "object": "model",
                "created": 1677652288,
                "owned_by": "kafeido"
            },
            {
                "id": "whisper-large-v3",
                "object": "model",
                "created": 1677652288,
                "owned_by": "kafeido"
            }
        ]
    }


@pytest.fixture
def mock_transcription_response():
    """Mock transcription response."""
    return {
        "text": "Hello, this is a test transcription.",
        "task": "transcribe",
        "language": "en",
        "duration": 5.2
    }
