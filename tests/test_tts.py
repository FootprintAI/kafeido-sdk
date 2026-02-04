"""Tests for TTS (text-to-speech) resource."""

import pytest
import httpx
import respx

from kafeido import CreateSpeechAsyncResponse, GetSpeechResultResponse


@respx.mock
def test_speech_create(client, base_url):
    """Test creating a TTS job."""
    mock_response = {"job_id": "tts-job-123", "status": "pending"}
    route = respx.post(f"{base_url}/v1/audio/speech").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.audio.speech.create(
        model="qwen3-tts",
        input="Hello, world!",
        voice="alloy",
    )

    assert isinstance(result, CreateSpeechAsyncResponse)
    assert result.job_id == "tts-job-123"
    assert result.status == "pending"
    assert route.called


@respx.mock
def test_speech_create_with_params(client, base_url):
    """Test creating a TTS job with all parameters."""
    mock_response = {"job_id": "tts-job-456", "status": "pending"}
    route = respx.post(f"{base_url}/v1/audio/speech").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.audio.speech.create(
        model="xtts-v2",
        input="Test speech synthesis.",
        voice="nova",
        response_format="mp3",
        speed=1.5,
        language="en",
        temperature=0.7,
    )

    assert isinstance(result, CreateSpeechAsyncResponse)
    assert result.job_id == "tts-job-456"
    assert route.called


@respx.mock
def test_speech_get_result_completed(client, base_url):
    """Test getting a completed TTS job result."""
    mock_response = {
        "status": "completed",
        "progress": 100.0,
        "result": {
            "download_url": "https://storage.example.com/audio/output.mp3",
            "duration": 3.5,
        },
    }
    route = respx.get(f"{base_url}/v1/audio/speech/tts-job-123").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.audio.speech.get_result(job_id="tts-job-123")

    assert isinstance(result, GetSpeechResultResponse)
    assert result.status == "completed"
    assert result.progress == 100.0
    assert result.result is not None
    assert result.result.download_url == "https://storage.example.com/audio/output.mp3"
    assert route.called


@respx.mock
def test_speech_get_result_pending(client, base_url):
    """Test getting a pending TTS job result."""
    mock_response = {"status": "processing", "progress": 45.0}
    route = respx.get(f"{base_url}/v1/audio/speech/tts-job-789").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.audio.speech.get_result(job_id="tts-job-789")

    assert result.status == "processing"
    assert result.progress == 45.0
    assert result.result is None
    assert route.called
