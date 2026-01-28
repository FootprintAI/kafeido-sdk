"""Tests for audio transcriptions and translations."""

import pytest
import httpx
import respx
from io import BytesIO

from kafeido import (
    OpenAI,
    Transcription,
    Translation,
    AsyncTranscriptionResponse,
    AsyncTranscriptionResult,
)


@pytest.fixture
def mock_transcription_response():
    """Mock transcription API response."""
    return {
        "text": "Hello, this is a test transcription."
    }


@pytest.fixture
def mock_translation_response():
    """Mock translation API response."""
    return {
        "text": "Hello, this is a test translation."
    }


@respx.mock
def test_transcription_create(client, base_url, mock_transcription_response):
    """Test basic audio transcription."""
    # Mock the API endpoint
    route = respx.post(f"{base_url}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json=mock_transcription_response)
    )

    # Create fake audio file
    audio_file = BytesIO(b"fake audio data")
    audio_file.name = "test.mp3"

    # Make request
    response = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3"
    )

    # Assertions
    assert isinstance(response, Transcription)
    assert response.text == "Hello, this is a test transcription."

    # Verify request was made
    assert route.called
    request = route.calls.last.request
    assert request.method == "POST"


@respx.mock
def test_transcription_with_language(client, base_url, mock_transcription_response):
    """Test transcription with language parameter."""
    route = respx.post(f"{base_url}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json=mock_transcription_response)
    )

    audio_file = BytesIO(b"fake audio data")
    audio_file.name = "test.mp3"

    response = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3",
        language="en"
    )

    assert response.text == "Hello, this is a test transcription."
    assert route.called


@respx.mock
def test_transcription_with_prompt(client, base_url, mock_transcription_response):
    """Test transcription with prompt parameter."""
    route = respx.post(f"{base_url}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json=mock_transcription_response)
    )

    audio_file = BytesIO(b"fake audio data")
    audio_file.name = "test.mp3"

    response = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-turbo",
        prompt="This is a test prompt"
    )

    assert response.text == "Hello, this is a test transcription."
    assert route.called


@respx.mock
def test_transcription_with_temperature(client, base_url, mock_transcription_response):
    """Test transcription with temperature parameter."""
    route = respx.post(f"{base_url}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json=mock_transcription_response)
    )

    audio_file = BytesIO(b"fake audio data")
    audio_file.name = "test.mp3"

    response = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3",
        temperature=0.5
    )

    assert response.text == "Hello, this is a test transcription."
    assert route.called


@respx.mock
def test_translation_create(client, base_url, mock_translation_response):
    """Test basic audio translation."""
    # Mock the API endpoint
    route = respx.post(f"{base_url}/v1/audio/translations").mock(
        return_value=httpx.Response(200, json=mock_translation_response)
    )

    # Create fake audio file
    audio_file = BytesIO(b"fake audio data")
    audio_file.name = "test.mp3"

    # Make request
    response = client.audio.translations.create(
        file=audio_file,
        model="whisper-large-v3"
    )

    # Assertions
    assert isinstance(response, Translation)
    assert response.text == "Hello, this is a test translation."

    # Verify request was made
    assert route.called
    request = route.calls.last.request
    assert request.method == "POST"


@respx.mock
def test_translation_with_prompt(client, base_url, mock_translation_response):
    """Test translation with prompt parameter."""
    route = respx.post(f"{base_url}/v1/audio/translations").mock(
        return_value=httpx.Response(200, json=mock_translation_response)
    )

    audio_file = BytesIO(b"fake audio data")
    audio_file.name = "test.mp3"

    response = client.audio.translations.create(
        file=audio_file,
        model="whisper-large-v3",
        prompt="This is a test prompt"
    )

    assert response.text == "Hello, this is a test translation."
    assert route.called


@respx.mock
def test_transcription_error_handling(client, base_url):
    """Test error handling for transcriptions."""
    from kafeido import NotFoundError

    # Mock 404 error
    respx.post(f"{base_url}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(
            404,
            json={"error": {"message": "Model not found"}}
        )
    )

    audio_file = BytesIO(b"fake audio data")
    audio_file.name = "test.mp3"

    # Should raise NotFoundError
    with pytest.raises(NotFoundError) as exc_info:
        client.audio.transcriptions.create(
            file=audio_file,
            model="nonexistent-model"
        )

    assert "Model not found" in str(exc_info.value)
    assert exc_info.value.status_code == 404


@respx.mock
def test_transcription_create_async(client, base_url):
    """Test creating an async transcription job."""
    mock_response = {"job_id": "asr-job-123", "status": "pending"}
    route = respx.post(f"{base_url}/v1/audio/transcriptions/async").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.audio.transcriptions.create_async(
        storage_key="org_123/audio.mp3",
        model="whisper-large-v3",
    )

    assert isinstance(result, AsyncTranscriptionResponse)
    assert result.job_id == "asr-job-123"
    assert result.status == "pending"
    assert route.called


@respx.mock
def test_transcription_get_result(client, base_url):
    """Test getting async transcription result."""
    mock_response = {
        "status": "completed",
        "progress": 100.0,
        "result": {"text": "Transcribed text from async job."},
    }
    route = respx.get(f"{base_url}/v1/audio/transcriptions/async/asr-job-123").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.audio.transcriptions.get_result(job_id="asr-job-123")

    assert isinstance(result, AsyncTranscriptionResult)
    assert result.status == "completed"
    assert result.result is not None
    assert result.result.text == "Transcribed text from async job."
    assert route.called
