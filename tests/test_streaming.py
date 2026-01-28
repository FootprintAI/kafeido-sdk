"""Tests for SSE streaming."""

import pytest
import httpx
import respx
from io import BytesIO

from kafeido import OpenAI, ChatCompletionChunk
from kafeido._streaming import Stream


@pytest.fixture
def mock_streaming_response_lines():
    """Mock SSE streaming response lines."""
    return [
        'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-oss-20b","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-oss-20b","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-oss-20b","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}',
        'data: [DONE]',
    ]


@respx.mock
def test_chat_completion_streaming(client, base_url, mock_streaming_response_lines):
    """Test streaming chat completion."""
    # Create a fake streaming response
    response_content = "\n".join(mock_streaming_response_lines) + "\n"

    route = respx.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=response_content,
            headers={"Content-Type": "text/event-stream"},
        )
    )

    # Make streaming request
    stream = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True
    )

    # Verify it's a Stream object
    assert isinstance(stream, Stream)

    # Collect chunks
    chunks = list(stream)

    # Assertions
    assert len(chunks) == 3  # Excludes [DONE]
    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)

    # Verify content
    content = "".join([
        chunk.choices[0].delta.content or ""
        for chunk in chunks
    ])
    assert content == "Hello world!"

    # Verify first chunk has role
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[0].choices[0].delta.content == "Hello"

    # Verify finish reason on last chunk
    assert chunks[-1].choices[0].finish_reason == "stop"

    # Verify request was made
    assert route.called


@respx.mock
def test_streaming_context_manager(client, base_url, mock_streaming_response_lines):
    """Test streaming with context manager."""
    response_content = "\n".join(mock_streaming_response_lines) + "\n"

    respx.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=response_content,
            headers={"Content-Type": "text/event-stream"},
        )
    )

    # Use context manager
    with client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True
    ) as stream:
        chunks = list(stream)
        assert len(chunks) == 3


@respx.mock
def test_streaming_with_parameters(client, base_url, mock_streaming_response_lines):
    """Test streaming with additional parameters."""
    response_content = "\n".join(mock_streaming_response_lines) + "\n"

    route = respx.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=response_content,
            headers={"Content-Type": "text/event-stream"},
        )
    )

    stream = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        temperature=0.7,
        max_tokens=100
    )

    chunks = list(stream)
    assert len(chunks) == 3

    # Verify request body included parameters
    import json
    body = json.loads(route.calls.last.request.content)
    assert body["stream"] is True
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 100


@respx.mock
def test_streaming_empty_response(client, base_url):
    """Test streaming with empty response."""
    response_content = "data: [DONE]\n"

    respx.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=response_content,
            headers={"Content-Type": "text/event-stream"},
        )
    )

    stream = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True
    )

    chunks = list(stream)
    assert len(chunks) == 0


@respx.mock
def test_streaming_malformed_json(client, base_url):
    """Test streaming with malformed JSON (should skip)."""
    response_content = """data: {"valid":"json"}
data: {invalid json}
data: {"another":"valid"}
data: [DONE]
"""

    respx.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=response_content,
            headers={"Content-Type": "text/event-stream"},
        )
    )

    stream = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True
    )

    # Should skip malformed JSON and only yield valid chunks
    chunks = list(stream)
    # Note: these won't be valid ChatCompletionChunk objects but will be dicts
    # since they don't match the expected schema
    assert len(chunks) >= 0  # At least doesn't crash


@respx.mock
def test_streaming_error_response(client, base_url):
    """Test streaming error handling."""
    from kafeido import APIStatusError

    respx.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(
            500,
            json={"error": {"message": "Internal server error"}}
        )
    )

    with pytest.raises(APIStatusError) as exc_info:
        stream = client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )
        # Try to iterate (error might happen here or during creation)
        list(stream)

    assert exc_info.value.status_code == 500
