"""Tests for chat completions."""

import pytest
import httpx
import respx

from kafeido import OpenAI, ChatCompletion


@respx.mock
def test_chat_completion_basic(client, base_url, mock_chat_response):
    """Test basic chat completion."""
    # Mock the API endpoint
    route = respx.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=mock_chat_response)
    )

    # Make request
    response = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[{"role": "user", "content": "Hello"}]
    )

    # Assertions
    assert isinstance(response, ChatCompletion)
    assert response.id == "chatcmpl-123"
    assert response.model == "gpt-oss-20b"
    assert len(response.choices) == 1
    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.choices[0].message.role == "assistant"
    assert response.usage.total_tokens == 30

    # Verify request was made
    assert route.called
    request = route.calls.last.request
    assert request.method == "POST"

    # Verify request body
    import json
    body = json.loads(request.content)
    assert body["model"] == "gpt-oss-20b"
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "Hello"


@respx.mock
def test_chat_completion_with_parameters(client, base_url, mock_chat_response):
    """Test chat completion with optional parameters."""
    route = respx.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=mock_chat_response)
    )

    response = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        max_tokens=100,
        top_p=0.9
    )

    assert response.id == "chatcmpl-123"

    # Verify parameters were sent
    import json
    body = json.loads(route.calls.last.request.content)
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 100
    assert body["top_p"] == 0.9


@respx.mock
def test_chat_completion_error_handling(client, base_url):
    """Test error handling for chat completions."""
    from kafeido import AuthenticationError

    # Mock 401 error
    respx.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(
            401,
            json={"error": {"message": "Invalid API key"}}
        )
    )

    # Should raise AuthenticationError
    with pytest.raises(AuthenticationError) as exc_info:
        client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[{"role": "user", "content": "Hello"}]
        )

    assert "Invalid API key" in str(exc_info.value)
    assert exc_info.value.status_code == 401


@respx.mock
def test_chat_completion_system_message(client, base_url, mock_chat_response):
    """Test chat completion with system message."""
    route = respx.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=mock_chat_response)
    )

    response = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
    )

    assert response.id == "chatcmpl-123"

    # Verify both messages were sent
    import json
    body = json.loads(route.calls.last.request.content)
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"
