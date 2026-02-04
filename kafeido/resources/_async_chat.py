"""Async chat completions resource."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import httpx

from kafeido._http_client import AsyncHTTPClient
from kafeido._streaming import AsyncStream
from kafeido.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)


class AsyncCompletions:
    """Async chat completions endpoint."""

    def __init__(self, http_client: AsyncHTTPClient) -> None:
        """Initialize async completions resource.

        Args:
            http_client: The async HTTP client to use for requests.
        """
        self._client = http_client

    async def create(
        self,
        *,
        model: str,
        messages: List[Union[Dict[str, Any], ChatCompletionMessageParam]],
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        user: Optional[str] = None,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        """Create a chat completion asynchronously.

        Args:
            model: ID of the model to use (e.g., "gpt-oss-20b").
            messages: List of messages in the conversation.
            frequency_penalty: Penalize new tokens based on frequency.
            logit_bias: Modify likelihood of specific tokens.
            logprobs: Whether to return log probabilities.
            top_logprobs: Number of most likely tokens to return.
            max_tokens: Maximum tokens to generate.
            n: Number of completions to generate.
            presence_penalty: Penalize new tokens based on presence.
            response_format: Format of the response.
            seed: Random seed for deterministic sampling.
            stop: Stop sequences.
            stream: Whether to stream the response.
            temperature: Sampling temperature (0-2).
            top_p: Nucleus sampling parameter.
            tools: List of tools the model can call.
            tool_choice: Controls which tool is called.
            user: Unique identifier for the end-user.

        Returns:
            ChatCompletion or AsyncStream[ChatCompletionChunk] if streaming.

        Example:
            >>> client = AsyncOpenAI(api_key="sk-...")
            >>> response = await client.chat.completions.create(
            ...     model="gpt-oss-20b",
            ...     messages=[{"role": "user", "content": "Hello!"}]
            ... )
            >>> print(response.choices[0].message.content)
        """
        # Build request body
        body: Dict[str, Any] = {
            "model": model,
            "messages": [
                msg if isinstance(msg, dict) else msg.model_dump(exclude_none=True)
                for msg in messages
            ],
        }

        # Add optional parameters
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        if logit_bias is not None:
            body["logit_bias"] = logit_bias
        if logprobs is not None:
            body["logprobs"] = logprobs
        if top_logprobs is not None:
            body["top_logprobs"] = top_logprobs
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if n is not None:
            body["n"] = n
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        if response_format is not None:
            body["response_format"] = response_format
        if seed is not None:
            body["seed"] = seed
        if stop is not None:
            body["stop"] = stop
        if stream:
            body["stream"] = True
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if user is not None:
            body["user"] = user

        # Handle streaming
        if stream:
            response = await self._client.request(
                "POST",
                "/v1/chat/completions",
                json=body,
                stream=True,
            )
            assert isinstance(response, httpx.Response)
            return AsyncStream(response, ChatCompletionChunk)

        # Non-streaming request
        response_data = await self._client.post("/v1/chat/completions", json=body)
        return ChatCompletion.model_validate(response_data)


class AsyncChat:
    """Async chat resource."""

    def __init__(self, http_client: AsyncHTTPClient) -> None:
        """Initialize async chat resource.

        Args:
            http_client: The async HTTP client to use for requests.
        """
        self._client = http_client
        self._completions = AsyncCompletions(http_client)

    @property
    def completions(self) -> AsyncCompletions:
        """Access async chat completions endpoint."""
        return self._completions
