"""Async vision resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from kafeido._http_client import AsyncHTTPClient
from kafeido._streaming import AsyncStream
from kafeido.types.vision import (
    CreateVisionAsyncResponse,
    CreateVisionChatResponse,
    CreateVisionResponse,
    GetVisionResultResponse,
)

if TYPE_CHECKING:
    from kafeido._warmup import AsyncWarmupHelper


class AsyncVisionAnalysis:
    """Async vision analysis endpoint."""

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        warmup_helper: Optional["AsyncWarmupHelper"] = None,
    ) -> None:
        self._client = http_client
        self._warmup_helper = warmup_helper

    async def create(
        self,
        *,
        model_id: str,
        storage_key: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_url: Optional[str] = None,
        prompt: Optional[str] = None,
        mode: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        wait_for_ready: bool = False,
        warmup_timeout: Optional[float] = None,
    ) -> CreateVisionResponse:
        """Analyze an image asynchronously.

        Args:
            model_id: Vision model ID.
            storage_key: Storage key of the image.
            image_base64: Base64-encoded image.
            image_url: URL of the image.
            prompt: Analysis prompt.
            mode: Analysis mode.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            top_p: Top-p sampling.
            top_k: Top-k sampling.
            repetition_penalty: Repetition penalty.
            wait_for_ready: If True, wait for the model to be ready.
            warmup_timeout: Maximum seconds to wait for warmup.

        Returns:
            CreateVisionResponse with analysis text.

        Raises:
            WarmupTimeoutError: If wait_for_ready is True and the model
                doesn't become ready within the timeout period.
        """
        # Handle cold start waiting if enabled
        if wait_for_ready and self._warmup_helper:
            await self._warmup_helper.wait_for_ready(model_id, timeout=warmup_timeout)

        body: Dict[str, Any] = {"model_id": model_id}

        if storage_key is not None:
            body["storage_key"] = storage_key
        if image_base64 is not None:
            body["image_base64"] = image_base64
        if image_url is not None:
            body["image_url"] = image_url
        if prompt is not None:
            body["prompt"] = prompt
        if mode is not None:
            body["mode"] = mode
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if repetition_penalty is not None:
            body["repetition_penalty"] = repetition_penalty

        response_data = await self._client.post("/v1/vision/analyze", json=body)
        return CreateVisionResponse.model_validate(response_data)

    async def create_async(
        self,
        *,
        model_id: str,
        storage_key: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_url: Optional[str] = None,
        prompt: Optional[str] = None,
        mode: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> CreateVisionAsyncResponse:
        """Create an async vision analysis job."""
        body: Dict[str, Any] = {"model_id": model_id}

        if storage_key is not None:
            body["storage_key"] = storage_key
        if image_base64 is not None:
            body["image_base64"] = image_base64
        if image_url is not None:
            body["image_url"] = image_url
        if prompt is not None:
            body["prompt"] = prompt
        if mode is not None:
            body["mode"] = mode
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if repetition_penalty is not None:
            body["repetition_penalty"] = repetition_penalty

        response_data = await self._client.post("/v1/vision/analyze/async", json=body)
        return CreateVisionAsyncResponse.model_validate(response_data)

    async def get_result(self, *, job_id: str) -> GetVisionResultResponse:
        """Get the result of an async vision job."""
        response_data = await self._client.get(f"/v1/vision/analyze/async/{job_id}")
        return GetVisionResultResponse.model_validate(response_data)


class AsyncVisionChat:
    """Async vision chat endpoint with streaming support."""

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        warmup_helper: Optional["AsyncWarmupHelper"] = None,
    ) -> None:
        self._client = http_client
        self._warmup_helper = warmup_helper

    async def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model_id: str,
        stream: bool = True,
        conversation_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        wait_for_ready: bool = False,
        warmup_timeout: Optional[float] = None,
    ) -> Union[CreateVisionChatResponse, AsyncStream[CreateVisionChatResponse]]:
        """Chat with images asynchronously.

        Args:
            messages: List of vision chat messages.
            model_id: Vision model ID.
            stream: Whether to stream the response.
            conversation_id: Conversation ID for multi-turn.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            top_p: Top-p sampling.
            top_k: Top-k sampling.
            repetition_penalty: Repetition penalty.
            wait_for_ready: If True, wait for the model to be ready.
            warmup_timeout: Maximum seconds to wait for warmup.

        Returns:
            Stream or CreateVisionChatResponse.

        Raises:
            WarmupTimeoutError: If wait_for_ready is True and the model
                doesn't become ready within the timeout period.
        """
        # Handle cold start waiting if enabled
        if wait_for_ready and self._warmup_helper:
            await self._warmup_helper.wait_for_ready(model_id, timeout=warmup_timeout)

        body: Dict[str, Any] = {
            "messages": messages,
            "model_id": model_id,
            "stream": stream,
        }

        if conversation_id is not None:
            body["conversation_id"] = conversation_id
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if repetition_penalty is not None:
            body["repetition_penalty"] = repetition_penalty

        if stream:
            response = await self._client.request(
                "POST", "/v1/vision/chat", json=body, stream=True
            )
            return AsyncStream(response=response, cast_to=CreateVisionChatResponse)

        response_data = await self._client.post("/v1/vision/chat", json=body)
        return CreateVisionChatResponse.model_validate(response_data)


class AsyncVision:
    """Async vision resource."""

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        warmup_helper: Optional["AsyncWarmupHelper"] = None,
    ) -> None:
        self._client = http_client
        self._analyze = AsyncVisionAnalysis(http_client, warmup_helper)
        self._chat = AsyncVisionChat(http_client, warmup_helper)

    @property
    def analyze(self) -> AsyncVisionAnalysis:
        """Access async vision analysis endpoints."""
        return self._analyze

    @property
    def chat(self) -> AsyncVisionChat:
        """Access async vision chat endpoint."""
        return self._chat
