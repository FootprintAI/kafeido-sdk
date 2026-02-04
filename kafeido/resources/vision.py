"""Vision resource."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from kafeido._http_client import HTTPClient
from kafeido._streaming import Stream
from kafeido.types.vision import (
    CreateVisionAsyncResponse,
    CreateVisionChatResponse,
    CreateVisionResponse,
    GetVisionResultResponse,
    VisionChatMessage,
)


class VisionAnalysis:
    """Vision analysis endpoint."""

    def __init__(self, http_client: HTTPClient) -> None:
        self._client = http_client

    def create(
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
    ) -> CreateVisionResponse:
        """Analyze an image.

        Args:
            model_id: Vision model ID (e.g., "llama-3.2-vision-11b").
            storage_key: Storage key of the image.
            image_base64: Base64-encoded image.
            image_url: URL of the image.
            prompt: Analysis prompt.
            mode: Analysis mode - "general", "document", "chart", "code", "detailed".
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            top_p: Top-p sampling.
            top_k: Top-k sampling.
            repetition_penalty: Repetition penalty.

        Returns:
            CreateVisionResponse with analysis text.
        """
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

        response_data = self._client.post("/v1/vision/analyze", json=body)
        return CreateVisionResponse.model_validate(response_data)

    def create_async(
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
        """Create an async vision analysis job.

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

        Returns:
            CreateVisionAsyncResponse with job_id for polling.
        """
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

        response_data = self._client.post("/v1/vision/analyze/async", json=body)
        return CreateVisionAsyncResponse.model_validate(response_data)

    def get_result(self, *, job_id: str) -> GetVisionResultResponse:
        """Get the result of an async vision job.

        Args:
            job_id: The job ID from create_async().

        Returns:
            GetVisionResultResponse with status, progress, and result.
        """
        response_data = self._client.get(f"/v1/vision/analyze/async/{job_id}")
        return GetVisionResultResponse.model_validate(response_data)


class VisionChat:
    """Vision chat endpoint with streaming support."""

    def __init__(self, http_client: HTTPClient) -> None:
        self._client = http_client

    def create(
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
    ) -> Union[CreateVisionChatResponse, Stream[CreateVisionChatResponse]]:
        """Chat with images.

        Args:
            messages: List of vision chat messages with text and optional images.
            model_id: Vision model ID.
            stream: Whether to stream the response.
            conversation_id: Optional conversation ID for multi-turn.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            top_p: Top-p sampling.
            top_k: Top-k sampling.
            repetition_penalty: Repetition penalty.

        Returns:
            Stream of CreateVisionChatResponse chunks if streaming,
            or a single CreateVisionChatResponse if not.
        """
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
            response = self._client.request(
                "POST", "/v1/vision/chat", json=body, stream=True
            )
            return Stream(response=response, cast_to=CreateVisionChatResponse)

        response_data = self._client.post("/v1/vision/chat", json=body)
        return CreateVisionChatResponse.model_validate(response_data)


class Vision:
    """Vision resource."""

    def __init__(self, http_client: HTTPClient) -> None:
        self._client = http_client
        self._analyze = VisionAnalysis(http_client)
        self._chat = VisionChat(http_client)

    @property
    def analyze(self) -> VisionAnalysis:
        """Access vision analysis endpoints."""
        return self._analyze

    @property
    def chat(self) -> VisionChat:
        """Access vision chat endpoint."""
        return self._chat
