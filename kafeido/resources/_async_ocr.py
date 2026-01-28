"""Async OCR resource."""

from typing import Optional

from kafeido._http_client import AsyncHTTPClient
from kafeido.types.ocr import (
    CreateOCRAsyncResponse,
    CreateOCRResponse,
    GetOCRResultResponse,
)


class AsyncOCRExtractions:
    """Async OCR extraction endpoint."""

    def __init__(self, http_client: AsyncHTTPClient) -> None:
        self._client = http_client

    async def create(
        self,
        *,
        model_id: str,
        file_id: Optional[str] = None,
        storage_key: Optional[str] = None,
        mode: Optional[str] = None,
        resolution: Optional[str] = None,
        language: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> CreateOCRResponse:
        """Extract text from an image asynchronously."""
        body = {"model_id": model_id}

        if file_id is not None:
            body["file_id"] = file_id
        if storage_key is not None:
            body["storage_key"] = storage_key
        if mode is not None:
            body["mode"] = mode
        if resolution is not None:
            body["resolution"] = resolution
        if language is not None:
            body["language"] = language
        if custom_prompt is not None:
            body["custom_prompt"] = custom_prompt
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        response_data = await self._client.post("/v1/ocr/extract", json=body)
        return CreateOCRResponse.model_validate(response_data)

    async def create_async(
        self,
        *,
        model_id: str,
        file_id: Optional[str] = None,
        storage_key: Optional[str] = None,
        mode: Optional[str] = None,
        resolution: Optional[str] = None,
        language: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> CreateOCRAsyncResponse:
        """Create an async OCR extraction job."""
        body = {"model_id": model_id}

        if file_id is not None:
            body["file_id"] = file_id
        if storage_key is not None:
            body["storage_key"] = storage_key
        if mode is not None:
            body["mode"] = mode
        if resolution is not None:
            body["resolution"] = resolution
        if language is not None:
            body["language"] = language
        if custom_prompt is not None:
            body["custom_prompt"] = custom_prompt
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        response_data = await self._client.post("/v1/ocr/extract/async", json=body)
        return CreateOCRAsyncResponse.model_validate(response_data)

    async def get_result(self, *, job_id: str) -> GetOCRResultResponse:
        """Get the result of an async OCR job."""
        response_data = await self._client.get(f"/v1/ocr/extract/async/{job_id}")
        return GetOCRResultResponse.model_validate(response_data)


class AsyncOCR:
    """Async OCR resource."""

    def __init__(self, http_client: AsyncHTTPClient) -> None:
        self._client = http_client
        self._extractions = AsyncOCRExtractions(http_client)

    @property
    def extractions(self) -> AsyncOCRExtractions:
        """Access async OCR extraction endpoints."""
        return self._extractions
