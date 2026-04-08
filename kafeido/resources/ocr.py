"""OCR resource."""

from typing import TYPE_CHECKING, Optional

from kafeido._http_client import HTTPClient
from kafeido.types.ocr import (
    CreateOCRAsyncResponse,
    CreateOCRResponse,
    GetOCRResultResponse,
)

if TYPE_CHECKING:
    from kafeido._warmup import WarmupHelper


class OCRExtractions:
    """OCR extraction endpoint."""

    def __init__(
        self,
        http_client: HTTPClient,
        warmup_helper: Optional["WarmupHelper"] = None,
    ) -> None:
        self._client = http_client
        self._warmup_helper = warmup_helper

    def create(
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
        wait_for_ready: bool = False,
        warmup_timeout: Optional[float] = None,
    ) -> CreateOCRResponse:
        """Extract text from an image.

        Args:
            model_id: OCR model ID (e.g., "deepseek-ocr", "paddle-ocr").
            file_id: ID of a previously uploaded file.
            storage_key: Storage key from upload service.
            mode: OCR mode - "markdown", "generic", "figure", or "grounding".
            resolution: Resolution - "auto", "tiny", "small", "base", or "large".
            language: Language hint (ISO-639-1 code).
            custom_prompt: Custom prompt for the OCR model.
            max_tokens: Maximum tokens in the response.
            wait_for_ready: If True, wait for the model to be ready before
                making the request.
            warmup_timeout: Maximum seconds to wait for model warmup.

        Returns:
            CreateOCRResponse with extracted text and optional regions.

        Raises:
            WarmupTimeoutError: If wait_for_ready is True and the model
                doesn't become ready within the timeout period.
        """
        # Handle cold start waiting if enabled
        if wait_for_ready and self._warmup_helper:
            self._warmup_helper.wait_for_ready(model_id, timeout=warmup_timeout)

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

        response_data = self._client.post("/v1/ocr/extract", json=body)
        return CreateOCRResponse.model_validate(response_data)

    def create_async(
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
        """Create an async OCR extraction job.

        Args:
            model_id: OCR model ID.
            file_id: ID of a previously uploaded file.
            storage_key: Storage key from upload service.
            mode: OCR mode.
            resolution: Resolution setting.
            language: Language hint.
            custom_prompt: Custom prompt.
            max_tokens: Maximum tokens.

        Returns:
            CreateOCRAsyncResponse with job_id for polling.
        """
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

        response_data = self._client.post("/v1/ocr/extract/async", json=body)
        return CreateOCRAsyncResponse.model_validate(response_data)

    def get_result(self, *, job_id: str) -> GetOCRResultResponse:
        """Get the result of an async OCR job.

        Args:
            job_id: The job ID from create_async().

        Returns:
            GetOCRResultResponse with status, progress, and result.
        """
        response_data = self._client.get(f"/v1/ocr/extract/async/{job_id}")
        return GetOCRResultResponse.model_validate(response_data)


class OCR:
    """OCR resource."""

    def __init__(
        self,
        http_client: HTTPClient,
        warmup_helper: Optional["WarmupHelper"] = None,
    ) -> None:
        self._client = http_client
        self._extractions = OCRExtractions(http_client, warmup_helper)

    @property
    def extractions(self) -> OCRExtractions:
        """Access OCR extraction endpoints."""
        return self._extractions
