"""Async Kafeido SDK client - OpenAI compatible."""

from __future__ import annotations

import os
from typing import Optional

from kafeido._auth import get_api_key
from kafeido._http_client import AsyncHTTPClient
from kafeido._warmup import AsyncWarmupHelper
from kafeido.resources._async_chat import AsyncChat
from kafeido.resources._async_audio import AsyncAudio
from kafeido.resources._async_models import AsyncModels
from kafeido.resources._async_files import AsyncFiles
from kafeido.resources._async_ocr import AsyncOCR
from kafeido.resources._async_vision import AsyncVision
from kafeido.resources._async_jobs import AsyncJobs
from kafeido.types.health import HealthResponse


class AsyncOpenAI:
    """Async Kafeido API client - OpenAI compatible.

    This client provides async access to Kafeido's AI inference API with an
    OpenAI-compatible interface.

    Example:
        >>> from kafeido import AsyncOpenAI
        >>> import asyncio
        >>>
        >>> async def main():
        ...     client = AsyncOpenAI(api_key="sk-...")
        ...     response = await client.chat.completions.create(
        ...         model="gpt-oss-20b",
        ...         messages=[{"role": "user", "content": "Hello!"}]
        ...     )
        ...     print(response.choices[0].message.content)
        ...
        >>> asyncio.run(main())
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        """Initialize the async Kafeido/OpenAI client.

        Args:
            api_key: API key for authentication. If not provided, will check
                KAFEIDO_API_KEY or OPENAI_API_KEY environment variables.
            base_url: Base URL for API requests. Defaults to https://api.kafeido.app.
            timeout: Request timeout in seconds. Default is 120 seconds.
            max_retries: Maximum number of retry attempts. Default is 2.

        Raises:
            AuthenticationError: If no valid API key is found.
        """
        # Get and validate API key
        self.api_key = get_api_key(api_key)

        # Set base URL
        self.base_url = (
            base_url
            or os.getenv("KAFEIDO_BASE_URL")
            or "https://api.kafeido.app"
        )

        # Create async HTTP client
        self._http_client = AsyncHTTPClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Initialize models resource first (needed for warmup helper)
        self._models = AsyncModels(self._http_client)

        # Initialize warmup helper for cold start handling
        self._warmup_helper = AsyncWarmupHelper(
            status_fn=self._models.status,
            warmup_fn=lambda m: self._models.warmup(model=m),
        )

        # Initialize resources with warmup helper
        self._chat = AsyncChat(self._http_client, self._warmup_helper)
        self._audio = AsyncAudio(self._http_client, self._warmup_helper)
        self._files = AsyncFiles(self._http_client)
        self._ocr = AsyncOCR(self._http_client, self._warmup_helper)
        self._vision = AsyncVision(self._http_client, self._warmup_helper)
        self._jobs = AsyncJobs(self._http_client)

    @property
    def chat(self) -> AsyncChat:
        """Access async chat completions API.

        Returns:
            AsyncChat resource for creating chat completions.

        Example:
            >>> response = await client.chat.completions.create(
            ...     model="gpt-oss-20b",
            ...     messages=[{"role": "user", "content": "Hi"}]
            ... )
        """
        return self._chat

    @property
    def audio(self) -> AsyncAudio:
        """Access async audio API.

        Returns:
            AsyncAudio resource for transcriptions and translations.

        Example:
            >>> with open("audio.mp3", "rb") as f:
            ...     transcript = await client.audio.transcriptions.create(
            ...         file=f,
            ...         model="whisper-large-v3"
            ...     )
        """
        return self._audio

    @property
    def models(self) -> AsyncModels:
        """Access async models API.

        Returns:
            AsyncModels resource for listing and retrieving models.

        Example:
            >>> models = await client.models.list()
            >>> print([m.id for m in models.data])
        """
        return self._models

    @property
    def files(self) -> AsyncFiles:
        """Access async files API.

        Returns:
            AsyncFiles resource for file operations.

        Example:
            >>> with open("audio.mp3", "rb") as f:
            ...     file_obj = await client.files.create(
            ...         file=f,
            ...         purpose="assistants"
            ...     )
        """
        return self._files

    @property
    def ocr(self) -> AsyncOCR:
        """Access async OCR API."""
        return self._ocr

    @property
    def vision(self) -> AsyncVision:
        """Access async vision API."""
        return self._vision

    @property
    def jobs(self) -> AsyncJobs:
        """Access async jobs API."""
        return self._jobs

    async def health(self) -> HealthResponse:
        """Check the health of the API service asynchronously."""
        response_data = await self._http_client.get("/v1/health")
        return HealthResponse.model_validate(response_data)

    async def close(self) -> None:
        """Close the async HTTP client.

        This should be called when you're done using the client to clean up
        resources.

        Example:
            >>> client = AsyncOpenAI(api_key="sk-...")
            >>> try:
            ...     response = await client.chat.completions.create(...)
            ... finally:
            ...     await client.close()
        """
        await self._http_client.close()

    async def __aenter__(self) -> AsyncOpenAI:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
