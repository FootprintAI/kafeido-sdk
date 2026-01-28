"""Main Kafeido SDK client - OpenAI compatible."""

import os
from typing import Optional

from kafeido._auth import get_api_key
from kafeido._http_client import HTTPClient
from kafeido.resources.chat import Chat
from kafeido.resources.audio import Audio
from kafeido.resources.models import Models
from kafeido.resources.files import Files
from kafeido.resources.ocr import OCR
from kafeido.resources.vision import Vision
from kafeido.resources.jobs import Jobs
from kafeido.types.health import HealthResponse


class OpenAI:
    """Kafeido API client - OpenAI compatible.

    This client provides access to Kafeido's AI inference API with an
    OpenAI-compatible interface.

    Example:
        >>> from kafeido import OpenAI
        >>> client = OpenAI(api_key="sk-...")
        >>>
        >>> # Chat completion
        >>> response = client.chat.completions.create(
        ...     model="gpt-oss-20b",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(response.choices[0].message.content)
        >>>
        >>> # Audio transcription
        >>> with open("audio.mp3", "rb") as f:
        ...     transcript = client.audio.transcriptions.create(
        ...         file=f,
        ...         model="whisper-large-v3"
        ...     )
        >>> print(transcript.text)
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        """Initialize the Kafeido/OpenAI client.

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

        # Create HTTP client
        self._http_client = HTTPClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Initialize resources
        self._chat = Chat(self._http_client)
        self._audio = Audio(self._http_client)
        self._models = Models(self._http_client)
        self._files = Files(self._http_client)
        self._ocr = OCR(self._http_client)
        self._vision = Vision(self._http_client)
        self._jobs = Jobs(self._http_client)

    @property
    def chat(self) -> Chat:
        """Access chat completions API.

        Returns:
            Chat resource for creating chat completions.

        Example:
            >>> response = client.chat.completions.create(
            ...     model="gpt-oss-20b",
            ...     messages=[{"role": "user", "content": "Hi"}]
            ... )
        """
        return self._chat

    @property
    def audio(self) -> Audio:
        """Access audio API (transcriptions and translations).

        Returns:
            Audio resource for transcription and translation.

        Example:
            >>> with open("audio.mp3", "rb") as f:
            ...     transcript = client.audio.transcriptions.create(
            ...         file=f,
            ...         model="whisper-large-v3"
            ...     )
        """
        return self._audio

    @property
    def models(self) -> Models:
        """Access models API.

        Returns:
            Models resource for listing and retrieving model information.

        Example:
            >>> models = client.models.list()
            >>> for model in models.data:
            ...     print(model.id)
        """
        return self._models

    @property
    def files(self) -> Files:
        """Access files API.

        Returns:
            Files resource for uploading and managing files.

        Example:
            >>> with open("audio.mp3", "rb") as f:
            ...     file_obj = client.files.create(file=f, purpose="assistants")
        """
        return self._files

    @property
    def ocr(self) -> OCR:
        """Access OCR API.

        Returns:
            OCR resource for text extraction from images.
        """
        return self._ocr

    @property
    def vision(self) -> Vision:
        """Access vision API.

        Returns:
            Vision resource for image analysis and chat.
        """
        return self._vision

    @property
    def jobs(self) -> Jobs:
        """Access jobs API.

        Returns:
            Jobs resource for tracking async job status and progress.
        """
        return self._jobs

    def health(self) -> HealthResponse:
        """Check the health of the API service.

        Returns:
            HealthResponse with status, version, and build info.
        """
        response_data = self._http_client.get("/v1/health")
        return HealthResponse.model_validate(response_data)

    def close(self) -> None:
        """Close the HTTP client and clean up resources."""
        self._http_client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes HTTP client."""
        self.close()
