"""Audio transcription and translation resources."""

from typing import BinaryIO, Literal, Optional, Union

from kafeido._http_client import HTTPClient
from kafeido.types.audio import Transcription, Translation


# Type alias for file inputs
FileTypes = Union[BinaryIO, bytes]


class Transcriptions:
    """Audio transcriptions endpoint."""

    def __init__(self, http_client: HTTPClient) -> None:
        """Initialize transcriptions resource.

        Args:
            http_client: The HTTP client to use for requests.
        """
        self._client = http_client

    def create(
        self,
        *,
        file: FileTypes,
        model: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "json",
        temperature: Optional[float] = None,
        timestamp_granularities: Optional[list[str]] = None,
    ) -> Transcription:
        """Transcribe audio to text.

        Args:
            file: The audio file to transcribe (file object or bytes).
            model: Model ID (e.g., "whisper-large-v3", "whisper-turbo").
            language: Language of the audio (ISO-639-1 code).
            prompt: Optional text to guide the model's style.
            response_format: Format of the response.
            temperature: Sampling temperature (0-1).
            timestamp_granularities: Granularity of timestamps.

        Returns:
            Transcription with text and optional segments.

        Example:
            >>> client = OpenAI(api_key="sk-...")
            >>> with open("audio.mp3", "rb") as f:
            ...     transcript = client.audio.transcriptions.create(
            ...         file=f,
            ...         model="whisper-large-v3"
            ...     )
            >>> print(transcript.text)
        """
        # Prepare multipart upload
        files = {"file": file}
        data = {
            "model": model,
            "response_format": response_format,
        }

        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        if temperature is not None:
            data["temperature"] = str(temperature)
        if timestamp_granularities:
            data["timestamp_granularities[]"] = timestamp_granularities

        response_data = self._client.post(
            "/v1/audio/transcriptions",
            data=data,
            files=files,
        )
        return Transcription.model_validate(response_data)


class Translations:
    """Audio translations endpoint."""

    def __init__(self, http_client: HTTPClient) -> None:
        """Initialize translations resource.

        Args:
            http_client: The HTTP client to use for requests.
        """
        self._client = http_client

    def create(
        self,
        *,
        file: FileTypes,
        model: str,
        prompt: Optional[str] = None,
        response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "json",
        temperature: Optional[float] = None,
    ) -> Translation:
        """Translate audio to English.

        Args:
            file: The audio file to translate (file object or bytes).
            model: Model ID (e.g., "whisper-large-v3").
            prompt: Optional text to guide the model's style.
            response_format: Format of the response.
            temperature: Sampling temperature (0-1).

        Returns:
            Translation with English text.

        Example:
            >>> client = OpenAI(api_key="sk-...")
            >>> with open("audio_spanish.mp3", "rb") as f:
            ...     translation = client.audio.translations.create(
            ...         file=f,
            ...         model="whisper-large-v3"
            ...     )
            >>> print(translation.text)  # English translation
        """
        # Prepare multipart upload
        files = {"file": file}
        data = {
            "model": model,
            "response_format": response_format,
        }

        if prompt:
            data["prompt"] = prompt
        if temperature is not None:
            data["temperature"] = str(temperature)

        response_data = self._client.post(
            "/v1/audio/translations",
            data=data,
            files=files,
        )
        return Translation.model_validate(response_data)


class Audio:
    """Audio resource."""

    def __init__(self, http_client: HTTPClient) -> None:
        """Initialize audio resource.

        Args:
            http_client: The HTTP client to use for requests.
        """
        self._client = http_client
        self._transcriptions = Transcriptions(http_client)
        self._translations = Translations(http_client)

    @property
    def transcriptions(self) -> Transcriptions:
        """Access audio transcriptions endpoint."""
        return self._transcriptions

    @property
    def translations(self) -> Translations:
        """Access audio translations endpoint."""
        return self._translations
