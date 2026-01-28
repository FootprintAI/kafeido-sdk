"""Async audio transcription and translation resources."""

from typing import BinaryIO, Literal, Optional, Union

from kafeido._http_client import AsyncHTTPClient
from kafeido.types.audio import (
    AsyncTranscriptionResponse,
    AsyncTranscriptionResult,
    Transcription,
    Translation,
)
from kafeido.types.tts import CreateSpeechAsyncResponse, GetSpeechResultResponse


# Type alias for file inputs
FileTypes = Union[BinaryIO, bytes]


class AsyncTranscriptions:
    """Async audio transcriptions endpoint."""

    def __init__(self, http_client: AsyncHTTPClient) -> None:
        """Initialize async transcriptions resource.

        Args:
            http_client: The async HTTP client to use for requests.
        """
        self._client = http_client

    async def create(
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
        """Transcribe audio to text asynchronously.

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
            >>> client = AsyncOpenAI(api_key="sk-...")
            >>> with open("audio.mp3", "rb") as f:
            ...     transcript = await client.audio.transcriptions.create(
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

        response_data = await self._client.post(
            "/v1/audio/transcriptions",
            data=data,
            files=files,
        )
        return Transcription.model_validate(response_data)

    async def create_async(
        self,
        *,
        storage_key: str,
        model: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "json",
        temperature: Optional[float] = None,
        timestamp_granularities: Optional[list[str]] = None,
    ) -> AsyncTranscriptionResponse:
        """Create an async transcription job."""
        body = {
            "storage_key": storage_key,
            "model": model,
            "response_format": response_format,
        }

        if language is not None:
            body["language"] = language
        if prompt is not None:
            body["prompt"] = prompt
        if temperature is not None:
            body["temperature"] = str(temperature)
        if timestamp_granularities is not None:
            body["timestamp_granularities"] = timestamp_granularities

        response_data = await self._client.post(
            "/v1/audio/transcriptions/async", json=body
        )
        return AsyncTranscriptionResponse.model_validate(response_data)

    async def get_result(self, *, job_id: str) -> AsyncTranscriptionResult:
        """Get the result of an async transcription job."""
        response_data = await self._client.get(
            f"/v1/audio/transcriptions/async/{job_id}"
        )
        return AsyncTranscriptionResult.model_validate(response_data)


class AsyncTranslations:
    """Async audio translations endpoint."""

    def __init__(self, http_client: AsyncHTTPClient) -> None:
        """Initialize async translations resource.

        Args:
            http_client: The async HTTP client to use for requests.
        """
        self._client = http_client

    async def create(
        self,
        *,
        file: FileTypes,
        model: str,
        prompt: Optional[str] = None,
        response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "json",
        temperature: Optional[float] = None,
    ) -> Translation:
        """Translate audio to English asynchronously.

        Args:
            file: The audio file to translate (file object or bytes).
            model: Model ID (e.g., "whisper-large-v3").
            prompt: Optional text to guide the model's style.
            response_format: Format of the response.
            temperature: Sampling temperature (0-1).

        Returns:
            Translation with English text.

        Example:
            >>> client = AsyncOpenAI(api_key="sk-...")
            >>> with open("audio_spanish.mp3", "rb") as f:
            ...     translation = await client.audio.translations.create(
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

        response_data = await self._client.post(
            "/v1/audio/translations",
            data=data,
            files=files,
        )
        return Translation.model_validate(response_data)


class AsyncSpeech:
    """Async text-to-speech endpoint."""

    def __init__(self, http_client: AsyncHTTPClient) -> None:
        self._client = http_client

    async def create(
        self,
        *,
        model: str,
        input: str,
        voice: str = "alloy",
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
        reference_audio_id: Optional[str] = None,
        reference_audio_key: Optional[str] = None,
        language: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> CreateSpeechAsyncResponse:
        """Create a text-to-speech job asynchronously."""
        body: dict = {
            "model": model,
            "input": input,
            "voice": voice,
        }

        if response_format is not None:
            body["response_format"] = response_format
        if speed is not None:
            body["speed"] = speed
        if reference_audio_id is not None:
            body["reference_audio_id"] = reference_audio_id
        if reference_audio_key is not None:
            body["reference_audio_key"] = reference_audio_key
        if language is not None:
            body["language"] = language
        if system_prompt is not None:
            body["system_prompt"] = system_prompt
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        response_data = await self._client.post("/v1/audio/speech", json=body)
        return CreateSpeechAsyncResponse.model_validate(response_data)

    async def get_result(self, *, job_id: str) -> GetSpeechResultResponse:
        """Get the result of a TTS job asynchronously."""
        response_data = await self._client.get(f"/v1/audio/speech/{job_id}")
        return GetSpeechResultResponse.model_validate(response_data)


class AsyncAudio:
    """Async audio resource."""

    def __init__(self, http_client: AsyncHTTPClient) -> None:
        """Initialize async audio resource.

        Args:
            http_client: The async HTTP client to use for requests.
        """
        self._client = http_client
        self._transcriptions = AsyncTranscriptions(http_client)
        self._translations = AsyncTranslations(http_client)
        self._speech = AsyncSpeech(http_client)

    @property
    def transcriptions(self) -> AsyncTranscriptions:
        """Access async audio transcriptions endpoint."""
        return self._transcriptions

    @property
    def translations(self) -> AsyncTranslations:
        """Access async audio translations endpoint."""
        return self._translations

    @property
    def speech(self) -> AsyncSpeech:
        """Access async text-to-speech endpoint."""
        return self._speech
