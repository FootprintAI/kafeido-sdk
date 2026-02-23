"""Async audio transcription and translation resources."""

from typing import TYPE_CHECKING, BinaryIO, Literal, Optional, Union

from kafeido._http_client import AsyncHTTPClient
from kafeido.types.audio import (
    AsyncTranscriptionResponse,
    AsyncTranscriptionResult,
    Transcription,
    Translation,
)
from kafeido.types.tts import CreateSpeechAsyncResponse, GetSpeechResultResponse

if TYPE_CHECKING:
    from kafeido._warmup import AsyncWarmupHelper


# Type alias for file inputs
FileTypes = Union[BinaryIO, bytes]


class AsyncTranscriptions:
    """Async audio transcriptions endpoint."""

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        warmup_helper: Optional["AsyncWarmupHelper"] = None,
    ) -> None:
        """Initialize async transcriptions resource.

        Args:
            http_client: The async HTTP client to use for requests.
            warmup_helper: Optional warmup helper for cold start handling.
        """
        self._client = http_client
        self._warmup_helper = warmup_helper

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
        wait_for_ready: bool = False,
        warmup_timeout: Optional[float] = None,
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
            wait_for_ready: If True, wait for the model to be ready before
                making the request.
            warmup_timeout: Maximum seconds to wait for model warmup.

        Returns:
            Transcription with text and optional segments.

        Raises:
            WarmupTimeoutError: If wait_for_ready is True and the model
                doesn't become ready within the timeout period.

        Example:
            >>> client = AsyncOpenAI(api_key="sk-...")
            >>> with open("audio.mp3", "rb") as f:
            ...     transcript = await client.audio.transcriptions.create(
            ...         file=f,
            ...         model="whisper-large-v3",
            ...         wait_for_ready=True,
            ...     )
            >>> print(transcript.text)
        """
        # Handle cold start waiting if enabled
        if wait_for_ready and self._warmup_helper:
            await self._warmup_helper.wait_for_ready(model, timeout=warmup_timeout)

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

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        warmup_helper: Optional["AsyncWarmupHelper"] = None,
    ) -> None:
        """Initialize async translations resource.

        Args:
            http_client: The async HTTP client to use for requests.
            warmup_helper: Optional warmup helper for cold start handling.
        """
        self._client = http_client
        self._warmup_helper = warmup_helper

    async def create(
        self,
        *,
        file: FileTypes,
        model: str,
        prompt: Optional[str] = None,
        response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "json",
        temperature: Optional[float] = None,
        wait_for_ready: bool = False,
        warmup_timeout: Optional[float] = None,
    ) -> Translation:
        """Translate audio to English asynchronously.

        Args:
            file: The audio file to translate (file object or bytes).
            model: Model ID (e.g., "whisper-large-v3").
            prompt: Optional text to guide the model's style.
            response_format: Format of the response.
            temperature: Sampling temperature (0-1).
            wait_for_ready: If True, wait for the model to be ready before
                making the request.
            warmup_timeout: Maximum seconds to wait for model warmup.

        Returns:
            Translation with English text.

        Raises:
            WarmupTimeoutError: If wait_for_ready is True and the model
                doesn't become ready within the timeout period.

        Example:
            >>> client = AsyncOpenAI(api_key="sk-...")
            >>> with open("audio_spanish.mp3", "rb") as f:
            ...     translation = await client.audio.translations.create(
            ...         file=f,
            ...         model="whisper-large-v3",
            ...         wait_for_ready=True,
            ...     )
            >>> print(translation.text)  # English translation
        """
        # Handle cold start waiting if enabled
        if wait_for_ready and self._warmup_helper:
            await self._warmup_helper.wait_for_ready(model, timeout=warmup_timeout)

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
        wait_for_ready: bool = False,
        warmup_timeout: Optional[float] = None,
    ) -> CreateSpeechAsyncResponse:
        """Create a text-to-speech job asynchronously.

        Args:
            model: TTS model ID.
            input: Text to synthesize.
            voice: Voice preset.
            response_format: Audio format.
            speed: Speech speed.
            reference_audio_id: File ID for voice cloning.
            reference_audio_key: Storage key for voice cloning.
            language: Language code.
            system_prompt: Scene description.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            top_k: Top-k sampling.
            max_tokens: Maximum tokens.
            wait_for_ready: If True, wait for the model to be ready.
            warmup_timeout: Maximum seconds to wait for warmup.

        Returns:
            CreateSpeechAsyncResponse with job_id.

        Raises:
            WarmupTimeoutError: If wait_for_ready is True and the model
                doesn't become ready within the timeout period.
        """
        # Handle cold start waiting if enabled
        if wait_for_ready and self._warmup_helper:
            await self._warmup_helper.wait_for_ready(model, timeout=warmup_timeout)

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

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        warmup_helper: Optional["AsyncWarmupHelper"] = None,
    ) -> None:
        """Initialize async audio resource.

        Args:
            http_client: The async HTTP client to use for requests.
            warmup_helper: Optional warmup helper for cold start handling.
        """
        self._client = http_client
        self._transcriptions = AsyncTranscriptions(http_client, warmup_helper)
        self._translations = AsyncTranslations(http_client, warmup_helper)
        self._speech = AsyncSpeech(http_client, warmup_helper)

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
