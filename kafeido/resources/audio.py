"""Audio transcription and translation resources."""

from typing import TYPE_CHECKING, BinaryIO, Literal, Optional, Union

from kafeido._http_client import HTTPClient
from kafeido.types.audio import (
    AsyncTranscriptionResponse,
    AsyncTranscriptionResult,
    Transcription,
    Translation,
)
from kafeido.types.tts import CreateSpeechAsyncResponse, GetSpeechResultResponse

if TYPE_CHECKING:
    from kafeido._warmup import WarmupHelper


# Type alias for file inputs
FileTypes = Union[BinaryIO, bytes]


class Transcriptions:
    """Audio transcriptions endpoint."""

    def __init__(
        self,
        http_client: HTTPClient,
        warmup_helper: Optional["WarmupHelper"] = None,
    ) -> None:
        """Initialize transcriptions resource.

        Args:
            http_client: The HTTP client to use for requests.
            warmup_helper: Optional warmup helper for cold start handling.
        """
        self._client = http_client
        self._warmup_helper = warmup_helper

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
        wait_for_ready: bool = False,
        warmup_timeout: Optional[float] = None,
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
            wait_for_ready: If True, wait for the model to be ready before
                making the request.
            warmup_timeout: Maximum seconds to wait for model warmup.

        Returns:
            Transcription with text and optional segments.

        Raises:
            WarmupTimeoutError: If wait_for_ready is True and the model
                doesn't become ready within the timeout period.

        Example:
            >>> client = OpenAI(api_key="sk-...")
            >>> with open("audio.mp3", "rb") as f:
            ...     transcript = client.audio.transcriptions.create(
            ...         file=f,
            ...         model="whisper-large-v3",
            ...         wait_for_ready=True,
            ...     )
            >>> print(transcript.text)
        """
        # Handle cold start waiting if enabled
        if wait_for_ready and self._warmup_helper:
            self._warmup_helper.wait_for_ready(model, timeout=warmup_timeout)

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

    def create_async(
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
        """Create an async transcription job.

        Args:
            storage_key: Storage key of the uploaded audio file.
            model: Model ID (e.g., "whisper-large-v3").
            language: Language of the audio (ISO-639-1 code).
            prompt: Optional text to guide the model's style.
            response_format: Format of the response.
            temperature: Sampling temperature (0-1).
            timestamp_granularities: Granularity of timestamps.

        Returns:
            AsyncTranscriptionResponse with job_id for polling.
        """
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

        response_data = self._client.post(
            "/v1/audio/transcriptions/async", json=body
        )
        return AsyncTranscriptionResponse.model_validate(response_data)

    def get_result(self, *, job_id: str) -> AsyncTranscriptionResult:
        """Get the result of an async transcription job.

        Args:
            job_id: The job ID from create_async().

        Returns:
            AsyncTranscriptionResult with status, progress, and result.
        """
        response_data = self._client.get(
            f"/v1/audio/transcriptions/async/{job_id}"
        )
        return AsyncTranscriptionResult.model_validate(response_data)


class Translations:
    """Audio translations endpoint."""

    def __init__(
        self,
        http_client: HTTPClient,
        warmup_helper: Optional["WarmupHelper"] = None,
    ) -> None:
        """Initialize translations resource.

        Args:
            http_client: The HTTP client to use for requests.
            warmup_helper: Optional warmup helper for cold start handling.
        """
        self._client = http_client
        self._warmup_helper = warmup_helper

    def create(
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
        """Translate audio to English.

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
            >>> client = OpenAI(api_key="sk-...")
            >>> with open("audio_spanish.mp3", "rb") as f:
            ...     translation = client.audio.translations.create(
            ...         file=f,
            ...         model="whisper-large-v3",
            ...         wait_for_ready=True,
            ...     )
            >>> print(translation.text)  # English translation
        """
        # Handle cold start waiting if enabled
        if wait_for_ready and self._warmup_helper:
            self._warmup_helper.wait_for_ready(model, timeout=warmup_timeout)

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


class Speech:
    """Text-to-speech endpoint."""

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
        """Create a text-to-speech job.

        Args:
            model: TTS model ID (e.g., "qwen3-tts", "xtts-v2", "tts-1").
            input: Text to synthesize (max 4096 chars).
            voice: Voice preset (alloy, echo, fable, onyx, nova, shimmer).
            response_format: Audio format (wav, mp3, opus, flac, aac, pcm).
            speed: Speech speed (0.25-4.0).
            reference_audio_id: File ID for voice cloning.
            reference_audio_key: Storage key for voice cloning.
            language: Language code for synthesis.
            system_prompt: Scene description for the model.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            top_k: Top-k sampling.
            max_tokens: Maximum tokens.
            wait_for_ready: If True, wait for the model to be ready before
                making the request.
            warmup_timeout: Maximum seconds to wait for model warmup.

        Returns:
            CreateSpeechAsyncResponse with job_id for polling.

        Raises:
            WarmupTimeoutError: If wait_for_ready is True and the model
                doesn't become ready within the timeout period.
        """
        # Handle cold start waiting if enabled
        if wait_for_ready and self._warmup_helper:
            self._warmup_helper.wait_for_ready(model, timeout=warmup_timeout)

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

        response_data = self._client.post("/v1/audio/speech", json=body)
        return CreateSpeechAsyncResponse.model_validate(response_data)

    def get_result(self, *, job_id: str) -> GetSpeechResultResponse:
        """Get the result of a TTS job.

        Args:
            job_id: The job ID from create().

        Returns:
            GetSpeechResultResponse with status, progress, and download URL.
        """
        response_data = self._client.get(f"/v1/audio/speech/{job_id}")
        return GetSpeechResultResponse.model_validate(response_data)


class Audio:
    """Audio resource."""

    def __init__(
        self,
        http_client: HTTPClient,
        warmup_helper: Optional["WarmupHelper"] = None,
    ) -> None:
        """Initialize audio resource.

        Args:
            http_client: The HTTP client to use for requests.
            warmup_helper: Optional warmup helper for cold start handling.
        """
        self._client = http_client
        self._transcriptions = Transcriptions(http_client, warmup_helper)
        self._translations = Translations(http_client, warmup_helper)
        self._speech = Speech(http_client, warmup_helper)

    @property
    def transcriptions(self) -> Transcriptions:
        """Access audio transcriptions endpoint."""
        return self._transcriptions

    @property
    def translations(self) -> Translations:
        """Access audio translations endpoint."""
        return self._translations

    @property
    def speech(self) -> Speech:
        """Access text-to-speech endpoint."""
        return self._speech
