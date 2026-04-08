"""Text-to-speech types."""

from typing import Optional

from pydantic import BaseModel


class TTSUsage(BaseModel):
    """TTS usage statistics."""

    character_count: Optional[int] = None
    audio_duration_seconds: Optional[float] = None
    processing_time_ms: Optional[float] = None
    real_time_factor: Optional[float] = None


class SpeechResult(BaseModel):
    """TTS result with download URL and audio metadata."""

    url: Optional[str] = None
    expires_at: Optional[int] = None
    duration_seconds: Optional[float] = None
    content_type: Optional[str] = None
    file_size: Optional[int] = None
    sample_rate: Optional[int] = None
    usage: Optional[TTSUsage] = None
    # Keep download_url for backward compatibility
    download_url: Optional[str] = None
    duration: Optional[float] = None


class CreateSpeechAsyncResponse(BaseModel):
    """Response from creating an async TTS job."""

    job_id: str
    status: str


class GetSpeechResultResponse(BaseModel):
    """Response from polling a TTS job."""

    status: str
    progress: Optional[int] = None
    result: Optional[SpeechResult] = None
    error: Optional[str] = None
