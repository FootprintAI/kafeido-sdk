"""Text-to-speech types."""

from typing import Optional

from pydantic import BaseModel


class SpeechResult(BaseModel):
    """TTS result with download URL."""

    download_url: str
    duration: Optional[float] = None


class CreateSpeechAsyncResponse(BaseModel):
    """Response from creating an async TTS job."""

    job_id: str
    status: str


class GetSpeechResultResponse(BaseModel):
    """Response from polling a TTS job."""

    status: str
    progress: Optional[float] = None
    result: Optional[SpeechResult] = None
    error: Optional[str] = None
