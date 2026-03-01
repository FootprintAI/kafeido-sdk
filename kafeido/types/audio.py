"""Audio transcription and translation types - OpenAI compatible."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class TranscriptionSegment(BaseModel):
    """A segment of transcribed audio."""

    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class Transcription(BaseModel):
    """Audio transcription response."""

    text: str
    task: Optional[str] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[TranscriptionSegment]] = None
    words: Optional[List[Dict[str, Any]]] = None


class Translation(BaseModel):
    """Audio translation response."""

    text: str
    task: Optional[str] = "translate"
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[TranscriptionSegment]] = None


class AsyncTranscriptionResponse(BaseModel):
    """Response from creating an async transcription job."""

    job_id: str
    status: str
    estimated_completion_time: Optional[str] = None


class AsyncTranscriptionResult(BaseModel):
    """Response from polling an async transcription job."""

    status: str
    progress: Optional[float] = None
    result: Optional[Transcription] = None
    error: Optional[str] = None


class StreamingSegment(BaseModel):
    """A segment from real-time streaming transcription."""

    start: float
    end: float
    text: str
    completed: bool


class StreamingTranscriptionResponse(BaseModel):
    """A single response from the streaming transcription WebSocket."""

    segments: List[StreamingSegment] = []
    language: Optional[str] = None
    language_prob: Optional[float] = None
