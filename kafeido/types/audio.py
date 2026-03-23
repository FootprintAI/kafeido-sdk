"""Audio transcription and translation types - OpenAI compatible."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class TranscriptionSegment(BaseModel):
    """A segment of transcribed audio."""

    id: Optional[int] = None
    seek: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None
    text: Optional[str] = None
    tokens: Optional[List[int]] = None
    temperature: Optional[float] = None
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None
    no_speech_prob: Optional[float] = None


class TranscriptionWord(BaseModel):
    """A word with timestamps from transcription."""

    word: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None


class Transcription(BaseModel):
    """Audio transcription response."""

    text: str
    task: Optional[str] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[TranscriptionSegment]] = None
    words: Optional[List[TranscriptionWord]] = None


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
    estimated_completion_time: Optional[int] = None


class AsyncTranscriptionResult(BaseModel):
    """Response from polling an async transcription job."""

    status: str
    result: Optional[Transcription] = None
    error: Optional[str] = None
    progress: Optional[int] = None


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
