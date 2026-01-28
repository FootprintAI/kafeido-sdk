"""Audio transcription and translation types - OpenAI compatible."""

from typing import Any, Dict, List, Literal, Optional

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
