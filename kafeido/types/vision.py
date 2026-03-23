"""Vision types."""

from typing import List, Optional

from pydantic import BaseModel


class VisionImageSource(BaseModel):
    """Image source for vision requests."""

    storage_key: Optional[str] = None
    image_base64: Optional[str] = None
    image_url: Optional[str] = None


class VisionChatMessage(BaseModel):
    """A message in a vision chat conversation."""

    role: str
    content: Optional[str] = None
    images: Optional[List[VisionImageSource]] = None


class VisionUsage(BaseModel):
    """Vision usage statistics."""

    prompt_tokens: Optional[int] = None
    vision_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    processing_time_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None


class CreateVisionResponse(BaseModel):
    """Response from a sync vision analysis."""

    text: str
    usage: Optional[VisionUsage] = None
    error: Optional[str] = None


class VisionChatChoice(BaseModel):
    """Vision chat choice (non-streaming)."""

    index: Optional[int] = None
    message: Optional[VisionChatMessage] = None
    finish_reason: Optional[str] = None


class VisionChatDelta(BaseModel):
    """Vision chat delta (streaming)."""

    role: Optional[str] = None
    content: Optional[str] = None


class CreateVisionChatResponse(BaseModel):
    """A response from vision chat (streaming or non-streaming)."""

    id: Optional[str] = None
    object: Optional[str] = None
    created: Optional[int] = None
    model: Optional[str] = None
    choice: Optional[VisionChatChoice] = None
    delta: Optional[VisionChatDelta] = None
    finish_reason: Optional[str] = None
    usage: Optional[VisionUsage] = None
    error: Optional[str] = None


class CreateVisionAsyncResponse(BaseModel):
    """Response from creating an async vision job."""

    job_id: str
    status: str


class GetVisionResultResponse(BaseModel):
    """Response from polling an async vision job."""

    status: str
    progress: Optional[int] = None
    result: Optional[CreateVisionResponse] = None
    error: Optional[str] = None
