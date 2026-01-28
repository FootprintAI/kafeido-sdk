"""Vision types."""

from typing import List, Literal, Optional

from pydantic import BaseModel


class VisionImageSource(BaseModel):
    """Image source for vision requests."""

    storage_key: Optional[str] = None
    base64: Optional[str] = None
    url: Optional[str] = None


class VisionChatMessage(BaseModel):
    """A message in a vision chat conversation."""

    role: Literal["user", "assistant", "system"]
    content: Optional[str] = None
    images: Optional[List[VisionImageSource]] = None


class VisionUsage(BaseModel):
    """Token usage for vision requests."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class CreateVisionResponse(BaseModel):
    """Response from a sync vision analysis."""

    text: str
    usage: Optional[VisionUsage] = None
    error: Optional[str] = None


class CreateVisionChatResponse(BaseModel):
    """A chunk from a vision chat stream."""

    id: Optional[str] = None
    object: Optional[str] = None
    created: Optional[int] = None
    model: Optional[str] = None
    text: Optional[str] = None
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
    progress: Optional[float] = None
    result: Optional[CreateVisionResponse] = None
    error: Optional[str] = None
