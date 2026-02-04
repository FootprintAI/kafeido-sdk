"""Model types - OpenAI compatible."""

from typing import List, Literal, Optional

from pydantic import BaseModel


class Model(BaseModel):
    """Model information."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "kafeido"


class ModelList(BaseModel):
    """List of models."""

    object: Literal["list"] = "list"
    data: List[Model]


class ColdStartProgress(BaseModel):
    """Cold start progress info for a model."""

    stage: Optional[str] = None
    progress: Optional[float] = None
    estimated_seconds: Optional[float] = None


class ModelStatusInfo(BaseModel):
    """Detailed model status information."""

    status: Optional[str] = None
    usage_percent: Optional[float] = None
    cold_start_progress: Optional[ColdStartProgress] = None


class ModelStatus(BaseModel):
    """Response from model status endpoint."""

    model_id: str
    status: Optional[ModelStatusInfo] = None


class WarmupResponse(BaseModel):
    """Response from model warmup endpoint."""

    already_warm: bool
    estimated_seconds: Optional[float] = None
