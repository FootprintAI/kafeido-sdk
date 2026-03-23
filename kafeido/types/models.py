"""Model types - OpenAI compatible."""

from typing import List, Literal, Optional

from pydantic import BaseModel


class ColdStartProgress(BaseModel):
    """Cold start progress info for a model."""

    stage: Optional[str] = None
    progress: Optional[float] = None
    estimated_seconds: Optional[int] = None
    message: Optional[str] = None


class ModelStatusInfo(BaseModel):
    """Runtime status of a model with additional metadata."""

    status: Optional[str] = None
    warm: Optional[bool] = None
    usage_percentage: Optional[int] = None
    cold_start_progress: Optional[ColdStartProgress] = None
    avg_tokens_per_second: Optional[float] = None


class Model(BaseModel):
    """Model information."""

    id: str
    object: Literal["model"] = "model"
    created: Optional[int] = None
    licence: Optional[str] = None
    input_format: Optional[str] = None
    output_format: Optional[str] = None
    family: Optional[str] = None
    model_status: Optional[ModelStatusInfo] = None
    available: Optional[bool] = None
    min_plan_tier: Optional[str] = None
    # Keep owned_by for backward compatibility
    owned_by: Optional[str] = None


class ModelList(BaseModel):
    """List of models."""

    object: Literal["list"] = "list"
    data: List[Model]


class ModelStatus(BaseModel):
    """Response from model status endpoint."""

    model_id: str
    status: Optional[ModelStatusInfo] = None


class WarmupResponse(BaseModel):
    """Response from model warmup endpoint."""

    already_warm: bool
    estimated_seconds: Optional[int] = None
