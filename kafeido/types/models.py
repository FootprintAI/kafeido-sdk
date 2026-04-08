"""Model types - OpenAI compatible."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ColdStartProgress(BaseModel):
    """Cold start progress info for a model."""

    model_config = {"populate_by_name": True}

    stage: Optional[str] = None
    progress: Optional[float] = None
    estimated_seconds: Optional[int] = Field(None, alias="estimatedSeconds")
    message: Optional[str] = None


class ModelStatusInfo(BaseModel):
    """Runtime status of a model with additional metadata."""

    model_config = {"populate_by_name": True}

    status: Optional[str] = None
    warm: Optional[bool] = None
    usage_percentage: Optional[int] = Field(None, alias="usagePercentage")
    cold_start_progress: Optional[ColdStartProgress] = Field(None, alias="coldStartProgress")
    avg_tokens_per_second: Optional[float] = Field(None, alias="avgTokensPerSecond")


class Model(BaseModel):
    """Model information."""

    model_config = {"populate_by_name": True}

    id: str
    object: Literal["model"] = "model"
    created: Optional[int] = None
    licence: Optional[str] = None
    input_format: Optional[str] = Field(None, alias="inputFormat")
    output_format: Optional[str] = Field(None, alias="outputFormat")
    family: Optional[str] = None
    model_status: Optional[ModelStatusInfo] = Field(None, alias="modelStatus")
    available: Optional[bool] = None
    min_plan_tier: Optional[str] = Field(None, alias="minPlanTier")
    owned_by: Optional[str] = Field(None, alias="ownedBy")


class ModelList(BaseModel):
    """List of models."""

    object: Literal["list"] = "list"
    data: List[Model]


class ModelStatus(BaseModel):
    """Response from model status endpoint."""

    model_config = {"populate_by_name": True}

    model_id: str = Field(..., alias="modelId")
    status: Optional[ModelStatusInfo] = None


class WarmupResponse(BaseModel):
    """Response from model warmup endpoint."""

    model_config = {"populate_by_name": True}

    already_warm: bool = Field(False, alias="alreadyWarm")
    estimated_seconds: Optional[int] = Field(None, alias="estimatedSeconds")
