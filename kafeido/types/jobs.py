"""Job and progress tracking types."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel


class JobDetail(BaseModel):
    """Full job detail from GET /v1/jobs/{job_id}."""

    id: str
    type: str
    status: Literal["pending", "processing", "completed", "failed"]
    created_at: Optional[int] = None
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ColdStartProgress(BaseModel):
    """Cold start progress for model loading."""

    stage: Optional[str] = None
    progress: Optional[float] = None
    estimated_seconds: Optional[float] = None


class RequestProgress(BaseModel):
    """Unified request progress combining warmup and job processing."""

    request_id: Optional[str] = None
    model_id: Optional[str] = None
    warmup_status: Optional[str] = None
    warmup_progress: Optional[float] = None
    cold_start: Optional[ColdStartProgress] = None
    job_id: Optional[str] = None
    job_status: Optional[str] = None
    job_progress: Optional[float] = None
    overall_progress: Optional[float] = None
    estimated_seconds: Optional[float] = None
