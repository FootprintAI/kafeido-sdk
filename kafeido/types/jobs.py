"""Job and progress tracking types."""

from typing import Optional

from pydantic import BaseModel


class JobDetail(BaseModel):
    """Full job detail from GET /v1/jobs/{job_id}."""

    id: str
    type: str
    status: str
    priority: Optional[str] = None
    created_at: Optional[int] = None
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    result: Optional[str] = None
    error: Optional[str] = None
    retry_count: Optional[int] = None
    queue_time_ms: Optional[int] = None
    processing_time_ms: Optional[int] = None


class ColdStartProgress(BaseModel):
    """Cold start progress for model loading."""

    stage: Optional[str] = None
    progress: Optional[float] = None
    estimated_seconds: Optional[int] = None
    message: Optional[str] = None


class JobProgress(BaseModel):
    """Job progress details (for processing phase)."""

    job_id: Optional[str] = None
    progress: Optional[float] = None
    status: Optional[str] = None
    queue_position: Optional[int] = None


class RequestProgress(BaseModel):
    """Unified request progress combining warmup and job processing."""

    phase: Optional[str] = None
    warmup: Optional[ColdStartProgress] = None
    processing: Optional[JobProgress] = None
    overall_progress: Optional[float] = None
    estimated_seconds: Optional[int] = None
    message: Optional[str] = None
