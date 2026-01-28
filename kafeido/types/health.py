"""Health check types."""

from typing import Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response from health check endpoint."""

    status: str
    version: Optional[str] = None
    build_time: Optional[str] = None
    commit: Optional[str] = None
