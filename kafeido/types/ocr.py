"""OCR types."""

from typing import List, Optional

from pydantic import BaseModel


class OCRRegion(BaseModel):
    """A detected text region with bounding box."""

    text: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: Optional[float] = None
    region_type: Optional[str] = None


class OCRUsage(BaseModel):
    """Token usage for OCR requests."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class CreateOCRResponse(BaseModel):
    """Sync OCR extraction response."""

    text: str
    regions: Optional[List[OCRRegion]] = None
    usage: Optional[OCRUsage] = None
    detected_language: Optional[str] = None


class OCRResult(BaseModel):
    """OCR result returned from async job polling."""

    text: str
    regions: Optional[List[OCRRegion]] = None
    usage: Optional[OCRUsage] = None
    detected_language: Optional[str] = None


class CreateOCRAsyncResponse(BaseModel):
    """Response from creating an async OCR job."""

    job_id: str
    status: str


class GetOCRResultResponse(BaseModel):
    """Response from polling an async OCR job."""

    status: str
    progress: Optional[float] = None
    result: Optional[OCRResult] = None
    error: Optional[str] = None
