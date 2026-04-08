"""Tests for OCR resource."""

import pytest
import httpx
import respx

from kafeido import CreateOCRResponse, CreateOCRAsyncResponse, GetOCRResultResponse


@respx.mock
def test_ocr_create(client, base_url):
    """Test sync OCR extraction."""
    mock_response = {
        "text": "Hello World",
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    }
    route = respx.post(f"{base_url}/v1/ocr/extract").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.ocr.extractions.create(
        model_id="deepseek-ocr",
        file_id="file-123",
        mode="markdown",
    )

    assert isinstance(result, CreateOCRResponse)
    assert result.text == "Hello World"
    assert result.usage is not None
    assert result.usage.total_tokens == 110
    assert route.called


@respx.mock
def test_ocr_create_with_grounding(client, base_url):
    """Test OCR with grounding mode returning regions."""
    mock_response = {
        "text": "Invoice #123",
        "regions": [
            {
                "text": "Invoice #123",
                "x1": 0.1,
                "y1": 0.05,
                "x2": 0.5,
                "y2": 0.1,
                "confidence": 0.98,
                "region_type": "text",
            }
        ],
    }
    route = respx.post(f"{base_url}/v1/ocr/extract").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.ocr.extractions.create(
        model_id="deepseek-ocr",
        storage_key="org_123/abc-image.png",
        mode="grounding",
    )

    assert result.text == "Invoice #123"
    assert result.regions is not None
    assert len(result.regions) == 1
    assert result.regions[0].confidence == 0.98
    assert route.called


@respx.mock
def test_ocr_create_async(client, base_url):
    """Test creating async OCR job."""
    mock_response = {"job_id": "ocr-job-123", "status": "pending"}
    route = respx.post(f"{base_url}/v1/ocr/extract/async").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.ocr.extractions.create_async(
        model_id="deepseek-ocr",
        file_id="file-456",
    )

    assert isinstance(result, CreateOCRAsyncResponse)
    assert result.job_id == "ocr-job-123"
    assert route.called


@respx.mock
def test_ocr_get_result(client, base_url):
    """Test getting async OCR result."""
    mock_response = {
        "status": "completed",
        "progress": 100.0,
        "result": {"text": "Extracted text content"},
    }
    route = respx.get(f"{base_url}/v1/ocr/extract/async/ocr-job-123").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.ocr.extractions.get_result(job_id="ocr-job-123")

    assert isinstance(result, GetOCRResultResponse)
    assert result.status == "completed"
    assert result.result is not None
    assert result.result.text == "Extracted text content"
    assert route.called
