"""Tests for jobs resource."""

import pytest
import httpx
import respx

from kafeido import JobDetail, RequestProgress


@respx.mock
def test_job_retrieve(client, base_url):
    """Test retrieving a job."""
    mock_response = {
        "id": "job-123",
        "type": "transcription",
        "status": "completed",
        "created_at": 1700000000,
        "completed_at": 1700000060,
        "result": {"text": "Hello world"},
    }
    route = respx.get(f"{base_url}/v1/jobs/job-123").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.jobs.retrieve(job_id="job-123")

    assert isinstance(result, JobDetail)
    assert result.id == "job-123"
    assert result.status == "completed"
    assert result.result == {"text": "Hello world"}
    assert route.called


@respx.mock
def test_job_retrieve_failed(client, base_url):
    """Test retrieving a failed job."""
    mock_response = {
        "id": "job-456",
        "type": "ocr",
        "status": "failed",
        "error": "Image format not supported",
    }
    route = respx.get(f"{base_url}/v1/jobs/job-456").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.jobs.retrieve(job_id="job-456")

    assert result.status == "failed"
    assert result.error == "Image format not supported"
    assert route.called


@respx.mock
def test_request_progress(client, base_url):
    """Test getting request progress."""
    mock_response = {
        "request_id": "req-123",
        "model_id": "whisper-large-v3",
        "warmup_status": "ready",
        "warmup_progress": 1.0,
        "job_id": "job-789",
        "job_status": "processing",
        "job_progress": 0.65,
        "overall_progress": 0.8,
        "estimated_seconds": 10.0,
    }
    route = respx.get(f"{base_url}/v1/requests/progress").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.jobs.progress(request_id="req-123")

    assert isinstance(result, RequestProgress)
    assert result.overall_progress == 0.8
    assert result.job_progress == 0.65
    assert route.called
