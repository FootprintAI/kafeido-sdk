"""Tests for jobs resource."""

import json

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
        "result": json.dumps({"text": "Hello world"}),
        "retry_count": 0,
        "queue_time_ms": 50,
        "processing_time_ms": 3200,
    }
    route = respx.get(f"{base_url}/v1/jobs/job-123").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.jobs.retrieve(job_id="job-123")

    assert isinstance(result, JobDetail)
    assert result.id == "job-123"
    assert result.status == "completed"
    assert json.loads(result.result) == {"text": "Hello world"}
    assert result.retry_count == 0
    assert result.queue_time_ms == 50
    assert result.processing_time_ms == 3200
    assert route.called


@respx.mock
def test_job_retrieve_failed(client, base_url):
    """Test retrieving a failed job."""
    mock_response = {
        "id": "job-456",
        "type": "ocr",
        "status": "failed",
        "error": "Image format not supported",
        "priority": "medium",
    }
    route = respx.get(f"{base_url}/v1/jobs/job-456").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.jobs.retrieve(job_id="job-456")

    assert result.status == "failed"
    assert result.error == "Image format not supported"
    assert result.priority == "medium"
    assert route.called


@respx.mock
def test_request_progress(client, base_url):
    """Test getting request progress."""
    mock_response = {
        "phase": "REQUEST_PHASE_PROCESSING",
        "warmup": {
            "stage": "COLD_START_STAGE_READY",
            "progress": 1.0,
            "estimated_seconds": 0,
            "message": "Model is ready",
        },
        "processing": {
            "job_id": "job-789",
            "progress": 0.65,
            "status": "processing",
            "queue_position": 0,
        },
        "overall_progress": 0.8,
        "estimated_seconds": 10,
        "message": "Processing audio file...",
    }
    route = respx.get(f"{base_url}/v1/requests/progress").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.jobs.progress(request_id="req-123")

    assert isinstance(result, RequestProgress)
    assert result.phase == "REQUEST_PHASE_PROCESSING"
    assert result.overall_progress == 0.8
    assert result.processing.progress == 0.65
    assert result.processing.job_id == "job-789"
    assert result.warmup.stage == "COLD_START_STAGE_READY"
    assert result.message == "Processing audio file..."
    assert route.called
