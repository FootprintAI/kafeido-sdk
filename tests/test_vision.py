"""Tests for vision resource."""

import pytest
import httpx
import respx

from kafeido import (
    CreateVisionResponse,
    CreateVisionAsyncResponse,
    GetVisionResultResponse,
)


@respx.mock
def test_vision_analyze(client, base_url):
    """Test sync vision analysis."""
    mock_response = {
        "text": "The image shows a cat sitting on a table.",
        "usage": {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250},
    }
    route = respx.post(f"{base_url}/v1/vision/analyze").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.vision.analyze.create(
        model_id="llama-3.2-vision-11b",
        image_url="https://example.com/cat.jpg",
        prompt="Describe this image",
    )

    assert isinstance(result, CreateVisionResponse)
    assert "cat" in result.text
    assert result.usage is not None
    assert result.usage.total_tokens == 250
    assert route.called


@respx.mock
def test_vision_analyze_with_base64(client, base_url):
    """Test vision analysis with base64 image."""
    mock_response = {"text": "A document with text."}
    route = respx.post(f"{base_url}/v1/vision/analyze").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.vision.analyze.create(
        model_id="llama-3.2-vision-11b",
        image_base64="iVBORw0KGgoAAAANS...",
        mode="document",
        max_tokens=500,
    )

    assert result.text == "A document with text."
    assert route.called


@respx.mock
def test_vision_analyze_async(client, base_url):
    """Test creating async vision analysis job."""
    mock_response = {"job_id": "vision-job-123", "status": "pending"}
    route = respx.post(f"{base_url}/v1/vision/analyze/async").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.vision.analyze.create_async(
        model_id="llama-3.2-vision-90b",
        storage_key="org_123/abc-image.jpg",
        prompt="Analyze this chart",
        mode="chart",
    )

    assert isinstance(result, CreateVisionAsyncResponse)
    assert result.job_id == "vision-job-123"
    assert route.called


@respx.mock
def test_vision_get_result(client, base_url):
    """Test getting async vision result."""
    mock_response = {
        "status": "completed",
        "progress": 100.0,
        "result": {"text": "Analysis complete: the chart shows growth."},
    }
    route = respx.get(f"{base_url}/v1/vision/analyze/async/vision-job-123").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.vision.analyze.get_result(job_id="vision-job-123")

    assert isinstance(result, GetVisionResultResponse)
    assert result.status == "completed"
    assert result.result is not None
    assert "growth" in result.result.text
    assert route.called
