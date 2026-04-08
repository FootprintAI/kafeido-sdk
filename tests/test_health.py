"""Tests for health endpoint."""

import pytest
import httpx
import respx

from kafeido import HealthResponse


@respx.mock
def test_health(client, base_url):
    """Test health check."""
    mock_response = {
        "status": "ok",
        "version": "1.4.0",
        "build_time": "2026-01-24T00:00:00Z",
        "commit": "abc123",
    }
    route = respx.get(f"{base_url}/v1/health").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.health()

    assert isinstance(result, HealthResponse)
    assert result.status == "ok"
    assert result.version == "1.4.0"
    assert route.called
