"""Tests for cold start waiting / warmup helpers."""

import pytest
import httpx
import respx
from unittest.mock import Mock, AsyncMock

from kafeido import OpenAI, AsyncOpenAI, WarmupTimeoutError
from kafeido._warmup import (
    WarmupHelper,
    AsyncWarmupHelper,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_MAX_WAIT_TIME,
    HEALTHY_STATUS,
)
from kafeido.types.models import ModelStatus, ModelStatusInfo, WarmupResponse


class TestWarmupHelper:
    """Tests for synchronous WarmupHelper."""

    def test_already_warm_returns_immediately(self):
        """When model is already warm, should return immediately."""
        warmup_fn = Mock(return_value=WarmupResponse(already_warm=True))
        status_fn = Mock()

        helper = WarmupHelper(status_fn, warmup_fn)
        helper.wait_for_ready("test-model")

        warmup_fn.assert_called_once_with("test-model")
        status_fn.assert_not_called()

    def test_polls_until_healthy(self):
        """Should poll until model becomes healthy."""
        warmup_fn = Mock(
            return_value=WarmupResponse(already_warm=False, estimated_seconds=10.0)
        )

        call_count = [0]

        def status_fn(model):
            call_count[0] += 1
            if call_count[0] >= 3:
                return ModelStatus(
                    model_id=model, status=ModelStatusInfo(status=HEALTHY_STATUS)
                )
            return ModelStatus(
                model_id=model, status=ModelStatusInfo(status="loading")
            )

        helper = WarmupHelper(
            status_fn, warmup_fn, poll_interval=0.01  # Fast for testing
        )
        helper.wait_for_ready("test-model")

        assert call_count[0] == 3

    def test_timeout_raises_error(self):
        """Should raise WarmupTimeoutError after max_wait_time."""
        warmup_fn = Mock(
            return_value=WarmupResponse(already_warm=False, estimated_seconds=60.0)
        )
        status_fn = Mock(
            return_value=ModelStatus(
                model_id="test", status=ModelStatusInfo(status="loading")
            )
        )

        helper = WarmupHelper(
            status_fn,
            warmup_fn,
            poll_interval=0.01,
            max_wait_time=0.05,
        )

        with pytest.raises(WarmupTimeoutError) as exc_info:
            helper.wait_for_ready("test-model")

        assert exc_info.value.model == "test-model"
        assert exc_info.value.waited_seconds >= 0.05

    def test_custom_timeout_override(self):
        """Should respect custom timeout parameter."""
        warmup_fn = Mock(return_value=WarmupResponse(already_warm=False))
        status_fn = Mock(
            return_value=ModelStatus(
                model_id="test", status=ModelStatusInfo(status="loading")
            )
        )

        helper = WarmupHelper(
            status_fn,
            warmup_fn,
            poll_interval=0.01,
            max_wait_time=10.0,  # Default is 10s
        )

        with pytest.raises(WarmupTimeoutError) as exc_info:
            helper.wait_for_ready("test-model", timeout=0.03)  # Override to 0.03s

        assert exc_info.value.waited_seconds < 0.1  # Should timeout quickly


class TestAsyncWarmupHelper:
    """Tests for asynchronous AsyncWarmupHelper."""

    @pytest.mark.asyncio
    async def test_already_warm_returns_immediately(self):
        """Async: When model is already warm, should return immediately."""
        warmup_fn = AsyncMock(return_value=WarmupResponse(already_warm=True))
        status_fn = AsyncMock()

        helper = AsyncWarmupHelper(status_fn, warmup_fn)
        await helper.wait_for_ready("test-model")

        warmup_fn.assert_called_once_with("test-model")
        status_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_polls_until_healthy(self):
        """Async: Should poll until model becomes healthy."""
        warmup_fn = AsyncMock(
            return_value=WarmupResponse(already_warm=False, estimated_seconds=10.0)
        )

        call_count = [0]

        async def status_fn(model):
            call_count[0] += 1
            if call_count[0] >= 3:
                return ModelStatus(
                    model_id=model, status=ModelStatusInfo(status=HEALTHY_STATUS)
                )
            return ModelStatus(
                model_id=model, status=ModelStatusInfo(status="loading")
            )

        helper = AsyncWarmupHelper(
            status_fn, warmup_fn, poll_interval=0.01  # Fast for testing
        )
        await helper.wait_for_ready("test-model")

        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_timeout_raises_error(self):
        """Async: Should raise WarmupTimeoutError after max_wait_time."""
        warmup_fn = AsyncMock(
            return_value=WarmupResponse(already_warm=False, estimated_seconds=60.0)
        )
        status_fn = AsyncMock(
            return_value=ModelStatus(
                model_id="test", status=ModelStatusInfo(status="loading")
            )
        )

        helper = AsyncWarmupHelper(
            status_fn,
            warmup_fn,
            poll_interval=0.01,
            max_wait_time=0.05,
        )

        with pytest.raises(WarmupTimeoutError) as exc_info:
            await helper.wait_for_ready("test-model")

        assert exc_info.value.model == "test-model"


class TestChatCompletionWithWaitForReady:
    """Integration tests for chat completion with wait_for_ready."""

    @respx.mock
    def test_chat_completion_with_warm_model(self, client, base_url, mock_chat_response):
        """Test chat completion with wait_for_ready when model is already warm."""
        # Mock warmup endpoint - model already warm
        respx.post(f"{base_url}/v1/models/warmup").mock(
            return_value=httpx.Response(200, json={"already_warm": True})
        )

        # Mock chat endpoint
        respx.post(f"{base_url}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_chat_response)
        )

        response = client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[{"role": "user", "content": "Hello!"}],
            wait_for_ready=True,
        )

        assert response.choices[0].message.content == "Hello! How can I help you today?"

    @respx.mock
    def test_chat_completion_warmup_polling(self, client, base_url, mock_chat_response):
        """Test that warmup polls until ready."""
        status_call_count = [0]

        def status_response(request):
            status_call_count[0] += 1
            if status_call_count[0] >= 2:
                return httpx.Response(
                    200,
                    json={"model_id": "gpt-oss-20b", "status": {"status": "healthy"}},
                )
            return httpx.Response(
                200,
                json={
                    "model_id": "gpt-oss-20b",
                    "status": {
                        "status": "loading",
                        "cold_start_progress": {"stage": "loading", "progress": 0.5},
                    },
                },
            )

        # Mock warmup - model not warm
        respx.post(f"{base_url}/v1/models/warmup").mock(
            return_value=httpx.Response(
                200, json={"already_warm": False, "estimated_seconds": 10.0}
            )
        )

        # Mock status endpoint with polling
        respx.get(f"{base_url}/v1/models/gpt-oss-20b/status").mock(
            side_effect=status_response
        )

        # Mock chat endpoint
        respx.post(f"{base_url}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_chat_response)
        )

        # Create a client with fast polling for testing
        from kafeido._warmup import WarmupHelper

        fast_client = OpenAI(api_key="sk-test123_dGVzdGtleQ==", base_url=base_url)
        # Replace warmup helper with fast polling version
        fast_client._warmup_helper = WarmupHelper(
            status_fn=fast_client._models.status,
            warmup_fn=lambda m: fast_client._models.warmup(model=m),
            poll_interval=0.01,  # Fast for testing
        )

        response = fast_client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[{"role": "user", "content": "Hello!"}],
            wait_for_ready=True,
        )

        assert status_call_count[0] >= 2
        assert response.choices[0].message.content is not None


class TestWarmupTimeoutErrorExport:
    """Test that WarmupTimeoutError is properly exported."""

    def test_warmup_timeout_error_exported(self):
        """WarmupTimeoutError should be importable from kafeido."""
        from kafeido import WarmupTimeoutError

        error = WarmupTimeoutError("test-model", 30.5)
        assert error.model == "test-model"
        assert error.waited_seconds == 30.5
        assert "test-model" in str(error)
        assert "30.5" in str(error)
