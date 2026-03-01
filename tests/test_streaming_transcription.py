"""Tests for WebSocket streaming transcription."""

import json
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from kafeido.types.audio import StreamingSegment, StreamingTranscriptionResponse
from kafeido.types.errors import APIError
from kafeido._streaming_transcription import (
    StreamingTranscription,
    AsyncStreamingTranscription,
    _build_ws_url,
)


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------


def test_build_ws_url_https():
    assert _build_ws_url("https://api.kafeido.app") == (
        "wss://api.kafeido.app/v1/audio/transcriptions/stream"
    )


def test_build_ws_url_http():
    assert _build_ws_url("http://localhost:8080") == (
        "ws://localhost:8080/v1/audio/transcriptions/stream"
    )


def test_build_ws_url_trailing_slash():
    assert _build_ws_url("https://api.kafeido.app/") == (
        "wss://api.kafeido.app/v1/audio/transcriptions/stream"
    )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


def test_streaming_segment():
    seg = StreamingSegment(start=0.0, end=2.5, text="Hello", completed=True)
    assert seg.start == 0.0
    assert seg.end == 2.5
    assert seg.text == "Hello"
    assert seg.completed is True


def test_streaming_transcription_response():
    resp = StreamingTranscriptionResponse.model_validate(
        {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Hello", "completed": True},
                {"start": 2.5, "end": 5.0, "text": " world", "completed": False},
            ],
            "language": "en",
            "language_prob": 0.95,
        }
    )
    assert len(resp.segments) == 2
    assert resp.segments[0].text == "Hello"
    assert resp.segments[1].completed is False
    assert resp.language == "en"
    assert resp.language_prob == 0.95


def test_streaming_transcription_response_defaults():
    resp = StreamingTranscriptionResponse.model_validate({})
    assert resp.segments == []
    assert resp.language is None
    assert resp.language_prob is None


# ---------------------------------------------------------------------------
# Sync StreamingTranscription
# ---------------------------------------------------------------------------


class TestStreamingTranscription:
    """Tests for the synchronous StreamingTranscription client."""

    def _make_stream(self, ws_mock):
        """Create a StreamingTranscription with a mocked WebSocket."""
        with patch("kafeido._streaming_transcription.ws_connect", return_value=ws_mock):
            return StreamingTranscription(
                ws_url="wss://api.kafeido.app/v1/audio/transcriptions/stream",
                api_key="sk-test",
                config={"model": "whisper-large-v3", "language": "en"},
            )

    def test_config_sent_as_first_frame(self):
        ws = MagicMock()
        stream = self._make_stream(ws)
        ws.send.assert_called_once_with(
            json.dumps({"model": "whisper-large-v3", "language": "en"})
        )
        stream.close()

    def test_send_audio_binary(self):
        ws = MagicMock()
        stream = self._make_stream(ws)
        ws.send.reset_mock()

        audio = b"\x00" * 64000
        stream.send(audio)
        ws.send.assert_called_once_with(audio)
        stream.close()

    def test_recv_parses_response(self):
        ws = MagicMock()
        ws.recv.return_value = json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "Hi", "completed": True}
                ],
                "language": "en",
                "language_prob": 0.99,
            }
        )
        stream = self._make_stream(ws)
        resp = stream.recv()

        assert isinstance(resp, StreamingTranscriptionResponse)
        assert len(resp.segments) == 1
        assert resp.segments[0].text == "Hi"
        stream.close()

    def test_recv_error_raises_api_error(self):
        ws = MagicMock()
        ws.recv.return_value = json.dumps({"error": "model not found"})
        stream = self._make_stream(ws)

        with pytest.raises(APIError, match="model not found"):
            stream.recv()
        stream.close()

    def test_iter_yields_responses(self):
        responses = [
            json.dumps(
                {
                    "segments": [
                        {"start": 0.0, "end": 1.0, "text": "A", "completed": True}
                    ]
                }
            ),
            json.dumps(
                {
                    "segments": [
                        {"start": 1.0, "end": 2.0, "text": "B", "completed": True}
                    ]
                }
            ),
        ]
        ws = MagicMock()
        call_count = 0

        def recv_side_effect():
            nonlocal call_count
            if call_count < len(responses):
                result = responses[call_count]
                call_count += 1
                return result
            raise Exception("connection closed")

        ws.recv.side_effect = recv_side_effect
        stream = self._make_stream(ws)

        results = list(stream)
        assert len(results) == 2
        assert results[0].segments[0].text == "A"
        assert results[1].segments[0].text == "B"
        stream.close()

    def test_context_manager(self):
        ws = MagicMock()
        with self._make_stream(ws) as stream:
            pass
        ws.close.assert_called_once()

    def test_close(self):
        ws = MagicMock()
        stream = self._make_stream(ws)
        stream.close()
        ws.close.assert_called_once()


# ---------------------------------------------------------------------------
# Async StreamingTranscription
# ---------------------------------------------------------------------------


class TestAsyncStreamingTranscription:
    """Tests for the asynchronous AsyncStreamingTranscription client."""

    @pytest.mark.asyncio
    async def test_send_config(self):
        ws = AsyncMock()
        session = AsyncStreamingTranscription(
            ws=ws, config={"model": "whisper-large-v3"}
        )
        await session._send_config()
        ws.send.assert_called_once_with(
            json.dumps({"model": "whisper-large-v3"})
        )
        await session.close()

    @pytest.mark.asyncio
    async def test_send_audio(self):
        ws = AsyncMock()
        session = AsyncStreamingTranscription(
            ws=ws, config={"model": "whisper-large-v3"}
        )
        audio = b"\x00" * 64000
        await session.send(audio)
        ws.send.assert_called_with(audio)
        await session.close()

    @pytest.mark.asyncio
    async def test_recv_parses_response(self):
        ws = AsyncMock()
        ws.recv.return_value = json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "Hi", "completed": True}
                ],
                "language": "en",
                "language_prob": 0.99,
            }
        )
        session = AsyncStreamingTranscription(
            ws=ws, config={"model": "whisper-large-v3"}
        )
        resp = await session.recv()

        assert isinstance(resp, StreamingTranscriptionResponse)
        assert resp.segments[0].text == "Hi"
        await session.close()

    @pytest.mark.asyncio
    async def test_recv_error_raises_api_error(self):
        ws = AsyncMock()
        ws.recv.return_value = json.dumps({"error": "internal error"})
        session = AsyncStreamingTranscription(
            ws=ws, config={"model": "whisper-large-v3"}
        )

        with pytest.raises(APIError, match="internal error"):
            await session.recv()
        await session.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        ws = AsyncMock()
        session = AsyncStreamingTranscription(
            ws=ws, config={"model": "whisper-large-v3"}
        )
        async with session:
            pass
        ws.close.assert_called_once()
