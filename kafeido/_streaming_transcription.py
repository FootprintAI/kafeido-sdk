"""WebSocket streaming transcription client."""

import json
from typing import Any, Dict, Iterator, Optional

from websockets.sync.client import connect as ws_connect

from kafeido.types.audio import StreamingTranscriptionResponse
from kafeido.types.errors import APIError


def _build_ws_url(base_url: str) -> str:
    """Convert an HTTP(S) base URL to a WebSocket URL with the streaming path.

    Args:
        base_url: The HTTP base URL (e.g., "https://api.kafeido.app").

    Returns:
        WebSocket URL (e.g., "wss://api.kafeido.app/v1/audio/transcriptions/stream").
    """
    url = base_url.rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url[len("https://"):]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://"):]
    return url + "/v1/audio/transcriptions/stream"


class StreamingTranscription:
    """Synchronous WebSocket client for real-time streaming transcription.

    Usage::

        stream = client.audio.transcriptions.stream(model="whisper-large-v3")
        with stream:
            stream.send(audio_bytes)
            for response in stream:
                for seg in response.segments:
                    print(seg.text)
    """

    def __init__(
        self,
        ws_url: str,
        api_key: str,
        config: Dict[str, Any],
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"}
        self._ws = ws_connect(ws_url, additional_headers=headers)
        # Send config as the first text frame
        self._ws.send(json.dumps(config))

    def send(self, audio_data: bytes) -> None:
        """Send a binary frame of audio data (float32 PCM, 16kHz, mono)."""
        self._ws.send(audio_data)

    def recv(self) -> StreamingTranscriptionResponse:
        """Receive and parse one JSON response from the server.

        Raises:
            APIError: If the server sends an error response.
        """
        raw = self._ws.recv()
        data = json.loads(raw)
        if "error" in data:
            raise APIError(message=data["error"])
        return StreamingTranscriptionResponse.model_validate(data)

    def __iter__(self) -> Iterator[StreamingTranscriptionResponse]:
        """Yield responses until the WebSocket connection closes."""
        try:
            while True:
                yield self.recv()
        except Exception:
            # Connection closed or error — stop iteration
            return

    def close(self) -> None:
        """Close the WebSocket connection."""
        self._ws.close()

    def __enter__(self) -> "StreamingTranscription":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class AsyncStreamingTranscription:
    """Asynchronous WebSocket client for real-time streaming transcription.

    Usage::

        stream = await client.audio.transcriptions.stream(model="whisper-large-v3")
        async with stream:
            await stream.send(audio_bytes)
            async for response in stream:
                for seg in response.segments:
                    print(seg.text)
    """

    def __init__(self, ws: Any, config: Dict[str, Any]) -> None:
        self._ws = ws
        self._config = config

    async def _send_config(self) -> None:
        """Send the config JSON as the first text frame."""
        await self._ws.send(json.dumps(self._config))

    async def send(self, audio_data: bytes) -> None:
        """Send a binary frame of audio data (float32 PCM, 16kHz, mono)."""
        await self._ws.send(audio_data)

    async def recv(self) -> StreamingTranscriptionResponse:
        """Receive and parse one JSON response from the server.

        Raises:
            APIError: If the server sends an error response.
        """
        raw = await self._ws.recv()
        data = json.loads(raw)
        if "error" in data:
            raise APIError(message=data["error"])
        return StreamingTranscriptionResponse.model_validate(data)

    async def __aiter__(self):
        """Yield responses until the WebSocket connection closes."""
        try:
            while True:
                yield await self.recv()
        except Exception:
            return

    async def close(self) -> None:
        """Close the WebSocket connection."""
        await self._ws.close()

    async def __aenter__(self) -> "AsyncStreamingTranscription":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
