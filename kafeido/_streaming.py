"""Streaming support for Server-Sent Events (SSE)."""

import json
from typing import Any, AsyncIterator, Dict, Iterator, Optional, TypeVar

import httpx

T = TypeVar("T")


class Stream:
    """Synchronous stream for SSE responses."""

    def __init__(self, response: httpx.Response, cast_to: type):
        """Initialize stream.

        Args:
            response: The httpx Response object to stream from.
            cast_to: The type to cast parsed JSON to.
        """
        self.response = response
        self.cast_to = cast_to
        self._iterator: Optional[Iterator[T]] = None

    def __iter__(self) -> Iterator[T]:
        if self._iterator is None:
            self._iterator = self._stream()
        return self._iterator

    def __next__(self) -> T:
        if self._iterator is None:
            self._iterator = self._stream()
        return next(self._iterator)

    def _stream(self) -> Iterator[T]:
        """Parse SSE stream and yield objects.

        Yields:
            Parsed objects of type cast_to.
        """
        try:
            buffer = ""
            for chunk in self.response.iter_lines():
                # SSE format: "data: {...}"
                line = chunk.strip()

                if not line:
                    continue

                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix

                    # Check for stream end
                    if data == "[DONE]":
                        break

                    # Parse JSON and yield
                    try:
                        json_data = json.loads(data)
                        # Cast to target type if it has a model_validate method (Pydantic)
                        if hasattr(self.cast_to, "model_validate"):
                            try:
                                yield self.cast_to.model_validate(json_data)
                            except Exception:
                                # Skip validation errors (malformed data)
                                continue
                        # Otherwise just pass the dict
                        else:
                            yield json_data  # type: ignore
                    except json.JSONDecodeError:
                        # Skip malformed JSON
                        continue

        finally:
            self.response.close()

    def close(self) -> None:
        """Close the underlying response."""
        self.response.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncStream:
    """Asynchronous stream for SSE responses."""

    def __init__(self, response: httpx.Response, cast_to: type):
        """Initialize async stream.

        Args:
            response: The httpx Response object to stream from.
            cast_to: The type to cast parsed JSON to.
        """
        self.response = response
        self.cast_to = cast_to
        self._iterator: Optional[AsyncIterator[T]] = None

    def __aiter__(self) -> AsyncIterator[T]:
        if self._iterator is None:
            self._iterator = self._stream()
        return self._iterator

    async def __anext__(self) -> T:
        if self._iterator is None:
            self._iterator = self._stream()
        return await self._iterator.__anext__()

    async def _stream(self) -> AsyncIterator[T]:
        """Parse SSE stream and yield objects asynchronously.

        Yields:
            Parsed objects of type cast_to.
        """
        try:
            async for line in self.response.aiter_lines():
                line = line.strip()

                if not line:
                    continue

                if line.startswith("data: "):
                    data = line[6:]

                    if data == "[DONE]":
                        break

                    try:
                        json_data = json.loads(data)
                        if hasattr(self.cast_to, "model_validate"):
                            try:
                                yield self.cast_to.model_validate(json_data)
                            except Exception:
                                # Skip validation errors (malformed data)
                                continue
                        else:
                            yield json_data  # type: ignore
                    except json.JSONDecodeError:
                        continue

        finally:
            await self.response.aclose()

    async def close(self) -> None:
        """Close the underlying response."""
        await self.response.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
