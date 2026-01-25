"""HTTP client with retry logic and error handling."""

import time
from typing import Any, Dict, Iterator, Mapping, Optional, Union

import httpx

from kafeido.types.errors import (
    APIConnectionError,
    APITimeoutError,
    error_from_response,
)


class HTTPClient:
    """Synchronous HTTP client for API requests."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        max_retries: int = 2,
        custom_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize HTTP client.

        Args:
            base_url: Base URL for API requests.
            api_key: API key for authentication.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            custom_headers: Optional custom headers to include in requests.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        # Build default headers
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "kafeido-python/0.1.0",
            "Content-Type": "application/json",
        }

        if custom_headers:
            self._headers.update(custom_headers)

        # Create httpx client
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], httpx.Response]:
        """Make an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: Request path (relative to base_url).
            json: JSON request body.
            data: Form data request body.
            files: Files to upload.
            params: Query parameters.
            headers: Additional headers.
            stream: Whether to return raw response for streaming.

        Returns:
            Parsed JSON response or raw httpx.Response if streaming.

        Raises:
            APIConnectionError: On connection errors.
            APITimeoutError: On timeout.
            APIStatusError: On 4xx/5xx responses.
        """
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"

        # Merge headers
        request_headers = self._headers.copy()
        if headers:
            request_headers.update(headers)

        # Remove Content-Type for multipart uploads
        if files:
            request_headers.pop("Content-Type", None)

        # Retry loop
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(
                    method=method,
                    url=url,
                    json=json,
                    data=data,
                    files=files,
                    params=params,
                    headers=request_headers,
                )

                # Check for errors
                if not response.is_success:
                    raise error_from_response(response)

                # Return raw response for streaming
                if stream:
                    return response

                # Parse JSON response
                return response.json()

            except httpx.TimeoutException as e:
                last_error = APITimeoutError(
                    message=f"Request timed out after {self.timeout}s",
                    request=e.request if hasattr(e, "request") else None,
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise last_error

            except httpx.ConnectError as e:
                last_error = APIConnectionError(
                    message=f"Connection failed: {str(e)}",
                    request=e.request if hasattr(e, "request") else None,
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise last_error

            except httpx.HTTPStatusError:
                # Don't retry on 4xx/5xx errors
                raise

        # Should not reach here, but raise last error if we do
        if last_error:
            raise last_error
        raise APIConnectionError(message="Request failed after retries")

    def get(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make a GET request."""
        return self.request("GET", path, params=params, headers=headers)  # type: ignore

    def post(
        self,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make a POST request."""
        return self.request(  # type: ignore
            "POST", path, json=json, data=data, files=files, headers=headers
        )

    def delete(
        self,
        path: str,
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make a DELETE request."""
        return self.request("DELETE", path, headers=headers)  # type: ignore

    def stream(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Iterator[bytes]:
        """Make a streaming request.

        Args:
            method: HTTP method.
            path: Request path.
            json: JSON request body.
            headers: Additional headers.

        Yields:
            Response bytes as they arrive.
        """
        response = self.request(
            method, path, json=json, headers=headers, stream=True
        )
        assert isinstance(response, httpx.Response), "Expected Response for streaming"

        try:
            for chunk in response.iter_bytes():
                yield chunk
        finally:
            response.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncHTTPClient:
    """Asynchronous HTTP client for API requests."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        max_retries: int = 2,
        custom_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize async HTTP client."""
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        # Build default headers
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "kafeido-python/0.1.0",
            "Content-Type": "application/json",
        }

        if custom_headers:
            self._headers.update(custom_headers)

        # Create async httpx client
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        )

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], httpx.Response]:
        """Make an async HTTP request with retry logic."""
        import asyncio

        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"

        # Merge headers
        request_headers = self._headers.copy()
        if headers:
            request_headers.update(headers)

        # Remove Content-Type for multipart uploads
        if files:
            request_headers.pop("Content-Type", None)

        # Retry loop
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.request(
                    method=method,
                    url=url,
                    json=json,
                    data=data,
                    files=files,
                    params=params,
                    headers=request_headers,
                )

                # Check for errors
                if not response.is_success:
                    raise error_from_response(response)

                # Return raw response for streaming
                if stream:
                    return response

                # Parse JSON response
                return response.json()

            except httpx.TimeoutException as e:
                last_error = APITimeoutError(
                    message=f"Request timed out after {self.timeout}s",
                    request=e.request if hasattr(e, "request") else None,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise last_error

            except httpx.ConnectError as e:
                last_error = APIConnectionError(
                    message=f"Connection failed: {str(e)}",
                    request=e.request if hasattr(e, "request") else None,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise last_error

            except httpx.HTTPStatusError:
                raise

        if last_error:
            raise last_error
        raise APIConnectionError(message="Request failed after retries")

    async def get(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make an async GET request."""
        return await self.request("GET", path, params=params, headers=headers)  # type: ignore

    async def post(
        self,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make an async POST request."""
        return await self.request(  # type: ignore
            "POST", path, json=json, data=data, files=files, headers=headers
        )

    async def delete(
        self,
        path: str,
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make an async DELETE request."""
        return await self.request("DELETE", path, headers=headers)  # type: ignore

    async def close(self) -> None:
        """Close the async HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
