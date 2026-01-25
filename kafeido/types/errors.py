"""Exception hierarchy for Kafeido SDK - OpenAI compatible."""

from typing import Any, Dict, Optional

import httpx


class OpenAIError(Exception):
    """Base exception for all Kafeido/OpenAI errors."""

    pass


class APIError(OpenAIError):
    """Base class for API-related errors."""

    def __init__(
        self,
        message: str,
        request: Optional[httpx.Request] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.request = request
        self.body = body

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r})"


class APIConnectionError(APIError):
    """Raised when an API request fails due to a network connectivity issue."""

    def __init__(
        self,
        message: str = "Connection error.",
        request: Optional[httpx.Request] = None,
    ) -> None:
        super().__init__(message=message, request=request)


class APITimeoutError(APIConnectionError):
    """Raised when an API request times out."""

    def __init__(
        self,
        message: str = "Request timed out.",
        request: Optional[httpx.Request] = None,
    ) -> None:
        super().__init__(message=message, request=request)


class APIStatusError(APIError):
    """Raised when an API response has a status code of 4xx or 5xx."""

    def __init__(
        self,
        message: str,
        response: httpx.Response,
        body: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, request=response.request, body=body)
        self.response = response
        self.status_code = response.status_code

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"status_code={self.status_code})"
        )


class BadRequestError(APIStatusError):
    """Raised when the API returns a 400 status code."""

    pass


class AuthenticationError(APIStatusError):
    """Raised when the API returns a 401 status code - invalid API key."""

    pass


class PermissionDeniedError(APIStatusError):
    """Raised when the API returns a 403 status code."""

    pass


class NotFoundError(APIStatusError):
    """Raised when the API returns a 404 status code."""

    pass


class ConflictError(APIStatusError):
    """Raised when the API returns a 409 status code."""

    pass


class UnprocessableEntityError(APIStatusError):
    """Raised when the API returns a 422 status code."""

    pass


class RateLimitError(APIStatusError):
    """Raised when the API returns a 429 status code - rate limit exceeded."""

    pass


class InternalServerError(APIStatusError):
    """Raised when the API returns a 5xx status code."""

    pass


def error_from_response(
    response: httpx.Response,
    message: Optional[str] = None,
) -> APIStatusError:
    """Create the appropriate exception from an HTTP response.

    Args:
        response: The HTTP response object.
        message: Optional custom error message.

    Returns:
        The appropriate APIStatusError subclass based on status code.
    """
    status_code = response.status_code

    # Try to parse error body
    try:
        body = response.json()
        if not message and isinstance(body, dict):
            # Extract error message from response body
            error_data = body.get("error", {})
            if isinstance(error_data, dict):
                message = error_data.get("message", "")
            elif isinstance(error_data, str):
                message = error_data
    except Exception:
        body = None

    if not message:
        message = f"Error code: {status_code}"

    # Map status codes to exception classes
    if status_code == 400:
        return BadRequestError(message, response, body)
    elif status_code == 401:
        return AuthenticationError(message, response, body)
    elif status_code == 403:
        return PermissionDeniedError(message, response, body)
    elif status_code == 404:
        return NotFoundError(message, response, body)
    elif status_code == 409:
        return ConflictError(message, response, body)
    elif status_code == 422:
        return UnprocessableEntityError(message, response, body)
    elif status_code == 429:
        return RateLimitError(message, response, body)
    elif status_code >= 500:
        return InternalServerError(message, response, body)
    else:
        # Generic API status error for other 4xx codes
        return APIStatusError(message, response, body)
