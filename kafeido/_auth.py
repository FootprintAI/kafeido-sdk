"""Authentication handling for Kafeido SDK."""

import os
import re
from typing import Optional

from kafeido.types.errors import AuthenticationError


def get_api_key(api_key: Optional[str] = None) -> str:
    """Get API key from parameter or environment variables.

    Checks in order:
    1. Provided api_key parameter
    2. KAFEIDO_API_KEY environment variable
    3. OPENAI_API_KEY environment variable (for compatibility)

    Args:
        api_key: Optional API key to use directly.

    Returns:
        The API key to use.

    Raises:
        AuthenticationError: If no API key is found or format is invalid.
    """
    # Try provided key first
    if api_key:
        validate_api_key(api_key)
        return api_key

    # Try environment variables
    key = os.getenv("KAFEIDO_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not key:
        raise AuthenticationError(
            message=(
                "No API key provided. Set KAFEIDO_API_KEY or OPENAI_API_KEY "
                "environment variable, or pass api_key parameter to client."
            ),
            response=None,  # type: ignore
        )

    validate_api_key(key)
    return key


def validate_api_key(api_key: str) -> None:
    """Validate API key format.

    Expected format: sk-{6chars}_{base64}
    Example: sk-abc123_dGVzdGtleQ==

    Args:
        api_key: The API key to validate.

    Raises:
        AuthenticationError: If API key format is invalid.
    """
    if not isinstance(api_key, str):
        raise AuthenticationError(
            message="API key must be a string",
            response=None,  # type: ignore
        )

    if not api_key.startswith("sk-"):
        raise AuthenticationError(
            message="Invalid API key format. API key must start with 'sk-'",
            response=None,  # type: ignore
        )

    # Basic format check: sk-{prefix}_{key}
    if "_" not in api_key:
        raise AuthenticationError(
            message="Invalid API key format. Expected format: sk-{prefix}_{key}",
            response=None,  # type: ignore
        )

    parts = api_key.split("_", 1)
    if len(parts) != 2:
        raise AuthenticationError(
            message="Invalid API key format",
            response=None,  # type: ignore
        )

    prefix = parts[0]  # Should be "sk-{6chars}"
    if len(prefix) < 9:  # "sk-" + at least 6 chars
        raise AuthenticationError(
            message="Invalid API key format. Prefix too short.",
            response=None,  # type: ignore
        )


def create_auth_headers(api_key: str) -> dict:
    """Create authentication headers for API requests.

    Args:
        api_key: The API key to use for authentication.

    Returns:
        Dictionary with Authorization header.
    """
    return {
        "Authorization": f"Bearer {api_key}",
    }
