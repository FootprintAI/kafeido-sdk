"""Kafeido Python SDK - OpenAI Compatible.

A Python client library for the Kafeido AI inference API, providing
OpenAI-compatible interfaces for LLM, ASR, and OCR models.

Example:
    >>> from kafeido import OpenAI
    >>> client = OpenAI(api_key="sk-...")
    >>>
    >>> # Chat completion
    >>> response = client.chat.completions.create(
    ...     model="gpt-oss-20b",
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )
    >>> print(response.choices[0].message.content)
"""

from kafeido.version import __version__
from kafeido.client import OpenAI
from kafeido.types import (
    # Errors
    OpenAIError,
    APIError,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    ConflictError,
    UnprocessableEntityError,
    RateLimitError,
    InternalServerError,
    # Chat
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    # Audio
    Transcription,
    Translation,
    # Models
    Model,
    ModelList,
    # Files
    FileObject,
    FileList,
    DeletedFile,
)

__all__ = [
    "__version__",
    "OpenAI",
    # Errors
    "OpenAIError",
    "APIError",
    "APIConnectionError",
    "APITimeoutError",
    "APIStatusError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "UnprocessableEntityError",
    "RateLimitError",
    "InternalServerError",
    # Chat
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionMessage",
    "ChatCompletionMessageParam",
    # Audio
    "Transcription",
    "Translation",
    # Models
    "Model",
    "ModelList",
    # Files
    "FileObject",
    "FileList",
    "DeletedFile",
]
