"""Type definitions for Kafeido SDK."""

from kafeido.types.errors import (
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
)
from kafeido.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatCompletionDelta,
    ChatCompletionChunkChoice,
)
from kafeido.types.audio import (
    Transcription,
    Translation,
    TranscriptionSegment,
)
from kafeido.types.models import (
    Model,
    ModelList,
)
from kafeido.types.files import (
    FileObject,
    FileList,
    DeletedFile,
)

__all__ = [
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
    "ChatCompletionChoice",
    "ChatCompletionUsage",
    "ChatCompletionDelta",
    "ChatCompletionChunkChoice",
    # Audio
    "Transcription",
    "Translation",
    "TranscriptionSegment",
    # Models
    "Model",
    "ModelList",
    # Files
    "FileObject",
    "FileList",
    "DeletedFile",
]
