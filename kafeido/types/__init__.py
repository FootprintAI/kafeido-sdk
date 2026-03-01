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
    AsyncTranscriptionResponse,
    AsyncTranscriptionResult,
    StreamingSegment,
    StreamingTranscriptionResponse,
)
from kafeido.types.models import (
    Model,
    ModelList,
    ModelStatus,
    ModelStatusInfo,
    WarmupResponse,
)
from kafeido.types.files import (
    FileObject,
    FileList,
    DeletedFile,
)
from kafeido.types.tts import (
    CreateSpeechAsyncResponse,
    SpeechResult,
    GetSpeechResultResponse,
)
from kafeido.types.ocr import (
    OCRRegion,
    OCRUsage,
    CreateOCRResponse,
    CreateOCRAsyncResponse,
    OCRResult,
    GetOCRResultResponse,
)
from kafeido.types.vision import (
    VisionImageSource,
    VisionChatMessage,
    VisionUsage,
    CreateVisionResponse,
    CreateVisionChatResponse,
    CreateVisionAsyncResponse,
    GetVisionResultResponse,
)
from kafeido.types.jobs import (
    JobDetail,
    ColdStartProgress,
    RequestProgress,
)
from kafeido.types.health import (
    HealthResponse,
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
    "AsyncTranscriptionResponse",
    "AsyncTranscriptionResult",
    "StreamingSegment",
    "StreamingTranscriptionResponse",
    # Models
    "Model",
    "ModelList",
    "ModelStatus",
    "ModelStatusInfo",
    "WarmupResponse",
    # Files
    "FileObject",
    "FileList",
    "DeletedFile",
    # TTS
    "CreateSpeechAsyncResponse",
    "SpeechResult",
    "GetSpeechResultResponse",
    # OCR
    "OCRRegion",
    "OCRUsage",
    "CreateOCRResponse",
    "CreateOCRAsyncResponse",
    "OCRResult",
    "GetOCRResultResponse",
    # Vision
    "VisionImageSource",
    "VisionChatMessage",
    "VisionUsage",
    "CreateVisionResponse",
    "CreateVisionChatResponse",
    "CreateVisionAsyncResponse",
    "GetVisionResultResponse",
    # Jobs
    "JobDetail",
    "ColdStartProgress",
    "RequestProgress",
    # Health
    "HealthResponse",
]
