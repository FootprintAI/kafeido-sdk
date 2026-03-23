"""Kafeido Python SDK - OpenAI Compatible.

A Python client library for the Kafeido AI inference API, providing
OpenAI-compatible interfaces for LLM, ASR, OCR, TTS, and Vision models.

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
from kafeido._async_client import AsyncOpenAI
from kafeido._warmup import WarmupTimeoutError
from kafeido.types import (
    # Enums
    ModelId,
    ModelFamily,
    ModelFormat,
    ModelStatusEnum,
    JobStatus,
    ColdStartStage,
    RequestPhase,
    PlanTier,
    OCRMode,
    OCRResolution,
    TTSLanguage,
    VisionAnalysisMode,
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
    TranscriptionWord,
    AsyncTranscriptionResponse,
    AsyncTranscriptionResult,
    StreamingSegment,
    StreamingTranscriptionResponse,
    # Models
    Model,
    ModelList,
    ModelStatus,
    ModelStatusInfo,
    ColdStartProgress,
    WarmupResponse,
    # Files
    FileObject,
    FileList,
    DeletedFile,
    # TTS
    TTSUsage,
    CreateSpeechAsyncResponse,
    SpeechResult,
    GetSpeechResultResponse,
    # OCR
    OCRRegion,
    OCRUsage,
    CreateOCRResponse,
    CreateOCRAsyncResponse,
    OCRResult,
    GetOCRResultResponse,
    # Vision
    VisionImageSource,
    VisionChatMessage,
    VisionUsage,
    VisionChatChoice,
    VisionChatDelta,
    CreateVisionResponse,
    CreateVisionChatResponse,
    CreateVisionAsyncResponse,
    GetVisionResultResponse,
    # Jobs
    JobDetail,
    JobProgress,
    RequestProgress,
    # Health
    HealthResponse,
)

__all__ = [
    "__version__",
    "OpenAI",
    "AsyncOpenAI",
    # Enums
    "ModelId",
    "ModelFamily",
    "ModelFormat",
    "ModelStatusEnum",
    "JobStatus",
    "ColdStartStage",
    "RequestPhase",
    "PlanTier",
    "OCRMode",
    "OCRResolution",
    "TTSLanguage",
    "VisionAnalysisMode",
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
    "WarmupTimeoutError",
    # Chat
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionMessage",
    "ChatCompletionMessageParam",
    # Audio
    "Transcription",
    "Translation",
    "TranscriptionWord",
    "AsyncTranscriptionResponse",
    "AsyncTranscriptionResult",
    "StreamingSegment",
    "StreamingTranscriptionResponse",
    # Models
    "Model",
    "ModelList",
    "ModelStatus",
    "ModelStatusInfo",
    "ColdStartProgress",
    "WarmupResponse",
    # Files
    "FileObject",
    "FileList",
    "DeletedFile",
    # TTS
    "TTSUsage",
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
    "VisionChatChoice",
    "VisionChatDelta",
    "CreateVisionResponse",
    "CreateVisionChatResponse",
    "CreateVisionAsyncResponse",
    "GetVisionResultResponse",
    # Jobs
    "JobDetail",
    "JobProgress",
    "RequestProgress",
    # Health
    "HealthResponse",
]
