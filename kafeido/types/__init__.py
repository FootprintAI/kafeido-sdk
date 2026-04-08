"""Type definitions for Kafeido SDK."""

from kafeido.types.enums import (
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
)
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
    TranscriptionWord,
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
    ColdStartProgress,
    WarmupResponse,
)
from kafeido.types.files import (
    FileObject,
    FileList,
    DeletedFile,
)
from kafeido.types.tts import (
    TTSUsage,
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
    VisionChatChoice,
    VisionChatDelta,
    CreateVisionResponse,
    CreateVisionChatResponse,
    CreateVisionAsyncResponse,
    GetVisionResultResponse,
)
from kafeido.types.jobs import (
    JobDetail,
    JobProgress,
    RequestProgress,
)
from kafeido.types.fine_tuning import (
    Quantization,
    FineTuningHyperparameters,
    FineTuningJob,
    FineTuningMetrics,
    FineTuningEvent,
    FineTuningJobList,
    FineTuningEventList,
)
from kafeido.types.health import (
    HealthResponse,
)

__all__ = [
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
    # Fine-tuning
    "Quantization",
    "FineTuningHyperparameters",
    "FineTuningJob",
    "FineTuningMetrics",
    "FineTuningEvent",
    "FineTuningJobList",
    "FineTuningEventList",
    # Health
    "HealthResponse",
]
