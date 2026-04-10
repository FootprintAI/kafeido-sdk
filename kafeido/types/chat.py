"""Chat completion types - OpenAI compatible."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ChatCompletionMessage(BaseModel):
    """A chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionMessageParam(BaseModel):
    """Parameter for chat message input."""

    role: Literal["system", "user", "assistant"]
    content: str
    name: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    """A chat completion choice."""

    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""

    # Server returns camelCase via gRPC-gateway (promptTokens, completionTokens, totalTokens).
    # populate_by_name=True lets us accept both snake_case and camelCase inputs.
    model_config = {"populate_by_name": True}

    prompt_tokens: int = Field(..., alias="promptTokens")
    completion_tokens: int = Field(..., alias="completionTokens")
    total_tokens: int = Field(..., alias="totalTokens")


class ChatCompletion(BaseModel):
    """Chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None
    system_fingerprint: Optional[str] = None


# Streaming types
class ChatCompletionDelta(BaseModel):
    """Delta for streaming chat completion."""

    role: Optional[Literal["system", "user", "assistant"]] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionChunkChoice(BaseModel):
    """Choice in streaming chat completion chunk."""

    index: int
    delta: ChatCompletionDelta
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionChunk(BaseModel):
    """Streaming chat completion chunk."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
    system_fingerprint: Optional[str] = None
