# Kafeido Python SDK

[![PyPI version](https://badge.fury.io/py/kafeido.svg)](https://badge.fury.io/py/kafeido)
[![Python Support](https://img.shields.io/pypi/pyversions/kafeido.svg)](https://pypi.org/project/kafeido/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

The official Python SDK for [Kafeido](https://kafeido.app) - An OpenAI-compatible AI inference API providing access to LLM, ASR, and OCR models.

## Features

- **OpenAI Compatible**: Drop-in replacement for OpenAI Python SDK
- **Multiple AI Models**:
  - **LLM**: `gpt-oss-20b`, `gpt-oss-120b`
  - **ASR**: `whisper-large-v3`, `whisper-turbo`
  - **OCR**: `deepseek-ocr`, `paddle-ocr`
- **Streaming Support**: Real-time streaming for chat completions
- **Async Support**: Full async/await support for all endpoints
- **Type Safety**: Comprehensive type hints and Pydantic models
- **Robust Error Handling**: Detailed exception hierarchy

## Installation

```bash
pip install kafeido
```

### Optional Dependencies

```bash
# For async support with HTTP/2
pip install kafeido[async]

# For development
pip install kafeido[dev]
```

## Quick Start

```python
from kafeido import OpenAI

# Initialize client (API key from environment or parameter)
client = OpenAI(api_key="sk-...")

# Chat completion
response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]
)
print(response.choices[0].message.content)

# Audio transcription
with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        file=f,
        model="whisper-large-v3"
    )
    print(transcript.text)

# List available models
models = client.models.list()
for model in models.data:
    print(f"- {model.id}")
```

## Authentication

The SDK supports multiple ways to provide your API key:

### Environment Variables

```bash
export KAFEIDO_API_KEY="sk-..."
# or
export OPENAI_API_KEY="sk-..."  # For OpenAI compatibility
```

```python
from kafeido import OpenAI

client = OpenAI()  # Automatically uses environment variable
```

### Direct Parameter

```python
from kafeido import OpenAI

client = OpenAI(api_key="sk-...")
```

## Usage Examples

### Chat Completions

#### Basic Completion

```python
response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=100
)
print(response.choices[0].message.content)
```

#### Streaming Completion

```python
stream = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[{"role": "user", "content": "Write a poem about AI"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### With System Message

```python
response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a Python expert."},
        {"role": "user", "content": "How do I read a file in Python?"}
    ]
)
```

### Audio Transcription

#### Transcribe Audio File

```python
with open("meeting.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3",
        language="en",  # Optional: specify language
        response_format="verbose_json"  # Get detailed output
    )

print(f"Transcript: {transcript.text}")
print(f"Language: {transcript.language}")
print(f"Duration: {transcript.duration}s")

# Access segments with timestamps
if transcript.segments:
    for segment in transcript.segments:
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
```

#### Audio Translation

```python
# Translate any language to English
with open("audio_spanish.mp3", "rb") as audio_file:
    translation = client.audio.translations.create(
        file=audio_file,
        model="whisper-large-v3"
    )

print(translation.text)  # English translation
```

### Model Management

#### List All Models

```python
models = client.models.list()
for model in models.data:
    print(f"{model.id} (owned by: {model.owned_by})")
```

#### Get Model Details

```python
model = client.models.retrieve("gpt-oss-20b")
print(f"Model: {model.id}")
print(f"Created: {model.created}")
```

### File Management

#### Upload Audio File

```python
with open("large_audio.mp3", "rb") as f:
    file_obj = client.files.create(
        file=f,
        purpose="assistants"
    )

print(f"Uploaded: {file_obj.id}")
print(f"Size: {file_obj.bytes} bytes")
```

#### List Uploaded Files

```python
files = client.files.list()
for file in files.data:
    print(f"{file.filename} - {file.created_at}")
```

#### Delete File

```python
result = client.files.delete("file-123")
print(f"Deleted: {result.deleted}")
```

## Async Usage

All methods have async equivalents:

```python
import asyncio
from kafeido import AsyncOpenAI

async def main():
    async with AsyncOpenAI(api_key="sk-...") as client:
        # Chat completion
        response = await client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)

        # Streaming
        stream = await client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[{"role": "user", "content": "Count to 5"}],
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="")

asyncio.run(main())
```

## Error Handling

The SDK provides a comprehensive exception hierarchy:

```python
from kafeido import OpenAI, AuthenticationError, RateLimitError, APIError

client = OpenAI(api_key="sk-...")

try:
    response = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[{"role": "user", "content": "Hello"}]
    )
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except APIError as e:
    print(f"API error: {e}")
```

### Exception Types

- `OpenAIError` - Base exception for all errors
- `APIError` - Base for API-related errors
- `APIConnectionError` - Network connectivity issues
- `APITimeoutError` - Request timeout
- `APIStatusError` - HTTP 4xx/5xx responses
- `AuthenticationError` - Invalid API key (401)
- `PermissionDeniedError` - Insufficient permissions (403)
- `NotFoundError` - Resource not found (404)
- `RateLimitError` - Rate limit exceeded (429)
- `InternalServerError` - Server errors (5xx)

## Migration from OpenAI SDK

The Kafeido SDK is designed as a drop-in replacement for the OpenAI Python SDK:

```python
# Before (OpenAI)
from openai import OpenAI
client = OpenAI(api_key="...")

# After (Kafeido)
from kafeido import OpenAI
client = OpenAI(api_key="sk-...", base_url="https://api.kafeido.app")
```

All method signatures and response types are compatible.

## Configuration

### Base URL

```python
# Production
client = OpenAI(
    api_key="sk-...",
    base_url="https://api.kafeido.app"
)

# Development/Self-hosted
client = OpenAI(
    api_key="sk-...",
    base_url="http://localhost:8080"
)
```

### Timeouts and Retries

```python
client = OpenAI(
    api_key="sk-...",
    timeout=60.0,  # 60 seconds (default: 120)
    max_retries=3   # Max retry attempts (default: 2)
)
```

## Supported Models

### Large Language Models (LLM)

- `gpt-oss-20b` - 20B parameter model (recommended)
- `gpt-oss-120b` - 120B parameter model (high performance)

### Automatic Speech Recognition (ASR)

- `whisper-large-v3` - Latest Whisper model (recommended)
- `whisper-turbo` - Faster inference

### Optical Character Recognition (OCR)

- `deepseek-ocr` - DeepSeek OCR model (recommended)
- `paddle-ocr` - PaddleOCR model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Links

- **Homepage**: https://kafeido.app
- **Documentation**: https://docs.kafeido.app
- **API Reference**: https://docs.kafeido.app/api
- **GitHub**: https://github.com/footprintai/kafeido-sdk
- **PyPI**: https://pypi.org/project/kafeido

## Support

For issues and questions:
- GitHub Issues: https://github.com/footprintai/kafeido-sdk/issues
- Email: info@footprintai.com
