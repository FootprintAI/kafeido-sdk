# Changelog

All notable changes to the Kafeido Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-02-04

### Added
- Text-to-Speech (TTS) support via `client.audio.speech.create()` and `get_result()`
- OCR resource via `client.ocr.extractions.create()` with sync, async, and grounding modes
- Vision resource via `client.vision.analyze.create()` and `client.vision.chat.create()` with streaming
- Async transcription jobs via `client.audio.transcriptions.create_async()` and `get_result()`
- Model status checking via `client.models.status()`
- Model warmup/prefetch via `client.models.warmup()`
- Jobs resource for tracking async job status via `client.jobs.retrieve()`
- Request progress tracking via `client.jobs.progress()`
- Health check endpoint via `client.health()`
- All new resources have both sync and async variants

### New Endpoints
- POST /v1/audio/speech (TTS)
- GET /v1/audio/speech/{job_id}
- POST /v1/audio/transcriptions/async
- GET /v1/audio/transcriptions/async/{job_id}
- POST /v1/ocr/extract
- POST /v1/ocr/extract/async
- GET /v1/ocr/extract/async/{job_id}
- POST /v1/vision/analyze
- POST /v1/vision/chat (streaming)
- POST /v1/vision/analyze/async
- GET /v1/vision/analyze/async/{job_id}
- GET /v1/models/{model_id}/status
- POST /v1/models/warmup
- GET /v1/jobs/{job_id}
- GET /v1/requests/progress
- GET /v1/health

### New Models Supported
- TTS: qwen3-tts, xtts-v2
- Vision: llama-3.2-vision-11b, llama-3.2-vision-90b

## [0.1.0] - 2026-01-24

### Added
- Initial release of Kafeido Python SDK
- OpenAI-compatible client interface
- Chat completions API with streaming support
- Audio transcriptions and translations (Whisper models)
- Model listing and retrieval
- File upload and management
- Comprehensive error handling with detailed exception hierarchy
- Full type hints and Pydantic models for all API responses
- Sync and async support for all endpoints
- Automatic retry logic with exponential backoff
- Environment variable support for API key (KAFEIDO_API_KEY, OPENAI_API_KEY)
- Examples and comprehensive documentation
- Support for Python 3.8+

### Supported Models
- LLM: gpt-oss-20b, gpt-oss-120b
- ASR: whisper-large-v3, whisper-turbo
- OCR: deepseek-ocr, paddle-ocr

### Supported Endpoints
- POST /v1/chat/completions
- POST /v1/audio/transcriptions
- POST /v1/audio/translations
- GET /v1/models
- GET /v1/models/{model}
- POST /v1/audio/upload
- GET /v1/audio/files
- GET /v1/audio/files/{file_id}
- DELETE /v1/audio/files/{file_id}

[1.4.0]: https://github.com/footprintai/kafeido-sdk/releases/tag/v1.4.0
[0.1.0]: https://github.com/footprintai/kafeido-sdk/releases/tag/v0.1.0
