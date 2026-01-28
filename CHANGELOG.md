# Changelog

All notable changes to the Kafeido Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/footprintai/kafeido-sdk/releases/tag/v0.1.0
