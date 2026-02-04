#!/usr/bin/env python3
"""Migration guide from OpenAI SDK to Kafeido SDK.

The Kafeido SDK is fully compatible with the OpenAI SDK API.
This means you can use it as a drop-in replacement with minimal changes.
"""

import os

# ==============================================================================
# 1. Installation
# ==============================================================================

"""
Old (OpenAI SDK):
    pip install openai

New (Kafeido SDK):
    pip install kafeido
"""

# ==============================================================================
# 2. Import Statement
# ==============================================================================

# OLD: from openai import OpenAI
from kafeido import OpenAI  # NEW: Just change the import!

# For async:
# OLD: from openai import AsyncOpenAI
from kafeido import AsyncOpenAI  # NEW: Just change the import!

# ==============================================================================
# 3. Client Initialization
# ==============================================================================

# The client initialization is IDENTICAL
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # Same environment variable!
    base_url="https://api.kafeido.app",    # Point to Kafeido API
    timeout=120.0,
    max_retries=2,
)

# You can also use KAFEIDO_API_KEY environment variable
# client = OpenAI(api_key=os.getenv("KAFEIDO_API_KEY"))

# ==============================================================================
# 4. Chat Completions - IDENTICAL API
# ==============================================================================

def chat_completion_example():
    """Chat completion works exactly the same way."""

    # This code is IDENTICAL between OpenAI and Kafeido SDKs
    response = client.chat.completions.create(
        model="gpt-oss-20b",  # Use Kafeido models
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        temperature=0.7,
        max_tokens=100
    )

    print(response.choices[0].message.content)

    # Streaming also works identically
    stream = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


# ==============================================================================
# 5. Audio Transcriptions - IDENTICAL API
# ==============================================================================

def audio_transcription_example():
    """Audio transcription works exactly the same way."""

    # This code is IDENTICAL between OpenAI and Kafeido SDKs
    with open("audio.mp3", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-large-v3",  # Use Kafeido Whisper models
            file=audio_file,
            language="en"
        )

    print(transcript.text)


# ==============================================================================
# 6. Models API - IDENTICAL API
# ==============================================================================

def models_example():
    """Models API works exactly the same way."""

    # List all models
    models = client.models.list()
    for model in models.data:
        print(f"- {model.id}")

    # Retrieve specific model
    model = client.models.retrieve("gpt-oss-20b")
    print(f"Model: {model.id}, Owner: {model.owned_by}")


# ==============================================================================
# 7. Async Usage - IDENTICAL API
# ==============================================================================

async def async_example():
    """Async usage is identical to OpenAI SDK."""
    import asyncio

    # Initialize async client
    async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Same API as OpenAI async client
    response = await async_client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    print(response.choices[0].message.content)

    # Async streaming
    stream = await async_client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

    await async_client.close()


# ==============================================================================
# 8. Error Handling - IDENTICAL API
# ==============================================================================

def error_handling_example():
    """Error handling works exactly the same way."""

    from kafeido import (
        AuthenticationError,
        NotFoundError,
        RateLimitError,
        APIError
    )

    try:
        response = client.chat.completions.create(
            model="invalid-model",
            messages=[{"role": "user", "content": "Hello"}]
        )
    except AuthenticationError as e:
        print(f"Auth error: {e}")
    except NotFoundError as e:
        print(f"Model not found: {e}")
    except RateLimitError as e:
        print(f"Rate limit exceeded: {e}")
    except APIError as e:
        print(f"API error: {e}")


# ==============================================================================
# 9. Key Differences
# ==============================================================================

"""
The main differences are:

1. **Model Names**: Use Kafeido model names instead of OpenAI model names
   - OpenAI: "gpt-4", "gpt-3.5-turbo"
   - Kafeido: "gpt-oss-20b", "gpt-oss-120b"
   - Whisper: "whisper-large-v3", "whisper-turbo"

2. **Base URL**: Point to Kafeido API
   - OpenAI: "https://api.openai.com/v1"
   - Kafeido: "https://api.kafeido.app"

3. **API Key**: Use Kafeido API keys (start with "sk-")
   - Set via KAFEIDO_API_KEY or OPENAI_API_KEY environment variable

Everything else is IDENTICAL! The SDK is designed to be a drop-in replacement.
"""

# ==============================================================================
# 10. Complete Migration Example
# ==============================================================================

def complete_migration_example():
    """Complete example showing before/after migration."""

    # BEFORE (OpenAI SDK):
    # from openai import OpenAI
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": "Hello!"}]
    # )

    # AFTER (Kafeido SDK) - Just 2 changes!
    from kafeido import OpenAI  # Change 1: Import from kafeido
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.kafeido.app"  # Change 2: Point to Kafeido
    )
    response = client.chat.completions.create(
        model="gpt-oss-20b",  # Change 3: Use Kafeido model name
        messages=[{"role": "user", "content": "Hello!"}]
    )

    print(response.choices[0].message.content)


# ==============================================================================
# Summary
# ==============================================================================

"""
Migration Steps:
1. Install kafeido: `pip install kafeido`
2. Change import: `from kafeido import OpenAI`
3. Update base_url to "https://api.kafeido.app"
4. Use Kafeido model names
5. Set KAFEIDO_API_KEY or OPENAI_API_KEY

That's it! Your code should work without any other changes.
"""

if __name__ == "__main__":
    print("Kafeido SDK Migration Guide")
    print("=" * 50)
    print("\nThe Kafeido SDK is 100% compatible with OpenAI SDK API.")
    print("Most code can be migrated by just changing 2-3 lines!\n")

    # Run examples
    print("Running chat completion example...")
    chat_completion_example()

    print("\nFor more examples, see:")
    print("- examples/chat_completion.py")
    print("- examples/chat_streaming.py")
    print("- examples/transcribe_audio.py")
    print("- examples/async_example.py")
