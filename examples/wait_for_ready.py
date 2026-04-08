"""Example: Using wait_for_ready for cold start handling.

This example demonstrates how to use the `wait_for_ready` parameter to
automatically handle model cold starts. When a model is not loaded (cold),
the SDK will trigger a warmup request and poll until the model becomes
healthy before making the actual request.

This is useful for:
- Serverless deployments where models may be unloaded after idle time
- First requests to a model that hasn't been used recently
- Ensuring reliable request completion without manual status checking

Usage:
    export KAFEIDO_API_KEY="your-api-key"
    python examples/wait_for_ready.py
"""

import asyncio

from kafeido import OpenAI, AsyncOpenAI, WarmupTimeoutError


def sync_example():
    """Synchronous example of wait_for_ready."""
    print("=== Synchronous Example ===\n")

    client = OpenAI()

    # Basic usage - wait for model to be ready before making request
    print("1. Basic chat completion with wait_for_ready:")
    try:
        response = client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[{"role": "user", "content": "Hello! What is 2+2?"}],
            wait_for_ready=True,  # Automatically handle cold start
        )
        print(f"   Response: {response.choices[0].message.content}\n")
    except WarmupTimeoutError as e:
        print(f"   Model {e.model} didn't warm up in {e.waited_seconds:.1f}s\n")

    # With custom timeout - useful for models that take longer to load
    print("2. Chat completion with custom timeout (2 minutes):")
    try:
        response = client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[{"role": "user", "content": "Tell me a short joke."}],
            wait_for_ready=True,
            warmup_timeout=120.0,  # Wait up to 2 minutes
        )
        print(f"   Response: {response.choices[0].message.content}\n")
    except WarmupTimeoutError as e:
        print(f"   Model {e.model} didn't warm up in {e.waited_seconds:.1f}s\n")

    # Streaming also works with wait_for_ready
    print("3. Streaming with wait_for_ready:")
    try:
        stream = client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            stream=True,
            wait_for_ready=True,
        )
        print("   Response: ", end="")
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")
    except WarmupTimeoutError as e:
        print(f"   Model {e.model} didn't warm up in {e.waited_seconds:.1f}s\n")


async def async_example():
    """Asynchronous example of wait_for_ready."""
    print("=== Asynchronous Example ===\n")

    async with AsyncOpenAI() as client:
        # Basic async usage
        print("1. Async chat completion with wait_for_ready:")
        try:
            response = await client.chat.completions.create(
                model="gpt-oss-20b",
                messages=[{"role": "user", "content": "What is the capital of France?"}],
                wait_for_ready=True,
            )
            print(f"   Response: {response.choices[0].message.content}\n")
        except WarmupTimeoutError as e:
            print(f"   Model {e.model} didn't warm up in {e.waited_seconds:.1f}s\n")


def audio_example():
    """Example with audio transcription."""
    print("=== Audio Transcription Example ===\n")

    client = OpenAI()

    # Note: You need an actual audio file for this example
    # This is just to show the API usage
    print("Audio transcription with wait_for_ready:")
    print("   (Requires an audio file to run)")
    print("""
   # Example code:
   with open("audio.mp3", "rb") as f:
       transcript = client.audio.transcriptions.create(
           file=f,
           model="whisper-large-v3",
           wait_for_ready=True,  # Wait for whisper model to load
           warmup_timeout=180.0,  # ASR models may take longer
       )
   print(transcript.text)
   """)


def main():
    """Run all examples."""
    print("=" * 60)
    print("Kafeido SDK - wait_for_ready Example")
    print("=" * 60)
    print()

    # Run sync example
    sync_example()

    # Run async example
    asyncio.run(async_example())

    # Show audio example code
    audio_example()

    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
