#!/usr/bin/env python3
"""Example demonstrating async usage of the Kafeido SDK."""

import asyncio
import os
from kafeido import AsyncOpenAI


async def main():
    """Demonstrate async API usage."""

    # Initialize async client
    # API key can be set via environment variable KAFEIDO_API_KEY or OPENAI_API_KEY
    api_key = os.getenv("KAFEIDO_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set KAFEIDO_API_KEY or OPENAI_API_KEY environment variable")
        return

    # Use async context manager for automatic cleanup
    async with AsyncOpenAI(api_key=api_key) as client:
        print("=== Async Chat Completion ===")
        response = await client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}\n")

        print("=== Async Streaming Chat ===")
        stream = await client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[
                {"role": "user", "content": "Count from 1 to 5"}
            ],
            stream=True
        )

        print("Streaming response: ", end="", flush=True)
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")

        print("=== Async Model List ===")
        models = await client.models.list()
        print(f"Available models: {len(models.data)}")
        for model in models.data[:5]:  # Show first 5
            print(f"  - {model.id}")
        print()

        # Example: Audio transcription (uncomment if you have an audio file)
        # print("=== Async Audio Transcription ===")
        # with open("audio.mp3", "rb") as audio_file:
        #     transcript = await client.audio.transcriptions.create(
        #         file=audio_file,
        #         model="whisper-large-v3"
        #     )
        #     print(f"Transcript: {transcript.text}\n")


async def concurrent_requests_example():
    """Demonstrate making multiple concurrent async requests."""

    api_key = os.getenv("KAFEIDO_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set KAFEIDO_API_KEY or OPENAI_API_KEY environment variable")
        return

    async with AsyncOpenAI(api_key=api_key) as client:
        print("=== Concurrent Async Requests ===")

        # Create multiple requests concurrently
        questions = [
            "What is 2+2?",
            "What is the capital of Japan?",
            "What is the largest ocean?"
        ]

        # Run all requests concurrently using asyncio.gather
        tasks = [
            client.chat.completions.create(
                model="gpt-oss-20b",
                messages=[{"role": "user", "content": q}],
                max_tokens=50
            )
            for q in questions
        ]

        responses = await asyncio.gather(*tasks)

        for question, response in zip(questions, responses):
            print(f"Q: {question}")
            print(f"A: {response.choices[0].message.content}\n")


if __name__ == "__main__":
    print("Running async examples...\n")

    # Run main example
    asyncio.run(main())

    # Run concurrent requests example
    print("\n" + "="*50 + "\n")
    asyncio.run(concurrent_requests_example())
