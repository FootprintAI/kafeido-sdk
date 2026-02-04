"""Text-to-speech example using Kafeido SDK."""

import time
from kafeido import OpenAI

client = OpenAI()

# Create a TTS job
job = client.audio.speech.create(
    model="qwen3-tts",
    input="Hello! This is a text-to-speech demonstration using the Kafeido API.",
    voice="alloy",
    response_format="mp3",
)
print(f"TTS job created: {job.job_id} (status: {job.status})")

# Poll for result
while True:
    result = client.audio.speech.get_result(job_id=job.job_id)
    print(f"Status: {result.status}, Progress: {result.progress}%")

    if result.status == "completed":
        print(f"Download URL: {result.result.download_url}")
        break
    elif result.status == "failed":
        print(f"Error: {result.error}")
        break

    time.sleep(2)
