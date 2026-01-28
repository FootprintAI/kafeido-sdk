"""Audio transcription example."""

import os
import sys
from kafeido import OpenAI

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_audio.py <audio_file>")
        print("Example: python transcribe_audio.py audio.mp3")
        sys.exit(1)

    audio_file_path = sys.argv[1]

    # Initialize client
    client = OpenAI(api_key=os.getenv("KAFEIDO_API_KEY"))

    print("Audio Transcription Example")
    print("=" * 50)
    print(f"Transcribing: {audio_file_path}\n")

    # Transcribe audio file
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            response_format="verbose_json"
        )

    print(f"Transcript: {transcript.text}\n")

    if transcript.language:
        print(f"Detected Language: {transcript.language}")

    if transcript.duration:
        print(f"Duration: {transcript.duration:.2f} seconds")

    # Show segments if available
    if transcript.segments:
        print(f"\nSegments ({len(transcript.segments)} total):")
        for i, segment in enumerate(transcript.segments[:5]):  # Show first 5
            print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

        if len(transcript.segments) > 5:
            print(f"... and {len(transcript.segments) - 5} more segments")


if __name__ == "__main__":
    main()
