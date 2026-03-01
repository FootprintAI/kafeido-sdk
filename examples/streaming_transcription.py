"""Real-time streaming transcription example.

Demonstrates sending audio chunks over WebSocket and receiving
transcription segments in real time.

Usage:
    python streaming_transcription.py <wav_file>

The WAV file should be 16kHz mono. Other formats will be resampled.
"""

import os
import struct
import sys
import wave

from kafeido import OpenAI


def read_wav_as_float32(path: str) -> bytes:
    """Read a WAV file and return raw float32 PCM bytes (16kHz mono)."""
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1, "WAV must be mono"
        assert wf.getsampwidth() == 2, "WAV must be 16-bit"
        assert wf.getframerate() == 16000, "WAV must be 16kHz"
        pcm16 = wf.readframes(wf.getnframes())

    # Convert int16 PCM to float32 PCM
    samples = struct.unpack(f"<{len(pcm16) // 2}h", pcm16)
    float32_bytes = struct.pack(f"<{len(samples)}f", *(s / 32768.0 for s in samples))
    return float32_bytes


def main():
    if len(sys.argv) < 2:
        print("Usage: python streaming_transcription.py <wav_file>")
        print("Example: python streaming_transcription.py recording.wav")
        sys.exit(1)

    wav_path = sys.argv[1]

    client = OpenAI(api_key=os.getenv("KAFEIDO_API_KEY"))

    print("Streaming Transcription Example")
    print("=" * 50)
    print(f"Streaming: {wav_path}\n")

    # Read WAV and convert to float32 PCM
    audio_data = read_wav_as_float32(wav_path)

    # Open streaming session
    stream = client.audio.transcriptions.stream(
        model="whisper-large-v3",
        language="en",
        use_vad=True,
    )

    # Send audio in 1-second chunks (16000 samples * 4 bytes/sample = 64000 bytes)
    chunk_size = 16000 * 4

    with stream:
        for offset in range(0, len(audio_data), chunk_size):
            chunk = audio_data[offset : offset + chunk_size]
            stream.send(chunk)

        # Receive all transcription responses
        for response in stream:
            if response.language:
                print(f"[lang={response.language} prob={response.language_prob:.2f}]")
            for seg in response.segments:
                status = "FINAL" if seg.completed else "partial"
                print(f"  [{seg.start:.2f}s - {seg.end:.2f}s] ({status}) {seg.text}")

    print("\nDone.")


if __name__ == "__main__":
    main()
