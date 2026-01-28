"""Vision analysis example using Kafeido SDK."""

from kafeido import OpenAI

client = OpenAI()

# Analyze an image by URL
result = client.vision.analyze.create(
    model_id="llama-3.2-vision-11b",
    image_url="https://example.com/photo.jpg",
    prompt="Describe what you see in this image.",
    mode="general",
)
print(f"Analysis: {result.text}")

# Vision chat with streaming
stream = client.vision.chat.create(
    model_id="llama-3.2-vision-11b",
    messages=[
        {
            "role": "user",
            "content": "What is shown in this chart?",
            "images": [{"url": "https://example.com/chart.png"}],
        }
    ],
    stream=True,
)

print("\nVision chat response:")
for chunk in stream:
    if chunk.text:
        print(chunk.text, end="", flush=True)
print()
