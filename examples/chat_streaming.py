"""Streaming chat completion example."""

import os
from kafeido import OpenAI

def main():
    # Initialize client
    client = OpenAI(api_key=os.getenv("KAFEIDO_API_KEY"))

    print("Streaming Chat Completion Example")
    print("=" * 50)
    print("Question: Write a short poem about artificial intelligence\n")
    print("Response: ", end="", flush=True)

    # Create streaming completion
    stream = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a creative poet."},
            {"role": "user", "content": "Write a short poem about artificial intelligence"}
        ],
        stream=True,
        temperature=0.8,
        max_tokens=200
    )

    # Stream the response
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    main()
