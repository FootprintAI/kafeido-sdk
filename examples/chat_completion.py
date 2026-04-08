"""Basic chat completion example."""

import os
from kafeido import OpenAI

def main():
    # Initialize client (reads API key from environment)
    client = OpenAI(api_key=os.getenv("KAFEIDO_API_KEY"))

    print("Chat Completion Example")
    print("=" * 50)

    # Simple completion
    response = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning in one sentence?"}
        ],
        temperature=0.7,
        max_tokens=100
    )

    print(f"Response: {response.choices[0].message.content}")
    print(f"\nModel: {response.model}")
    print(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")


if __name__ == "__main__":
    main()
