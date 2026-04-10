"""Run inference on a fine-tuned model.

Usage:
    export KAFEIDO_API_KEY="sk-..."
    python run_inference.py <model-name>

    # Examples:
    python run_inference.py ft:gemma-3-27b:acme-support-1f971c25
"""

import sys

from kafeido import OpenAI


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_inference.py <fine-tuned-model-name>")
        print("Example: python run_inference.py ft:gemma-3-27b:acme-support-1f971c25")
        sys.exit(1)

    model = sys.argv[1]
    client = OpenAI()

    print(f"Using model: {model}")
    print("Waiting for model to be ready...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful customer support agent for Acme Cloud."},
            {"role": "user", "content": "How do I reset my password?"},
        ],
        wait_for_ready=True,
        warmup_timeout=600,
    )
    print(f"Response: {response.choices[0].message.content}")


if __name__ == "__main__":
    main()
