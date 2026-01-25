"""List available models example."""

import os
from kafeido import OpenAI

def main():
    # Initialize client
    client = OpenAI(api_key=os.getenv("KAFEIDO_API_KEY"))

    print("Available Models")
    print("=" * 50)

    # List all models
    models = client.models.list()

    print(f"Found {len(models.data)} models:\n")

    # Group by type
    llm_models = []
    asr_models = []
    ocr_models = []

    for model in models.data:
        if "gpt" in model.id.lower():
            llm_models.append(model)
        elif "whisper" in model.id.lower():
            asr_models.append(model)
        elif "ocr" in model.id.lower():
            ocr_models.append(model)

    if llm_models:
        print("Language Models (LLM):")
        for model in llm_models:
            print(f"  - {model.id}")
        print()

    if asr_models:
        print("Speech Recognition (ASR):")
        for model in asr_models:
            print(f"  - {model.id}")
        print()

    if ocr_models:
        print("Optical Character Recognition (OCR):")
        for model in ocr_models:
            print(f"  - {model.id}")
        print()

    # Get details for a specific model
    if models.data:
        print(f"Example - Details for '{models.data[0].id}':")
        model_details = client.models.retrieve(models.data[0].id)
        print(f"  ID: {model_details.id}")
        print(f"  Type: {model_details.object}")
        print(f"  Owned by: {model_details.owned_by}")


if __name__ == "__main__":
    main()
