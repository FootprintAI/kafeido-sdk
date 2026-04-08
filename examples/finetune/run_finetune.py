"""Fine-tune a model with LoRA using the Kafeido SDK.

This example demonstrates the full fine-tuning workflow:
1. Generate synthetic training data (JSONL)
2. Upload the training file
3. Create a fine-tuning job with LoRA hyperparameters
4. Poll for job completion
5. Monitor training events/metrics

Supported base models for fine-tuning (standard HuggingFace format):
  - gemma-3-1b       (fast for testing)
  - gemma-3-12b      (with QLoRA 4-bit)
  - gemma-3-27b      (with QLoRA 4-bit)
  - qwen3.5-9b       (with QLoRA)
  - qwen3.5-35b-a3b  (with QLoRA)

Note: gpt-oss-20b/120b are NOT compatible with standard QLoRA fine-tuning.

Usage:
    export KAFEIDO_API_KEY="sk-..."
    python run_finetune.py [model]

    # Examples:
    python run_finetune.py                  # defaults to gemma-3-1b
    python run_finetune.py qwen3.5-9b       # use Qwen 3.5 9B
"""

import sys
import time

from kafeido import OpenAI, FineTuningHyperparameters
from prepare_data import generate_training_file

# Default model for fine-tuning (smallest, fastest for testing)
DEFAULT_MODEL = "gemma-3-1b"
SUPPORTED_MODELS = {
    "gemma-3-1b": {"batch_size": 4, "lora_rank": 16},
    "gemma-3-12b": {"batch_size": 2, "lora_rank": 16},
    "gemma-3-27b": {"batch_size": 1, "lora_rank": 8},
    "qwen3.5-9b": {"batch_size": 2, "lora_rank": 16},
    "qwen3.5-35b-a3b": {"batch_size": 1, "lora_rank": 8},
}


def main():
    # Parse model from CLI args
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    if model_name not in SUPPORTED_MODELS:
        print(f"Unsupported model: {model_name}")
        print(f"Supported: {', '.join(SUPPORTED_MODELS.keys())}")
        sys.exit(1)

    model_config = SUPPORTED_MODELS[model_name]
    print(f"Using model: {model_name}")

    client = OpenAI()

    # Step 1: Generate synthetic training and validation data
    print("=== Step 1: Generating training and validation data ===")
    training_file_path = generate_training_file("training_data.jsonl", n_examples=1000)
    validation_file_path = generate_training_file("validation_data.jsonl", n_examples=100)

    # Step 2: Upload the training and validation files
    print("\n=== Step 2: Uploading training and validation files ===")
    with open(training_file_path, "rb") as f:
        train_file_obj = client.files.create(file=f, purpose="fine-tune")
    train_file_id = train_file_obj.file_id or train_file_obj.id
    print(f"Uploaded training file: {train_file_id}")

    with open(validation_file_path, "rb") as f:
        val_file_obj = client.files.create(file=f, purpose="fine-tune")
    val_file_id = val_file_obj.file_id or val_file_obj.id
    print(f"Uploaded validation file: {val_file_id}")

    # Step 3: Create fine-tuning job with LoRA
    print(f"\n=== Step 3: Creating fine-tuning job (LoRA on {model_name}) ===")
    job = client.fine_tuning.jobs.create(
        model=model_name,
        training_file=train_file_id,
        validation_file=val_file_id,
        suffix="acme-support",
        hyperparameters=FineTuningHyperparameters(
            n_epochs=3,
            learning_rate=2e-4,
            batch_size=model_config["batch_size"],
            lora_rank=model_config["lora_rank"],
            lora_alpha=model_config["lora_rank"] * 2,
            lora_dropout=0.05,
            quantization="QUANTIZATION_4BIT",  # QLoRA
        ),
    )
    print(f"Job created: {job.id}")
    print(f"Status: {job.status}")
    print(f"Base model: {job.model}")

    # Step 4: Poll for completion
    print("\n=== Step 4: Waiting for job to complete ===")
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(f"  Status: {job.status}", end="")

        if job.trained_tokens:
            print(f" | Trained tokens: {job.trained_tokens}", end="")
        print()

        if job.status in ("succeeded", "failed", "cancelled"):
            break

        time.sleep(10)

    # Step 5: Show results
    print("\n=== Step 5: Results ===")
    if job.status == "succeeded":
        print(f"Fine-tuned model: {job.fine_tuned_model}")
        print(f"Trained tokens: {job.trained_tokens}")
        print(f"Finished at: {job.finished_at}")

        # List training events/metrics
        print("\n--- Training Events ---")
        events = client.fine_tuning.jobs.list_events(job.id, limit=20)
        for event in events.data or []:
            msg = f"[{event.level}] {event.message}"
            if event.data:
                msg += f" (step={event.data.step}, loss={event.data.train_loss})"
            print(f"  {msg}")
    elif job.status == "failed":
        print(f"Job failed: {job.error}")
    else:
        print(f"Job was cancelled")

    # Bonus: Use the fine-tuned model for inference
    if job.status == "succeeded" and job.fine_tuned_model:
        print("\n=== Bonus: Testing fine-tuned model ===")
        response = client.chat.completions.create(
            model=job.fine_tuned_model,
            messages=[
                {"role": "system", "content": "You are a helpful customer support agent for Acme Cloud."},
                {"role": "user", "content": "How do I reset my password?"},
            ],
            wait_for_ready=True,
        )
        print(f"Response: {response.choices[0].message.content}")


if __name__ == "__main__":
    main()
