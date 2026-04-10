"""Fine-tune a model with LoRA using the Kafeido SDK.

This example demonstrates the full fine-tuning workflow:
1. Generate synthetic training data (JSONL)
2. Upload the training file
3. Create a fine-tuning job with LoRA hyperparameters
4. Poll for job completion with live training progress
5. Monitor training events/metrics

Supports resuming: pass an existing job ID to monitor a running job
without creating a new one. The job ID is printed at creation time.

Supported base models for fine-tuning (standard HuggingFace format):
  - gemma-3-1b       (fast for testing)
  - gemma-3-12b      (with QLoRA 4-bit)
  - gemma-3-27b      (with QLoRA 4-bit)
  - qwen3.5-9b       (with QLoRA)
  - qwen3.5-35b-a3b  (with QLoRA)

Note: gpt-oss-20b/120b are NOT compatible with standard QLoRA fine-tuning.

Usage:
    export KAFEIDO_API_KEY="sk-..."

    # Create a new fine-tuning job:
    python run_finetune.py [model]

    # Resume monitoring an existing job:
    python run_finetune.py --job-id <job-id>

    # List recent fine-tuning jobs:
    python run_finetune.py --list

    # Examples:
    python run_finetune.py                              # defaults to gemma-3-1b
    python run_finetune.py gemma-3-27b                  # use Gemma 3 27B
    python run_finetune.py --job-id c9f479f8-1925-...   # resume monitoring
    python run_finetune.py --list                       # show recent jobs
"""

import re
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


def list_jobs(client):
    """List recent fine-tuning jobs."""
    jobs = client.fine_tuning.jobs.list(limit=10)
    if not jobs.data:
        print("No fine-tuning jobs found.")
        return
    print(f"{'ID':<40} {'Model':<20} {'Status':<12} {'Created'}")
    print("-" * 100)
    for j in jobs.data:
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(j.created_at)) if j.created_at else "?"
        print(f"{j.id:<40} {j.model or '?':<20} {j.status:<12} {created}")


def create_job(client, model_name):
    """Create a new fine-tuning job and return it."""
    model_config = SUPPORTED_MODELS[model_name]
    print(f"Using model: {model_name}")

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
    print(f"\n  To resume monitoring later: python run_finetune.py --job-id {job.id}")
    return job


def main():
    client = OpenAI()

    # Parse CLI args
    if "--list" in sys.argv:
        list_jobs(client)
        return

    if "--job-id" in sys.argv:
        idx = sys.argv.index("--job-id")
        if idx + 1 >= len(sys.argv):
            print("Error: --job-id requires a value")
            sys.exit(1)
        job_id = sys.argv[idx + 1]
        print(f"=== Resuming monitoring for job {job_id} ===")
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"Model: {job.model}, Status: {job.status}")
    else:
        model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
        if model_name not in SUPPORTED_MODELS:
            print(f"Unsupported model: {model_name}")
            print(f"Supported: {', '.join(SUPPORTED_MODELS.keys())}")
            sys.exit(1)
        job = create_job(client, model_name)

    # Step 4: Poll for completion with live progress
    print("\n=== Step 4: Waiting for job to complete ===")
    last_event_id = None
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)

        # Show new training progress events since last poll
        try:
            events = client.fine_tuning.jobs.list_events(job.id, limit=20, after=last_event_id)
            if events.data:
                last_event_id = events.data[-1].id
                for event in events.data:
                    if event.type == "metrics":
                        d = event.data
                        step = (d.step or 0) if d else 0
                        total = (d.total_steps or 0) if d else 0
                        loss = (d.train_loss or 0.0) if d else 0.0
                        eta = (d.eta_seconds or 0) if d else 0
                        sps = (d.steps_per_second or 0.0) if d else 0.0
                        # Fallback: parse from message string if structured fields missing
                        if not step and event.message:
                            m = re.search(r"Step (\d+)/(\d+)", event.message)
                            if m:
                                step, total = int(m.group(1)), int(m.group(2))
                            m = re.search(r"loss:\s*([\d.]+)", event.message)
                            if m:
                                loss = float(m.group(1))
                            m = re.search(r"speed:\s*([\d.]+)", event.message)
                            if m:
                                sps = float(m.group(1))
                            m = re.search(r"ETA:\s*(\d+)s", event.message)
                            if m:
                                eta = int(m.group(1))
                        pct = (step / total * 100) if total > 0 else 0
                        print(f"  {pct:3.0f}% | {step}/{total} [loss={loss:.4f}, {sps:.2f} step/s, ETA={eta:.0f}s]")
                    else:
                        print(f"  [{event.level}] {event.message}")
        except Exception:
            pass  # Events may not be available yet

        if job.status in ("succeeded", "failed", "cancelled"):
            break

        time.sleep(10)

    # Step 5: Show results
    print("\n=== Step 5: Results ===")
    if job.status == "succeeded":
        print(f"Fine-tuned model: {job.fine_tuned_model}")
        print(f"Trained tokens: {job.trained_tokens}")
        print(f"Finished at: {job.finished_at}")

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
            warmup_timeout=600,  # fine-tuned models may need longer to cold-start
        )
        print(f"Response: {response.choices[0].message.content}")


if __name__ == "__main__":
    main()
