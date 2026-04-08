"""Model status and warmup example using Kafeido SDK."""

from kafeido import OpenAI

client = OpenAI()

# Check model status
status = client.models.status("whisper-large-v3")
print(f"Model: {status.model_id}")
if status.status:
    print(f"  Status: {status.status.status}")
    if status.status.cold_start_progress:
        print(f"  Cold start stage: {status.status.cold_start_progress.stage}")
        print(f"  Progress: {status.status.cold_start_progress.progress}")

# Warmup a model
warmup = client.models.warmup(model="whisper-large-v3")
if warmup.already_warm:
    print("\nModel is already warm and ready to serve requests.")
else:
    print(f"\nModel is warming up. ETA: {warmup.estimated_seconds}s")

# Check health
health = client.health()
print(f"\nAPI Health: {health.status}")
print(f"Version: {health.version}")
