# Fine-Tuning with Kafeido SDK

This guide walks you through fine-tuning a model end-to-end on [Kafeido](https://api.kafeido.app), from account setup to running inference on your custom model.

## Step 1: Create Your Account

1. Go to [api.kafeido.app](https://api.kafeido.app) and sign up with Google or GitHub.

2. **Upgrade to the Pro tier** — fine-tuning requires a Starter plan or higher, but **Pro is recommended** as it includes:
   - 3M API tokens/month (enough for multiple fine-tuning runs)
   - Higher rate limits (100 requests/min, 10K requests/day)
   - Pay-as-you-go overage billing (continue using the API after your monthly allocation runs out)

   Navigate to **Billing & Subscription** in the console to upgrade.

3. **Create an API key:**
   - Go to **Settings > API Keys** in the console
   - Click **Create API Key**, give it a name (e.g., "fine-tuning")
   - Copy the key — it will only be shown once
   - Set it as an environment variable:
     ```bash
     export KAFEIDO_API_KEY="sk-..."
     ```

## Step 2: Install the SDK

```bash
pip install kafeido
```

## Step 3: Fine-Tune a Model

We recommend **Gemma 3 27B** as a strong starting point — it's a 27B parameter model that runs efficiently with QLoRA 4-bit quantization.

### Prepare Your Training Data

Training data must be in JSONL format with chat-completion messages:

```json
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "How do I..."}, {"role": "assistant", "content": "To do that..."}]}
```

This example includes synthetic customer support data. To use your own data, replace `training_data.jsonl` and `validation_data.jsonl`.

### Submit a Fine-Tuning Job

```bash
# Fine-tune Gemma 3 27B (recommended)
python run_finetune.py gemma-3-27b

# Or start small with Gemma 3 1B for quick testing
python run_finetune.py gemma-3-1b
```

The script will:
1. Generate training data (1000 examples) and validation data (100 examples)
2. Upload both files to the Kafeido API
3. Create a fine-tuning job with optimized LoRA hyperparameters
4. Show live training progress (loss, step/s, ETA)

You can safely close the terminal and resume monitoring later:

```bash
python run_finetune.py --job-id <job-id>
```

### Supported Models

| Model | Parameters | Use Case | Batch Size | LoRA Rank |
|-------|-----------|----------|------------|-----------|
| `gemma-3-1b` | 1B | Quick testing | 4 | 16 |
| `gemma-3-12b` | 12B | Good balance | 2 | 16 |
| `gemma-3-27b` | 27B | Best quality (recommended) | 1 | 8 |
| `qwen3.5-9b` | 9B | Alternative | 2 | 16 |
| `qwen3.5-35b-a3b` | 35B (MoE) | Large MoE | 1 | 8 |

## Step 4: Run Inference on Your Fine-Tuned Model

Once the fine-tuning job succeeds, you'll get a model name like `ft:gemma-3-27b:acme-support-1f971c25`.

```bash
python run_inference.py ft:gemma-3-27b:acme-support-1f971c25
```

The first request triggers a cold start (model loading), which takes 1-3 minutes. The SDK handles this automatically with `wait_for_ready=True`.

### Use in Your Own Code

```python
from kafeido import OpenAI

client = OpenAI()  # Uses KAFEIDO_API_KEY env var

response = client.chat.completions.create(
    model="ft:gemma-3-27b:acme-support-1f971c25",
    messages=[
        {"role": "system", "content": "You are a helpful customer support agent."},
        {"role": "user", "content": "How do I reset my password?"},
    ],
    wait_for_ready=True,   # Auto-triggers cold start if model is not loaded
    warmup_timeout=600,    # Wait up to 10 minutes for model to load
)

print(response.choices[0].message.content)
```

## Important Notes

- **Fine-tuning is exclusive** — one fine-tuning job runs per GPU at a time. Additional jobs queue automatically and start when resources are available.
- **Cold start** — fine-tuned models are loaded on-demand. The first inference request after fine-tuning (or after idle timeout) takes 1-3 minutes. Use `wait_for_ready=True` to handle this automatically.
- **Token billing** — fine-tuning consumes API tokens based on the number of training tokens processed. A typical run with 1000 examples on Gemma 3 27B uses ~3M tokens.
- **Pay-as-you-go** — if you exceed your monthly token allocation, enable overage billing in **Billing > Overage Billing** to continue using the API. You can set a monthly spending limit to control costs.
- **API compatibility** — the Kafeido SDK is a drop-in wrapper around the OpenAI Python SDK. You can use `from kafeido import OpenAI` anywhere you'd use `from openai import OpenAI`.

## Files in This Example

| File | Description |
|------|-------------|
| `run_finetune.py` | Submit and monitor a fine-tuning job |
| `run_inference.py` | Run inference on a fine-tuned model |
| `prepare_data.py` | Generate synthetic training data |
| `training_data.jsonl` | Sample training data (customer support) |
| `validation_data.jsonl` | Sample validation data |
