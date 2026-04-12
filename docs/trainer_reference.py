# ============================================================================
# Fine-Tuning Trainer Reference (READ-ONLY)
#
# This is a copy of the server-side fine-tuning trainer for transparency.
# It shows exactly how your training data is processed when you submit a
# fine-tuning job via the Kafeido API.
#
# You CANNOT modify this script or supply your own — fine-tuning jobs
# always use this pipeline. You can configure hyperparameters via the API.
#
# Key details:
#   - Method: QLoRA (4-bit quantization + LoRA adapters) via HuggingFace SFTTrainer
#   - Data format: OpenAI chat completion JSONL
#   - Configurable: epochs, learning_rate, batch_size, lora_rank, lora_alpha,
#                   lora_dropout, quantization (4bit/8bit/none)
#   - Not configurable: optimizer, trainer, loss function, chat template
# ============================================================================
#
# Copyright 2025 FootprintAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LoRA fine-tuning trainer using PEFT + TRL.

Supports QLoRA (4-bit quantized base model + LoRA adapters) for memory-efficient
fine-tuning of large language models. Training data uses OpenAI chat completion
JSONL format.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

logger = logging.getLogger(__name__)


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA fine-tuning."""

    # Model
    base_model_path: str = ""
    output_dir: str = "/tmp/lora-output"

    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # Training parameters
    n_epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 100

    # Quantization mode: "4bit" (QLoRA), "8bit", or "none" (full precision)
    quantization: str = "4bit"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # Device: "auto" (default), "cuda", "cpu"
    device: str = "auto"


def load_chat_jsonl(path: str, tokenizer) -> Dataset:
    """Load OpenAI chat completion JSONL training data.

    Expected format per line:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    conversations = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping invalid JSON at line %d: %s", line_num, e)
                continue

            messages = data.get("messages", [])
            if not messages:
                logger.warning("Skipping line %d: no messages found", line_num)
                continue

            # Apply chat template to convert messages to a single training string
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            conversations.append({"text": text})

    logger.info("Loaded %d training examples from %s", len(conversations), path)
    return Dataset.from_list(conversations)


def train_lora(
    config: LoRATrainingConfig,
    training_data_path: str,
    progress_callback: Optional[Callable] = None,
    validation_data_path: Optional[str] = None,
) -> dict:
    """Run LoRA fine-tuning and return results.

    Args:
        config: Training configuration.
        training_data_path: Path to JSONL training file.
        progress_callback: Optional callback(step, loss, accuracy) for progress reporting.
        validation_data_path: Optional path to JSONL validation file for evaluation.

    Returns:
        dict with keys: adapter_dir, trained_tokens, final_loss
    """
    logger.info("Starting LoRA training: model=%s, epochs=%d, lr=%s",
                config.base_model_path, config.n_epochs, config.learning_rate)

    # Resolve device
    device = config.device.lower() if config.device else "auto"
    use_cpu = device == "cpu"

    if use_cpu:
        logger.info("Forcing CPU mode — disabling quantization")

    # Set up quantization config based on mode
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype, torch.bfloat16)
    if use_cpu:
        compute_dtype = torch.float32  # CPU doesn't support bfloat16 well on all hardware
    bnb_config = None
    quant_mode = config.quantization.lower() if config.quantization else "4bit"

    # BitsAndBytes quantization requires CUDA — skip on CPU
    if use_cpu:
        quant_mode = "none"

    if quant_mode == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif quant_mode == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    # If the model has an "original/" subdirectory with BF16 weights, use those
    # for fine-tuning. The root directory may contain MXFP4 weights which are
    # incompatible with BitsAndBytes quantization (QLoRA).
    # The original/ dir only has safetensors weights — copy config files from root,
    # stripping the quantization_config to avoid MXFP4/BnB conflict.
    model_load_path = config.base_model_path
    original_path = os.path.join(config.base_model_path, "original")
    if os.path.isdir(original_path) and bnb_config is not None:
        logger.info("Using original/ BF16 weights for QLoRA (root has MXFP4 weights)")
        import json as _json
        import shutil

        for config_file in ["generation_config.json", "tokenizer_config.json",
                            "tokenizer.json", "special_tokens_map.json", "chat_template.jinja"]:
            src = os.path.join(config.base_model_path, config_file)
            dst = os.path.join(original_path, config_file)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

        # Copy config.json but strip quantization_config (MXFP4 conflicts with BnB).
        # Always overwrite — the original/ config.json may be minimal (missing model_type).
        src_config = os.path.join(config.base_model_path, "config.json")
        dst_config = os.path.join(original_path, "config.json")
        if os.path.exists(src_config):
            with open(src_config) as f:
                model_config = _json.load(f)
            model_config.pop("quantization_config", None)
            with open(dst_config, "w") as f:
                _json.dump(model_config, f, indent=2)
            logger.info("Wrote config.json to original/ (stripped quantization_config)")

        model_load_path = original_path

    device_map = "cpu" if use_cpu else "auto"
    logger.info("Loading base model from %s with %s quantization on device_map=%s...", model_load_path, quant_mode, device_map)
    model = AutoModelForCausalLM.from_pretrained(
        model_load_path,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable_params, total_params = model.get_nb_trainable_parameters()
    logger.info(
        "LoRA parameters: trainable=%d (%.2f%%), total=%d",
        trainable_params,
        100 * trainable_params / total_params,
        total_params,
    )

    # Load training data
    dataset = load_chat_jsonl(training_data_path, tokenizer)
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty")

    # Load validation data (optional)
    eval_dataset = None
    if validation_data_path:
        eval_dataset = load_chat_jsonl(validation_data_path, tokenizer)
        if len(eval_dataset) == 0:
            logger.warning("Validation dataset is empty, skipping evaluation")
            eval_dataset = None
        else:
            logger.info("Loaded %d validation examples", len(eval_dataset))

    # Set up training arguments
    os.makedirs(config.output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        bf16=not use_cpu and compute_dtype == torch.bfloat16,
        fp16=not use_cpu and compute_dtype == torch.float16,
        optim="paged_adamw_32bit" if quant_mode in ("4bit", "8bit") else "adamw_torch",
        use_cpu=use_cpu,
        report_to="none",
        remove_unused_columns=False,
        eval_strategy="epoch" if eval_dataset is not None else "no",
    )

    # Custom callback for progress reporting
    callbacks = []
    if progress_callback is not None:
        from transformers import TrainerCallback

        class ProgressReporter(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    progress_callback(
                        step=state.global_step,
                        loss=logs.get("loss", 0.0),
                        accuracy=logs.get("train_mean_token_accuracy", 0.0),
                        total_steps=state.max_steps,
                    )

        callbacks.append(ProgressReporter())

    # Initialize trainer
    # Note: max_seq_length moved to TrainingArguments in trl>=0.16.0
    training_args.max_seq_length = config.max_seq_length
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Train
    logger.info("Starting training with %d examples...", len(dataset))
    train_result = trainer.train()

    # Save adapter only (not the full model)
    adapter_dir = os.path.join(config.output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("LoRA adapter saved to %s", adapter_dir)

    # Compute results
    final_loss = train_result.training_loss
    trained_tokens = int(train_result.metrics.get("train_tokens", 0))
    if trained_tokens == 0:
        # SFTTrainer tokenizes internally so our dataset may not have input_ids.
        # Serialize each example to a string and tokenize to count actual tokens.
        # This is structure-agnostic — works regardless of dataset format
        # (text, messages, or any other schema).
        import json as _json
        total_tokens_per_epoch = 0
        for example in dataset:
            serialized = _json.dumps(example, ensure_ascii=False)
            total_tokens_per_epoch += len(tokenizer.encode(serialized, truncation=True,
                                                            max_length=config.max_seq_length))
        if total_tokens_per_epoch > 0:
            trained_tokens = total_tokens_per_epoch * config.n_epochs
            logger.info("Counted actual trained tokens: %d per epoch x %d epochs = %d total",
                        total_tokens_per_epoch, config.n_epochs, trained_tokens)
        else:
            trained_tokens = len(dataset) * config.n_epochs * config.max_seq_length // 2
            logger.warning("Could not determine actual token count from dataset, "
                           "using max_seq_length estimate: %d", trained_tokens)

    return {
        "adapter_dir": adapter_dir,
        "trained_tokens": trained_tokens,
        "final_loss": final_loss,
    }
