"""Fine-tuning types."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Quantization(str, Enum):
    """Quantization mode for base model loading during fine-tuning."""

    UNSPECIFIED = "QUANTIZATION_UNSPECIFIED"
    FOUR_BIT = "QUANTIZATION_4BIT"
    EIGHT_BIT = "QUANTIZATION_8BIT"
    NONE = "QUANTIZATION_NONE"


class FineTuningHyperparameters(BaseModel):
    """Training hyperparameters for fine-tuning."""

    n_epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    quantization: Optional[str] = None


class FineTuningJob(BaseModel):
    """A fine-tuning job (OpenAI-compatible)."""

    model_config = {"populate_by_name": True}

    id: str
    object: Optional[str] = "fine_tuning.job"
    model: Optional[str] = None
    fine_tuned_model: Optional[str] = Field(None, alias="fineTunedModel")
    organization_id: Optional[str] = Field(None, alias="organizationId")
    status: Optional[str] = None
    training_file: Optional[str] = Field(None, alias="trainingFile")
    validation_file: Optional[str] = Field(None, alias="validationFile")
    hyperparameters: Optional[FineTuningHyperparameters] = None
    trained_tokens: Optional[int] = Field(None, alias="trainedTokens")
    error: Optional[str] = None
    created_at: Optional[str] = Field(None, alias="createdAt")
    finished_at: Optional[str] = Field(None, alias="finishedAt")
    suffix: Optional[str] = None


class FineTuningMetrics(BaseModel):
    """Per-step training metrics."""

    model_config = {"populate_by_name": True}

    step: Optional[int] = None
    train_loss: Optional[float] = Field(None, alias="trainLoss")
    train_mean_token_accuracy: Optional[float] = Field(None, alias="trainMeanTokenAccuracy")
    total_steps: Optional[int] = Field(None, alias="totalSteps")
    elapsed_seconds: Optional[float] = Field(None, alias="elapsedSeconds")
    eta_seconds: Optional[float] = Field(None, alias="etaSeconds")
    steps_per_second: Optional[float] = Field(None, alias="stepsPerSecond")


class FineTuningEvent(BaseModel):
    """A training progress event."""

    model_config = {"populate_by_name": True}

    id: Optional[str] = None
    object: Optional[str] = "fine_tuning.event"
    created_at: Optional[str] = Field(None, alias="createdAt")
    level: Optional[str] = None
    message: Optional[str] = None
    type: Optional[str] = None
    data: Optional[FineTuningMetrics] = None


class FineTuningJobList(BaseModel):
    """List of fine-tuning jobs."""

    data: Optional[List[FineTuningJob]] = None
    has_more: Optional[bool] = None
    object: Optional[str] = "list"


class FineTuningEventList(BaseModel):
    """List of fine-tuning events."""

    model_config = {"populate_by_name": True}

    data: Optional[List[FineTuningEvent]] = None
    has_more: Optional[bool] = Field(None, alias="hasMore")
    object: Optional[str] = "list"
