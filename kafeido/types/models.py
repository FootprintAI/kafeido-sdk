"""Model types - OpenAI compatible."""

from typing import List, Literal, Optional

from pydantic import BaseModel


class Model(BaseModel):
    """Model information."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "kafeido"


class ModelList(BaseModel):
    """List of models."""

    object: Literal["list"] = "list"
    data: List[Model]
