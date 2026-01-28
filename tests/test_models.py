"""Tests for models resource."""

import pytest
import httpx
import respx

from kafeido import OpenAI, Model, ModelList


@respx.mock
def test_models_list(client, base_url, mock_models_list):
    """Test listing models."""
    route = respx.get(f"{base_url}/v1/models").mock(
        return_value=httpx.Response(200, json=mock_models_list)
    )

    # List models
    models = client.models.list()

    # Assertions
    assert isinstance(models, ModelList)
    assert models.object == "list"
    assert len(models.data) == 2

    # Check first model
    assert isinstance(models.data[0], Model)
    assert models.data[0].id == "gpt-oss-20b"
    assert models.data[0].object == "model"
    assert models.data[0].owned_by == "kafeido"

    # Check second model
    assert models.data[1].id == "whisper-large-v3"

    # Verify request
    assert route.called
    assert route.calls.last.request.method == "GET"


@respx.mock
def test_models_retrieve(client, base_url):
    """Test retrieving a specific model."""
    mock_model = {
        "id": "gpt-oss-20b",
        "object": "model",
        "created": 1677652288,
        "owned_by": "kafeido"
    }

    route = respx.get(f"{base_url}/v1/models/gpt-oss-20b").mock(
        return_value=httpx.Response(200, json=mock_model)
    )

    # Retrieve model
    model = client.models.retrieve("gpt-oss-20b")

    # Assertions
    assert isinstance(model, Model)
    assert model.id == "gpt-oss-20b"
    assert model.object == "model"
    assert model.owned_by == "kafeido"

    # Verify request
    assert route.called
    assert route.calls.last.request.method == "GET"


@respx.mock
def test_models_not_found(client, base_url):
    """Test retrieving non-existent model."""
    from kafeido import NotFoundError

    respx.get(f"{base_url}/v1/models/invalid-model").mock(
        return_value=httpx.Response(
            404,
            json={"error": {"message": "Model not found"}}
        )
    )

    # Should raise NotFoundError
    with pytest.raises(NotFoundError) as exc_info:
        client.models.retrieve("invalid-model")

    assert exc_info.value.status_code == 404
