"""Async models resource."""

from kafeido._http_client import AsyncHTTPClient
from kafeido.types.models import Model, ModelList, ModelStatus, WarmupResponse


class AsyncModels:
    """Async models resource for listing and retrieving model information."""

    def __init__(self, http_client: AsyncHTTPClient) -> None:
        """Initialize async models resource.

        Args:
            http_client: The async HTTP client to use for requests.
        """
        self._client = http_client

    async def list(self) -> ModelList:
        """List available models asynchronously.

        Returns:
            ModelList containing all available models.

        Example:
            >>> client = AsyncOpenAI(api_key="sk-...")
            >>> models = await client.models.list()
            >>> for model in models.data:
            ...     print(model.id)
        """
        response_data = await self._client.get("/v1/models")
        return ModelList.model_validate(response_data)

    async def retrieve(self, model: str) -> Model:
        """Retrieve information about a specific model asynchronously.

        Args:
            model: The model ID to retrieve (e.g., "gpt-oss-20b").

        Returns:
            Model information.

        Example:
            >>> client = AsyncOpenAI(api_key="sk-...")
            >>> model = await client.models.retrieve("gpt-oss-20b")
            >>> print(model.id, model.owned_by)
        """
        response_data = await self._client.get(f"/v1/models/{model}")
        return Model.model_validate(response_data)

    async def status(self, model: str) -> ModelStatus:
        """Get the status of a model asynchronously."""
        response_data = await self._client.get(f"/v1/models/{model}/status")
        return ModelStatus.model_validate(response_data)

    async def warmup(self, *, model: str) -> WarmupResponse:
        """Warmup/prefetch a model asynchronously."""
        response_data = await self._client.post(
            "/v1/models/warmup", json={"model_id": model}
        )
        return WarmupResponse.model_validate(response_data)
