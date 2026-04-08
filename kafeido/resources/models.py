"""Models resource."""

from urllib.parse import quote

from kafeido._http_client import HTTPClient
from kafeido.types.models import Model, ModelList, ModelStatus, WarmupResponse


class Models:
    """Models resource for listing and retrieving model information."""

    def __init__(self, http_client: HTTPClient) -> None:
        """Initialize models resource.

        Args:
            http_client: The HTTP client to use for requests.
        """
        self._client = http_client

    def list(self) -> ModelList:
        """List available models.

        Returns:
            ModelList containing all available models.

        Example:
            >>> client = OpenAI(api_key="sk-...")
            >>> models = client.models.list()
            >>> for model in models.data:
            ...     print(model.id)
        """
        response_data = self._client.get("/v1/models")
        return ModelList.model_validate(response_data)

    def retrieve(self, model: str) -> Model:
        """Retrieve information about a specific model.

        Args:
            model: The model ID to retrieve (e.g., "gpt-oss-20b").

        Returns:
            Model information.

        Example:
            >>> client = OpenAI(api_key="sk-...")
            >>> model = client.models.retrieve("gpt-oss-20b")
            >>> print(model.id, model.owned_by)
        """
        response_data = self._client.get(f"/v1/models/{quote(model, safe='')}")
        return Model.model_validate(response_data)

    def status(self, model: str) -> ModelStatus:
        """Get the status of a model including cold start progress.

        Args:
            model: The model ID to check status for.

        Returns:
            ModelStatus with loading status and cold start progress.
        """
        response_data = self._client.get(f"/v1/models/{quote(model, safe='')}/status")
        return ModelStatus.model_validate(response_data)

    def warmup(self, *, model: str) -> WarmupResponse:
        """Warmup/prefetch a model to reduce cold start time.

        Args:
            model: The model ID to warm up.

        Returns:
            WarmupResponse indicating if model is already warm and ETA.
        """
        # Use "model" string field for ft:* names, "model_id" for standard models.
        # Server validates adapter ownership for ft:* names before warmup.
        if model.startswith("ft:"):
            body = {"model": model}
        else:
            body = {"model_id": model}
        response_data = self._client.post(
            "/v1/models/warmup", json=body
        )
        return WarmupResponse.model_validate(response_data)
