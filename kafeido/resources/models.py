"""Models resource."""

from kafeido._http_client import HTTPClient
from kafeido.types.models import Model, ModelList


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
        response_data = self._client.get(f"/v1/models/{model}")
        return Model.model_validate(response_data)
