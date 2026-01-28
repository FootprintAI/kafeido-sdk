"""Async files resource for file uploads."""

from typing import BinaryIO, Optional, Union

from kafeido._http_client import AsyncHTTPClient
from kafeido.types.files import FileObject, FileList, DeletedFile


# Type alias for file inputs
FileTypes = Union[BinaryIO, bytes]


class AsyncFiles:
    """Async files resource for managing uploaded files."""

    def __init__(self, http_client: AsyncHTTPClient) -> None:
        """Initialize async files resource.

        Args:
            http_client: The async HTTP client to use for requests.
        """
        self._client = http_client

    async def create(
        self,
        *,
        file: FileTypes,
        purpose: str = "assistants",
    ) -> FileObject:
        """Upload a file asynchronously.

        Args:
            file: The file to upload (file object or bytes).
            purpose: The purpose of the file (e.g., "assistants", "fine-tune").

        Returns:
            FileObject with upload information.

        Example:
            >>> client = AsyncOpenAI(api_key="sk-...")
            >>> with open("audio.mp3", "rb") as f:
            ...     file_obj = await client.files.create(file=f, purpose="assistants")
            >>> print(file_obj.id, file_obj.filename)
        """
        # Prepare multipart upload
        files = {"file": file}
        data = {"purpose": purpose}

        response_data = await self._client.post(
            "/v1/audio/upload",
            data=data,
            files=files,
        )
        return FileObject.model_validate(response_data)

    async def list(
        self,
        *,
        purpose: Optional[str] = None,
    ) -> FileList:
        """List uploaded files asynchronously.

        Args:
            purpose: Optional filter by purpose.

        Returns:
            FileList containing uploaded files.

        Example:
            >>> client = AsyncOpenAI(api_key="sk-...")
            >>> files = await client.files.list()
            >>> for file in files.data:
            ...     print(file.filename, file.created_at)
        """
        params = {}
        if purpose:
            params["purpose"] = purpose

        response_data = await self._client.get("/v1/audio/files", params=params)
        return FileList.model_validate(response_data)

    async def retrieve(self, file_id: str) -> FileObject:
        """Retrieve information about a specific file asynchronously.

        Args:
            file_id: The file ID to retrieve.

        Returns:
            FileObject with file information.

        Example:
            >>> client = AsyncOpenAI(api_key="sk-...")
            >>> file = await client.files.retrieve("file-123")
            >>> print(file.filename, file.bytes)
        """
        response_data = await self._client.get(f"/v1/audio/files/{file_id}")
        return FileObject.model_validate(response_data)

    async def delete(self, file_id: str) -> DeletedFile:
        """Delete a file asynchronously.

        Args:
            file_id: The file ID to delete.

        Returns:
            DeletedFile confirmation.

        Example:
            >>> client = AsyncOpenAI(api_key="sk-...")
            >>> result = await client.files.delete("file-123")
            >>> print(result.deleted)  # True
        """
        response_data = await self._client.delete(f"/v1/audio/files/{file_id}")
        return DeletedFile.model_validate(response_data)
