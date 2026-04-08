"""Files resource for file uploads."""

import os
from typing import Any, BinaryIO, Optional, Union

import httpx

from kafeido._http_client import HTTPClient
from kafeido.types.files import FileObject, FileList, DeletedFile


# Type alias for file inputs
FileTypes = Union[BinaryIO, bytes]


class Files:
    """Files resource for managing uploaded files."""

    def __init__(self, http_client: HTTPClient) -> None:
        """Initialize files resource.

        Args:
            http_client: The HTTP client to use for requests.
        """
        self._client = http_client

    def create(
        self,
        *,
        file: FileTypes,
        purpose: str = "assistants",
    ) -> FileObject:
        """Upload a file via presigned URL (3-step flow).

        1. POST /v1/storage/prepare → get presigned PUT URL
        2. PUT to presigned URL → upload directly to cloud storage
        3. POST /v1/storage/confirm → confirm upload

        Args:
            file: The file to upload (file object or bytes).
            purpose: The purpose of the file (e.g., "assistants", "fine-tune").

        Returns:
            FileObject with upload information.

        Example:
            >>> client = OpenAI(api_key="sk-...")
            >>> with open("train.jsonl", "rb") as f:
            ...     file_obj = client.files.create(file=f, purpose="fine-tune")
            >>> print(file_obj.id, file_obj.filename)
        """
        # Read file content
        if isinstance(file, bytes):
            file_data = file
            filename = "upload.bin"
        else:
            file_data = file.read()
            filename = getattr(file, 'name', 'upload.bin')
            if filename:
                filename = os.path.basename(filename)

        # Map purpose string to proto enum
        purpose_map = {
            "fine-tune": "UPLOAD_PURPOSE_FINE_TUNING",
            "assistants": "UPLOAD_PURPOSE_ASSISTANTS",
            "transcription": "UPLOAD_PURPOSE_TRANSCRIPTION",
            "ocr": "UPLOAD_PURPOSE_OCR",
            "tts": "UPLOAD_PURPOSE_TTS",
        }
        proto_purpose = purpose_map.get(purpose, "UPLOAD_PURPOSE_UNSPECIFIED")

        # Step 1: Prepare upload
        prepare_resp = self._client.post("/v1/storage/prepare", json={
            "filename": filename,
            "contentType": "application/octet-stream",
            "purpose": proto_purpose,
        })

        upload_url = prepare_resp.get("uploadUrl", "")
        storage_key = prepare_resp.get("storageKey", "")
        file_id = prepare_resp.get("fileId", "")

        if not upload_url or not storage_key:
            raise RuntimeError(f"PrepareUpload failed: {prepare_resp}")

        # Step 2: Upload to presigned URL (direct to cloud storage)
        put_resp = httpx.put(
            upload_url,
            content=file_data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=300,
        )
        if not put_resp.is_success:
            raise RuntimeError(f"Upload to presigned URL failed: {put_resp.status_code} {put_resp.text}")

        # Step 3: Confirm upload
        confirm_resp = self._client.post("/v1/storage/confirm", json={
            "storageKey": storage_key,
            "fileId": file_id,
        })

        # Use storage_key as the file identifier — this is the full GCS path
        # that the dispatcher uses to generate download URLs
        return FileObject(
            id=storage_key,
            file_id=storage_key,
            filename=filename,
            bytes=len(file_data),
            purpose=purpose,
        )

    def list(
        self,
        *,
        purpose: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> FileList:
        """List uploaded files.

        Args:
            purpose: Optional filter by purpose.
            limit: Optional pagination limit (default: 100).
            offset: Optional offset for pagination.

        Returns:
            FileList containing uploaded files.

        Example:
            >>> client = OpenAI(api_key="sk-...")
            >>> files = client.files.list()
            >>> for file in (files.files or []):
            ...     print(file.filename)
        """
        params = {}
        if purpose:
            params["purpose"] = purpose
        if limit is not None:
            params["limit"] = str(limit)
        if offset is not None:
            params["offset"] = str(offset)

        response_data = self._client.get("/v1/audio/files", params=params)
        return FileList.model_validate(response_data)

    def retrieve(self, file_id: str) -> FileObject:
        """Retrieve information about a specific file.

        Args:
            file_id: The file ID to retrieve.

        Returns:
            FileObject with file information.

        Example:
            >>> client = OpenAI(api_key="sk-...")
            >>> file = client.files.retrieve("file-123")
            >>> print(file.filename, file.bytes)
        """
        response_data = self._client.get(f"/v1/audio/files/{file_id}")
        return FileObject.model_validate(response_data)

    def delete(self, file_id: str) -> DeletedFile:
        """Delete a file.

        Args:
            file_id: The file ID to delete.

        Returns:
            DeletedFile confirmation.

        Example:
            >>> client = OpenAI(api_key="sk-...")
            >>> result = client.files.delete("file-123")
            >>> print(result.deleted)  # True
        """
        response_data = self._client.delete(f"/v1/audio/files/{file_id}")
        return DeletedFile.model_validate(response_data)
