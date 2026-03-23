"""File upload types."""

from typing import List, Optional

from pydantic import BaseModel


class FileObject(BaseModel):
    """Uploaded file information."""

    file_id: Optional[str] = None
    file_size: Optional[int] = None
    uploaded_at: Optional[int] = None
    expires_in: Optional[int] = None
    filename: Optional[str] = None
    # Keep OpenAI-compatible fields for backward compatibility
    id: Optional[str] = None
    object: Optional[str] = None
    bytes: Optional[int] = None
    created_at: Optional[int] = None
    purpose: Optional[str] = None
    status: Optional[str] = None
    status_details: Optional[str] = None


class FileList(BaseModel):
    """List of uploaded files."""

    files: Optional[List[FileObject]] = None
    total: Optional[int] = None
    total_size: Optional[int] = None
    # Keep OpenAI-compatible fields for backward compatibility
    object: Optional[str] = None
    data: Optional[List[FileObject]] = None


class DeletedFile(BaseModel):
    """Deleted file confirmation."""

    success: Optional[bool] = None
    message: Optional[str] = None
    # Keep OpenAI-compatible fields for backward compatibility
    id: Optional[str] = None
    object: Optional[str] = None
    deleted: Optional[bool] = None
