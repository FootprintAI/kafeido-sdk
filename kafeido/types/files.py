"""File upload types - OpenAI compatible."""

from typing import List, Literal, Optional

from pydantic import BaseModel


class FileObject(BaseModel):
    """Uploaded file information."""

    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: Optional[str] = None
    status_details: Optional[str] = None


class FileList(BaseModel):
    """List of uploaded files."""

    object: Literal["list"] = "list"
    data: List[FileObject]


class DeletedFile(BaseModel):
    """Deleted file confirmation."""

    id: str
    object: Literal["file"] = "file"
    deleted: bool = True
