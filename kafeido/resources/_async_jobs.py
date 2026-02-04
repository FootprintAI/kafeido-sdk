"""Async jobs resource."""

from typing import Optional

from kafeido._http_client import AsyncHTTPClient
from kafeido.types.jobs import JobDetail, RequestProgress


class AsyncJobs:
    """Async jobs resource for tracking async job status and progress."""

    def __init__(self, http_client: AsyncHTTPClient) -> None:
        self._client = http_client

    async def retrieve(self, *, job_id: str) -> JobDetail:
        """Get the status and result of a job asynchronously."""
        response_data = await self._client.get(f"/v1/jobs/{job_id}")
        return JobDetail.model_validate(response_data)

    async def progress(
        self,
        *,
        request_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> RequestProgress:
        """Get unified request progress asynchronously."""
        params = {}
        if request_id is not None:
            params["request_id"] = request_id
        if model_id is not None:
            params["model_id"] = model_id

        response_data = await self._client.get("/v1/requests/progress", params=params)
        return RequestProgress.model_validate(response_data)
