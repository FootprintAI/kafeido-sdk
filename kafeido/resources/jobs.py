"""Jobs resource."""

from typing import Optional

from kafeido._http_client import HTTPClient
from kafeido.types.jobs import JobDetail, RequestProgress


class Jobs:
    """Jobs resource for tracking async job status and progress."""

    def __init__(self, http_client: HTTPClient) -> None:
        self._client = http_client

    def retrieve(self, *, job_id: str) -> JobDetail:
        """Get the status and result of a job.

        Args:
            job_id: The job ID to look up.

        Returns:
            JobDetail with status, timestamps, and result/error.
        """
        response_data = self._client.get(f"/v1/jobs/{job_id}")
        return JobDetail.model_validate(response_data)

    def progress(
        self,
        *,
        request_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> RequestProgress:
        """Get unified request progress (warmup + job processing).

        Args:
            request_id: The request ID to check progress for.
            model_id: The model ID associated with the request.

        Returns:
            RequestProgress with warmup and job progress.
        """
        params = {}
        if request_id is not None:
            params["request_id"] = request_id
        if model_id is not None:
            params["model_id"] = model_id

        response_data = self._client.get("/v1/requests/progress", params=params)
        return RequestProgress.model_validate(response_data)
