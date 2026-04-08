"""Fine-tuning resource."""

from typing import Optional

from kafeido._http_client import HTTPClient
from kafeido.types.fine_tuning import (
    FineTuningEvent,
    FineTuningEventList,
    FineTuningHyperparameters,
    FineTuningJob,
    FineTuningJobList,
)


class FineTuningJobs:
    """Fine-tuning jobs endpoint."""

    def __init__(self, http_client: HTTPClient) -> None:
        self._client = http_client

    def create(
        self,
        *,
        model: str,
        training_file: str,
        validation_file: Optional[str] = None,
        suffix: Optional[str] = None,
        hyperparameters: Optional[FineTuningHyperparameters] = None,
    ) -> FineTuningJob:
        """Create a fine-tuning job.

        Args:
            model: Base model to fine-tune (e.g., "gpt-oss-20b").
            training_file: Storage key of uploaded JSONL training file.
            validation_file: Optional storage key of uploaded JSONL validation file.
            suffix: Custom suffix for the fine-tuned model name.
            hyperparameters: Override default hyperparameters.

        Returns:
            FineTuningJob with job details.
        """
        body: dict = {
            "model": model,
            "training_file": training_file,
        }
        if validation_file is not None:
            body["validation_file"] = validation_file
        if suffix is not None:
            body["suffix"] = suffix
        if hyperparameters is not None:
            body["hyperparameters"] = hyperparameters.model_dump(exclude_none=True)

        response_data = self._client.post("/v1/fine_tuning/jobs", json=body)
        return FineTuningJob.model_validate(response_data)

    def list(
        self,
        *,
        limit: Optional[int] = None,
        after: Optional[str] = None,
    ) -> FineTuningJobList:
        """List fine-tuning jobs.

        Args:
            limit: Max results (default: 20, max: 100).
            after: Cursor for pagination (job ID).

        Returns:
            FineTuningJobList with jobs and pagination info.
        """
        params = {}
        if limit is not None:
            params["limit"] = str(limit)
        if after is not None:
            params["after"] = after

        response_data = self._client.get("/v1/fine_tuning/jobs", params=params)
        return FineTuningJobList.model_validate(response_data)

    def retrieve(self, fine_tuning_job_id: str) -> FineTuningJob:
        """Retrieve a fine-tuning job by ID.

        Args:
            fine_tuning_job_id: The job ID to retrieve.

        Returns:
            FineTuningJob with job details.
        """
        response_data = self._client.get(
            f"/v1/fine_tuning/jobs/{fine_tuning_job_id}"
        )
        return FineTuningJob.model_validate(response_data)

    def cancel(self, fine_tuning_job_id: str) -> FineTuningJob:
        """Cancel a running fine-tuning job.

        Args:
            fine_tuning_job_id: The job ID to cancel.

        Returns:
            FineTuningJob with updated status.
        """
        response_data = self._client.post(
            f"/v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel", json={}
        )
        return FineTuningJob.model_validate(response_data)

    def list_events(
        self,
        fine_tuning_job_id: str,
        *,
        limit: Optional[int] = None,
        after: Optional[str] = None,
    ) -> FineTuningEventList:
        """List training progress events for a fine-tuning job.

        Args:
            fine_tuning_job_id: The job ID to list events for.
            limit: Max results (default: 20, max: 100).
            after: Cursor for pagination.

        Returns:
            FineTuningEventList with events and pagination info.
        """
        params = {}
        if limit is not None:
            params["limit"] = str(limit)
        if after is not None:
            params["after"] = after

        response_data = self._client.get(
            f"/v1/fine_tuning/jobs/{fine_tuning_job_id}/events", params=params
        )
        return FineTuningEventList.model_validate(response_data)


class FineTuning:
    """Fine-tuning resource."""

    def __init__(self, http_client: HTTPClient) -> None:
        self._client = http_client
        self._jobs = FineTuningJobs(http_client)

    @property
    def jobs(self) -> FineTuningJobs:
        """Access fine-tuning jobs endpoint."""
        return self._jobs
