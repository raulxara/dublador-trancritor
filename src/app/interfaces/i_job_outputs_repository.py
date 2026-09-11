from typing import Protocol

from app.entities.dubbing_job.job_output_entity import JobOutputEntity


class IJobOutputsRepository(Protocol):
    def list(self, office_id: str, job_id: str) -> list[JobOutputEntity]: ...
